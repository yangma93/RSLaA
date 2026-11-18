# -*- coding: utf-8 -*-
"""RSLaA 第二部分 —— 脚本 3：交通参数抽取 + 协同感知评价 + 结构化数据整理.

流程
----
1) traffic —— 交通参数抽取 (每个交通场景只抽一次, 缓存为 traffic_params.json)
   个体比例不按 vType, 而按 agent mesh model 匹配到的模型分三类:
       VRU  : 匹配到 ped / cyclist / bicycle 类模型
       car  : 其余模型中 车长 <= 6 m   (小车)
       truck: 其余模型中 车长  > 6 m   (大车)
   并进一步提炼两个连续变量:
       ratio_car_truck = N_car / N_truck
       ratio_motor_vru = (N_car + N_truck) / N_vru
   同时输出交通量 veh/h 与平均车速 m/s (由前后点位移/时间计算), 全时段 + 逐窗.

2) eval —— 调用 rslaa.auto_evaluate (frame_step=2)
   每个交通场景: 先静态基线, 再按 3 个配置单元依次 9 条动态配置.
   每条配置的结果落到 Results/scene_XXX/traffic_YY/eval_<位姿名>/
   (iou_dev_series.csv / comp_summary.csv / evals_raw.csv / config.json).

3) summarize —— 汇总并计算 score, 生成分析数据集
   指标口径 (metric_version = 2, 详见 rslaa.common.frame_metrics):
       IoU     : 逐帧全部 GT 个体的 3D IoU 均值 (未感知个体计 0)
       Dev     : 逐帧仅 IoU>0 的可感知个体, 感知/GT 底面中心 XY 距离均值 (m)
       HA_diff : 逐帧仅 IoU>0 的可感知个体, 航向角绝对误差均值 (rad, pi/2 周期)
       MR      : 逐帧漏检率 = IoU<=0.15 的个体数 / 该帧 GT 个体数
       Comp    : 逐轨迹有效感知帧数 / 在场帧数 (IoU>0.15 视为有效感知)
   score = 0.4*IoU + 0.3*(1 - min(Dev,5)/5) + 0.3*Comp
   因变量 = 动态 - 同场景静态; 自变量含交通参数 / 静态环境参数 /
   位姿自由度 / 方法 (MPC / MARL 匹配 / MARL 错配) 等.
   输出 Results/summary/part2_runs.csv, part2_analysis.json,
        part2_summary_by_method.csv, part2_summary_by_config.csv

用法
----
    python -m rslaa.rslaa_stage3_evaluate all
    python -m rslaa.rslaa_stage3_evaluate traffic [--site scene_001] [--force]
    python -m rslaa.rslaa_stage3_evaluate eval    [--site scene_001] [--traffic traffic_01]
    python -m rslaa.rslaa_stage3_evaluate summarize
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rslaa.rslaa_stage1_preprocess import SITES  # noqa: E402  站点表唯一来源

from rslaa.paths import MODELS_DIR, REPO_ROOT  # noqa: E402

MODELS_DIR = str(MODELS_DIR)
RESULTS_DIR = os.environ.get("RSLAA_RESULTS", str(REPO_ROOT / "Results"))
PLAN_PATH = os.path.join(RESULTS_DIR, "plan", "experiment_plan.json")

PRESETS = ["hdl32", "vlp16"]
FRAME_STEP = 4                    # 评价一律 frame_step=2,现在为4
DEV_CLIP = 5.0                    # Dev 截断上限 (m)
SCORE_W = (0.4, 0.3, 0.3)         # IoU / (1-Dev) / Comp 权重
CAR_MAX_LEN = 6.0                 # 小车车长上限 (m)
WINDOWS = [5, 15, 30]

# 指标口径版本 (与 rslaa.common.METRIC_VERSION / DET_IOU_THR 保持一致;
# 此处本地定义, 避免在汇总进程中引入 lidarsim/GPU 依赖)
METRIC_VERSION = 2
DET_IOU_THR = 0.15
METRIC_COLUMNS = ["t", "window_idx", "iou_mean", "dev_mean", "ha_diff_mae",
                  "mr", "n_gt", "n_detected", "n_perceived", "n_matched"]

INIT_NAME = "lidar_pose_init.json"
STATIC_NAME = "lidar_pose_static.json"


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def score_of(iou: float, dev: float, comp: float) -> float:
    """综合感知性能得分 (与实验计划 4.4 节一致).

    Dev 新口径只统计 IoU>0 的可感知个体, 无定义时为 NaN/None —— 此时按
    DEV_CLIP 计 (即不得分), 避免"大面积漏检但可感知个体很准"被误判为高分;
    漏检本身由 IoU / Comp 反映.
    """
    d = float(DEV_CLIP) if dev is None or not np.isfinite(float(dev)) \
        else float(np.clip(float(dev), 0.0, DEV_CLIP))
    return (SCORE_W[0] * float(iou)
            + SCORE_W[1] * (1.0 - d / DEV_CLIP)
            + SCORE_W[2] * float(comp))


def eval_dirname(pose_name: str) -> str:
    stem = pose_name[:-5] if pose_name.endswith(".json") else pose_name
    return "eval_" + stem[len("lidar_pose_"):]


def tdir_of(results_dir: str, site_id: str, traffic: str) -> str:
    return os.path.join(results_dir, site_id, traffic)


def _site_by_id(site_id: str) -> dict:
    for s in SITES:
        if s["id"] == site_id:
            return s
    raise KeyError(site_id)


def _traj_file(site_id: str, k: int) -> str:
    s = _site_by_id(site_id)
    return os.path.join(s["traj_dir"], s["stem"].format(i=k) + ".txt")


def load_plan(results_dir: str = RESULTS_DIR) -> dict:
    p = os.path.join(results_dir, "plan", "experiment_plan.json")
    if not os.path.isfile(p):
        raise SystemExit(f"未找到规划文件 {p}, 请先运行脚本 2 的 plan 子命令")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_records(path: str) -> np.ndarray:
    rec = np.loadtxt(path, encoding="utf-8-sig")
    if rec.ndim == 1:
        rec = rec.reshape(1, -1)
    return rec


# ---------------------------------------------------------------------------
# 1) 交通参数
# ---------------------------------------------------------------------------
def vtype_class_map(traj_path: str) -> dict:
    """vType -> 网格模型 -> {car / truck / VRU} 分类表."""
    from lidarsim.meshes import ModelLibrary, _class_of
    from lidarsim.trajectory import Trajectories

    ml = ModelLibrary(MODELS_DIR)
    tr = Trajectories.load(traj_path)
    out = {}
    for vt, (L, W) in sorted(tr.vtype_dims().items()):
        mesh = ml.match(L, W)
        cls = _class_of(mesh.name)
        length = float(mesh.dims[0])
        if cls in ("ped", "cyclist", "bicycle"):
            group = "vru"
        elif length <= CAR_MAX_LEN:
            group = "car"
        else:
            group = "truck"
        out[str(int(vt))] = dict(
            mesh=mesh.name, mesh_class=cls, model_length_m=round(length, 3),
            model_width_m=round(float(mesh.dims[1]), 3),
            traj_length_m=round(float(L), 3), traj_width_m=round(float(W), 3),
            group=group)
    return out


def _period_stats(rec: np.ndarray, group_of_vtype: dict) -> dict:
    """给定时段记录, 统计交通量 / 平均车速 / 车型构成 (全向量化)."""
    if len(rec) == 0:
        return dict(n_vehicles=0, volume_veh_per_h=0.0, mean_speed_mps=0.0,
                    composition=dict(car=0.0, truck=0.0, vru=0.0),
                    ratio_car_truck=None, ratio_motor_vru=None,
                    mean_speed_kmh=0.0, by_vtype={})
    t, vid = rec[:, 0], rec[:, 3].astype(np.int64)
    vt = rec[:, 4].astype(np.int64)
    xy = rec[:, 6:8]

    order = np.lexsort((t, vid))
    vid_s, t_s, xy_s, vt_s = vid[order], t[order], xy[order], vt[order]
    uniq, start, counts = np.unique(vid_s, return_index=True, return_counts=True)
    last = start + counts - 1
    dt = t_s[last] - t_s[start]
    dist = np.linalg.norm(xy_s[last] - xy_s[start], axis=1)
    valid = dt >= 1.0
    speed = np.where(valid, dist / np.maximum(dt, 1e-6), np.nan)

    n = int(len(uniq))
    t0, t1 = float(t.min()), float(t.max())
    hours = max((t1 - t0 + 1.0) / 3600.0, 1e-6)
    mean_speed = float(np.nanmean(speed)) if np.isfinite(speed).any() else 0.0

    # 每个个体取首次出现时的 vType -> 分组
    vtype_first = vt_s[start]
    groups = np.array([group_of_vtype.get(str(int(v)), "car")
                       for v in vtype_first])
    cnt = {g: int((groups == g).sum()) for g in ("car", "truck", "vru")}
    motor = cnt["car"] + cnt["truck"]
    ratio_ct = (cnt["car"] / cnt["truck"]) if cnt["truck"] > 0 else None
    ratio_mv = (motor / cnt["vru"]) if cnt["vru"] > 0 else None

    by_vt = {}
    for v in np.unique(vtype_first):
        m = vtype_first == v
        by_vt[str(int(v))] = dict(
            group=group_of_vtype.get(str(int(v)), "car"),
            n=int(m.sum()),
            mean_speed_mps=round(float(np.nanmean(speed[m])), 4)
            if np.isfinite(speed[m]).any() else 0.0)

    return dict(
        n_vehicles=n,
        volume_veh_per_h=round(n / hours, 3),
        mean_speed_mps=round(mean_speed, 4),
        mean_speed_kmh=round(mean_speed * 3.6, 4),
        composition={g: round(cnt[g] / max(n, 1), 6) for g in cnt},
        composition_counts=cnt,
        ratio_car_truck=None if ratio_ct is None else round(ratio_ct, 6),
        ratio_motor_vru=None if ratio_mv is None else round(ratio_mv, 6),
        by_vtype=by_vt,
        t_range=[t0, t1],
    )


def extract_traffic_params(traj_path: str, group_of_vtype: dict) -> dict:
    rec = _load_records(traj_path)
    times = np.unique(rec[:, 0])
    full = _period_stats(rec, group_of_vtype)

    windows = {}
    for wmin in WINDOWS:
        win_s = wmin * 60
        n_win = int((times[-1] - times[0] + 1e-6) // win_s)
        items = []
        for k in range(n_win):
            a = times[0] + k * win_s
            b = a + win_s - 1
            sel = rec[(rec[:, 0] >= a - 1e-6) & (rec[:, 0] <= b + 1e-6)]
            st = _period_stats(sel, group_of_vtype)
            st["index"] = k + 1
            st["t_range"] = [float(a), float(b)]
            items.append(st)
        windows[str(wmin)] = items
    return dict(traj_file=os.path.abspath(traj_path), full=full, windows=windows)


def cmd_traffic(args):
    plan = load_plan(args.results_dir)
    sites = _select(plan, args)
    n_done = n_skip = 0
    for site in sites:
        for tr in site["traffic"]:
            if tr.get("status", "ok") != "ok":
                log(f"  [跳过] {site['id']}/{tr['traffic']}: "
                    f"{tr.get('skip_reason', '')}")
                continue
            tdir = tdir_of(args.results_dir, site["id"], tr["traffic"])
            os.makedirs(tdir, exist_ok=True)
            out = os.path.join(tdir, "traffic_params.json")
            if os.path.isfile(out) and not args.force:
                n_skip += 1
                log(f"  跳过 (已存在): {site['id']}/{tr['traffic']}")
                continue
            log(f"[交通参数] {site['id']}/{tr['traffic']} ...")
            gmap = vtype_class_map(tr["traj_file"])
            gstr = {k: v["group"] for k, v in gmap.items()}
            params = extract_traffic_params(tr["traj_file"], gstr)
            params.update(site=site["id"], site_name=site["name"],
                          family=site["family"], traffic=tr["traffic"],
                          vtype_class=gmap,
                          generated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            with open(out, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            f0 = params["full"]
            log(f"  {f0['n_vehicles']} 车, {f0['volume_veh_per_h']:.1f} veh/h, "
                f"{f0['mean_speed_kmh']:.1f} km/h, "
                f"car/truck/VRU={f0['composition']}, "
                f"car/truck={f0['ratio_car_truck']}, "
                f"motor/VRU={f0['ratio_motor_vru']}")
            _write_road_params(args.results_dir, site, tdir)
            n_done += 1
    log(f"[交通参数] 新提取 {n_done}, 跳过 {n_skip}")
    return 0


def _write_road_params(results_dir: str, site: dict, tdir: str):
    """把脚本 1 的静态环境参数摘到交通场景目录 (便于一一对应)."""
    sp_path = os.path.join(results_dir, "static_params", site["id"] + ".json")
    if not os.path.isfile(sp_path):
        return
    with open(sp_path, "r", encoding="utf-8") as f:
        sp = json.load(f)
    out = dict(site=site["id"], site_name=site["name"], family=site["family"],
               env_path=sp.get("env_path"), road=sp.get("road", {}),
               roadside_obstacle=sp.get("roadside_obstacle", {}),
               missing=sp.get("notes", []))
    with open(os.path.join(tdir, "road_params.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 2) 评价
# ---------------------------------------------------------------------------
def _pose_ok_result(edir: str) -> bool:
    """评估结果是否可直接复用: 文件齐全且为当前指标口径 (metric_version=2).

    旧口径 (只有 iou_mean/dev_mean/n_matched/n_gt 四列) 的结果会被自动重跑,
    避免新旧口径混在同一份分析数据里.
    """
    cfg_p = os.path.join(edir, "config.json")
    ser_p = os.path.join(edir, "iou_dev_series.csv")
    if not (os.path.isfile(cfg_p) and os.path.isfile(ser_p)):
        return False
    if _csv_header(ser_p) != METRIC_COLUMNS:
        return False
    cfg = _load_json(cfg_p) or {}
    return int(cfg.get("metric_version", 1)) == METRIC_VERSION


def _csv_header(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return next(csv.reader(f), [])
    except (OSError, StopIteration):
        return []


def _run_eval(pose_path: str, edir: str, site: dict, tr: dict, args) -> tuple:
    cmd = [sys.executable, "-m", "rslaa.auto_evaluate",
           "--pose-seq", pose_path, "--traj", tr["traj_file"],
           "--env", site["env"], "--models", MODELS_DIR,
           "--out", edir, "--device", args.device,
           "--frame-step", str(args.frame_step),
           "--workers", str(args.workers), "--quiet"]
    log_path = os.path.join(args.results_dir, "logs", site["id"], tr["traffic"],
                            os.path.basename(edir) + ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        r = subprocess.run(cmd, cwd=_REPO, stdout=lf, stderr=subprocess.STDOUT,
                           env=env)
    return r.returncode == 0 and _pose_ok_result(edir), time.time() - t0, log_path


def _eval_targets(tr: dict) -> list:
    """(pose_stem, method, w, m, mismatch, pos, ang) 列表, 静态在最前."""
    out = [(tr["static_stem"], "static", None, None, 0, None, None)]
    for u in tr["units"]:
        for run in u["runs"]:
            method = "mpc" if run["method"] == "mpc" else (
                "marl_match" if run["mismatch"] == 0 else "marl_mismatch")
            out.append((run["stem"], method, u["w"], run["m"],
                        run["mismatch"], u["pos_m"], u["ang_deg"]))
    return out


def cmd_eval(args):
    plan = load_plan(args.results_dir)
    sites = _select(plan, args)
    for site in sites:
        for tr in site["traffic"]:
            if tr.get("status", "ok") != "ok":
                continue
            if args.traffic and tr["traffic"] not in args.traffic.split(","):
                continue
            tdir = tdir_of(args.results_dir, site["id"], tr["traffic"])
            log(f"\n[评价] {site['id']}/{tr['traffic']}")
            targets = _eval_targets(tr)
            if args.methods:
                want = set(args.methods.split(","))
                targets = [t for t in targets if t[1] in want]
            for (stem, method, w, m, mis, pos, ang) in targets:
                pose_path = os.path.join(tdir, stem)
                edir = os.path.join(tdir, eval_dirname(stem))
                if not os.path.isfile(pose_path):
                    log(f"  缺少位姿文件, 跳过: {stem}")
                    continue
                if _pose_ok_result(edir) and not args.force:
                    log(f"  跳过 (已评价): {os.path.basename(edir)}")
                    continue
                if args.dry_run:
                    log(f"  [dry-run] {edir}")
                    continue
                ok, dt, lp = _run_eval(pose_path, edir, site, tr, args)
                log(f"  {os.path.basename(edir):<48} "
                    f"{'OK' if ok else 'FAIL'}  {dt / 60:.2f} min")
                if not ok:
                    log(f"    日志: {lp}")
    return 0


# ---------------------------------------------------------------------------
# 3) 汇总
# ---------------------------------------------------------------------------
def read_eval_metrics(edir: str) -> dict:
    """从评估目录读指标 (逐帧值的均值; 无定义的帧不计入平均).

    IoU     : 逐帧 IoU 均值 (全部 GT 个体, 未感知计 0)
    Dev     : 逐帧"仅 IoU>0 个体"的底面中心 XY 距离均值的平均 (m)
    HA_diff : 逐帧"仅 IoU>0 个体"的航向角绝对误差均值的平均 (rad)
    MR      : 逐帧漏检率的平均 (IoU<=0.15 视为漏检)
    Comp    : 逐轨迹有效感知帧数/在场帧数 的均值 (IoU>0.15 视为有效感知)
    """
    sp = os.path.join(edir, "iou_dev_series.csv")
    if not os.path.isfile(sp):
        return None
    data = np.genfromtxt(sp, delimiter=",", names=True, encoding="utf-8-sig")
    if data.size == 0:
        return None
    names = set(data.dtype.names or ())
    cfg = _load_json(os.path.join(edir, "config.json")) or {}

    def col_mean(name):
        """缺失列 -> None; 忽略 NaN (无定义帧)."""
        if name not in names:
            return None
        v = np.asarray(data[name], np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else None

    iou = col_mean("iou_mean")
    dev = col_mean("dev_mean")
    ha = col_mean("ha_diff_mae")
    mr = col_mean("mr")
    n_frames = int(len(data))
    try:
        n_windows = int(len(np.unique(data["window_idx"])))
    except Exception:
        n_windows = 1

    # 回退: 旧口径结果没有 ha_diff_mae / mr 列, 只有全局汇总
    if iou is None:
        iou = cfg.get("global_iou_mean")
    if dev is None:
        dev = cfg.get("global_dev_mean")
    if ha is None:
        ha = cfg.get("global_ha_diff_mae")
    if mr is None:
        mr = cfg.get("global_mr")
    iou = 0.0 if iou is None else float(iou)

    comp = None
    cp = os.path.join(edir, "comp_summary.csv")
    if os.path.isfile(cp):
        with open(cp, "r", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] == "global_mean_comp":
                    comp = float(row[1])
    if comp is None:
        comp = cfg.get("global_comp", 0.0)
    comp = 0.0 if comp is None else float(comp)
    return dict(iou=iou, dev=None if dev is None else float(dev),
                ha=None if ha is None else float(ha),
                mr=None if mr is None else float(mr),
                comp=comp, score=score_of(iou, dev, comp),
                n_frames=n_frames, n_windows=n_windows,
                metric_version=int(cfg.get("metric_version", 1)))


def _traffic_mean(tp: dict, w) -> dict:
    """按窗长 w 取窗口平均; 静态 (w=None) 用全时段."""
    if not tp:
        return {}
    if w is None:
        base = tp["full"]
    else:
        items = tp.get("windows", {}).get(str(int(w)), [])
        if not items:
            base = tp["full"]
        else:
            keys = ["volume_veh_per_h", "mean_speed_mps", "mean_speed_kmh"]
            base = {k: float(np.mean([it[k] for it in items])) for k in keys}
            comp = {}
            for g in ("car", "truck", "vru"):
                comp[g] = float(np.mean([it["composition"][g] for it in items]))
            base["composition"] = comp
            base["ratio_car_truck"] = _safe_mean(
                [it["ratio_car_truck"] for it in items])
            base["ratio_motor_vru"] = _safe_mean(
                [it["ratio_motor_vru"] for it in items])
    return dict(
        volume_veh_per_h=round(base["volume_veh_per_h"], 3),
        mean_speed_mps=round(base["mean_speed_mps"], 4),
        mean_speed_kmh=round(base.get("mean_speed_kmh",
                                      base["mean_speed_mps"] * 3.6), 4),
        composition_car=round(base["composition"]["car"], 6),
        composition_truck=round(base["composition"]["truck"], 6),
        composition_vru=round(base["composition"]["vru"], 6),
        ratio_car_truck=None if base.get("ratio_car_truck") is None
        else round(float(base["ratio_car_truck"]), 6),
        ratio_motor_vru=None if base.get("ratio_motor_vru") is None
        else round(float(base["ratio_motor_vru"]), 6),
    )


def _safe_mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _r(v, nd: int = 6):
    """数值四舍五入; None/NaN -> None (CSV 记空)."""
    if v is None:
        return None
    try:
        return round(float(v), nd) if np.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return None


def _diff(a, b):
    """差值 a-b; 任一侧无定义则 None (不做缺省填充, 避免伪差值)."""
    if a is None or b is None:
        return None
    return _r(float(a) - float(b))


def cmd_summarize(args):
    plan = load_plan(args.results_dir)
    rows = []
    for site in plan["sites"]:
        rp = _load_json(os.path.join(args.results_dir, "static_params",
                                     site["id"] + ".json"))
        road = (rp or {}).get("road", {}) or {}
        obs = (rp or {}).get("roadside_obstacle", {}) or {}
        for tr in site["traffic"]:
            if tr.get("status", "ok") != "ok":
                continue
            tdir = tdir_of(args.results_dir, site["id"], tr["traffic"])
            tp = _load_json(os.path.join(tdir, "traffic_params.json"))
            static_m = read_eval_metrics(
                os.path.join(tdir, eval_dirname(tr["static_stem"])))
            for (stem, method, w, m, mis, pos, ang) in _eval_targets(tr):
                edir = os.path.join(tdir, eval_dirname(stem))
                met = read_eval_metrics(edir)
                row = dict(
                    run_id=f"{site['id']}/{tr['traffic']}/{stem[:-5]}",
                    site=site["id"], site_name=site["name"],
                    family=site["family"], traffic=tr["traffic"],
                    method=method, w=w, m=m, mismatch=int(mis),
                    pos_m=pos, ang_deg=ang,
                    n_lidar=len(PRESETS), presets="+".join(PRESETS),
                    frame_step=args.frame_step,
                    pose_file=os.path.join(tdir, stem),
                    eval_dir=edir,
                    road_width_m=road.get("road_width_m"),
                    lane_count=road.get("lane_count"),
                    lane_width_m=road.get("lane_width_m"),
                    radius_min_m=road.get("radius_min_m"),
                    curved_fraction=road.get("curved_fraction"),
                    is_curved=road.get("is_curved"),
                    obstacle_mean_height_m=obs.get("mean_height_m"),
                    obstacle_long_std_m=obs.get("longitudinal_std_m"),
                    obstacle_count=obs.get("count"),
                )
                row.update(_traffic_mean(tp, w))
                if met is None:
                    row.update(eval_ok=0, missing_reason="缺少评估结果",
                               iou=None, dev=None, ha_diff=None, mr=None,
                               comp=None, score=None, metric_version=None,
                               d_iou=None, d_dev=None, d_ha_diff=None,
                               d_mr=None, d_comp=None, d_score=None,
                               n_frames=0, n_windows=0)
                else:
                    row.update(eval_ok=1, missing_reason="")
                    row.update(iou=_r(met["iou"]), dev=_r(met["dev"]),
                               ha_diff=_r(met["ha"]), mr=_r(met["mr"]),
                               comp=_r(met["comp"]), score=_r(met["score"]),
                               metric_version=met["metric_version"],
                               n_frames=met["n_frames"],
                               n_windows=met["n_windows"])
                    if static_m is None:
                        row.update(d_iou=None, d_dev=None, d_ha_diff=None,
                                   d_mr=None, d_comp=None, d_score=None)
                    else:
                        row.update(
                            d_iou=_diff(met["iou"], static_m["iou"]),
                            d_dev=_diff(met["dev"], static_m["dev"]),
                            d_ha_diff=_diff(met["ha"], static_m["ha"]),
                            d_mr=_diff(met["mr"], static_m["mr"]),
                            d_comp=_diff(met["comp"], static_m["comp"]),
                            d_score=_diff(met["score"], static_m["score"]))
                if static_m is not None:
                    row.update(static_iou=_r(static_m["iou"]),
                               static_dev=_r(static_m["dev"]),
                               static_ha_diff=_r(static_m["ha"]),
                               static_mr=_r(static_m["mr"]),
                               static_comp=_r(static_m["comp"]),
                               static_score=_r(static_m["score"]))
                else:
                    row.update(static_iou=None, static_dev=None,
                               static_ha_diff=None, static_mr=None,
                               static_comp=None, static_score=None)
                rows.append(row)

    sdir = os.path.join(args.results_dir, "summary")
    os.makedirs(sdir, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    csv_path = os.path.join(sdir, "part2_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)
    with open(os.path.join(sdir, "part2_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            metric_version=METRIC_VERSION,
            det_iou_thr=DET_IOU_THR,
            score_formula="0.4*IoU + 0.3*(1-min(Dev,5)/5) + 0.3*Comp",
            metric_def=dict(
                iou="逐帧全部 GT 个体 3D IoU 均值 (未感知计 0)",
                dev="逐帧仅 IoU>0 个体的感知/GT 底面中心 XY 距离均值 (m), 无定义帧不计",
                ha_diff="逐帧仅 IoU>0 个体的航向角绝对误差均值 (rad, pi/2 周期)",
                mr="逐帧 IoU<=0.15 的个体数 / GT 个体数",
                comp="逐轨迹 IoU>0.15 的有效感知帧数 / 在场帧数",
            ),
            frame_step=args.frame_step,
            dependent_variable="dynamic - static (score / IoU / Dev / HA_diff / MR / Comp)",
            n_rows=len(rows), rows=rows), f, ensure_ascii=False, indent=2)

    _write_group_summary(rows, sdir, "method", "part2_summary_by_method.csv")
    _write_group_summary(rows, sdir,
                         ["method", "w", "pos_m", "ang_deg"],
                         "part2_summary_by_config.csv")
    _write_group_summary(rows, sdir, ["site", "traffic"],
                         "part2_summary_by_scene.csv")

    ok = sum(1 for r in rows if r["eval_ok"])
    n_legacy = sum(1 for r in rows
                   if r["eval_ok"] and r.get("metric_version") != METRIC_VERSION)
    log(f"[汇总] {len(rows)} 行, 已评价 {ok}; 输出目录 {sdir}")
    if n_legacy:
        log(f"[警告] {n_legacy} 行仍为旧指标口径 (metric_version=1), "
            f"请用 eval --force 重跑后再分析")
    log(f"  {csv_path}")
    return 0


def _load_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_group_summary(rows: list, sdir: str, keys, fname: str):
    import collections
    if isinstance(keys, str):
        keys = [keys]
    groups = collections.OrderedDict()
    for r in rows:
        if not r.get("eval_ok"):
            continue
        k = tuple(r.get(x) for x in keys)
        groups.setdefault(k, []).append(r)

    def _mean(g, col):
        v = [x[col] for x in g if x.get(col) is not None]
        return round(float(np.mean(v)), 6) if v else ""

    out = []
    for k, g in groups.items():
        d = {key: val for key, val in zip(keys, k)}
        d.update(n=len(g),
                 iou=_mean(g, "iou"), dev=_mean(g, "dev"),
                 ha_diff=_mean(g, "ha_diff"), mr=_mean(g, "mr"),
                 comp=_mean(g, "comp"), score=_mean(g, "score"),
                 static_score=_mean(g, "static_score"),
                 d_score=_mean(g, "d_score"),
                 d_iou=_mean(g, "d_iou"), d_dev=_mean(g, "d_dev"),
                 d_ha_diff=_mean(g, "d_ha_diff"), d_mr=_mean(g, "d_mr"),
                 d_comp=_mean(g, "d_comp"))
        out.append(d)
    if not out:
        return
    cols = list(keys) + ["n", "iou", "dev", "ha_diff", "mr", "comp", "score",
                         "static_score", "d_score", "d_iou", "d_dev",
                         "d_ha_diff", "d_mr", "d_comp"]
    with open(os.path.join(sdir, fname), "w", newline="",
              encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(out)


# ---------------------------------------------------------------------------
def _select(plan: dict, args) -> list:
    want = None
    if getattr(args, "site", ""):
        want = [s.strip() for s in args.site.split(",") if s.strip()]
    if want:
        sites = [s for s in plan["sites"] if s["id"] in want or s["name"] in want]
        if not sites:
            raise SystemExit(f"未匹配到地点: {want}")
        return sites
    return plan["sites"]


def cmd_all(args):
    rc = cmd_traffic(args)
    rc |= cmd_eval(args)
    rc |= cmd_summarize(args)
    return rc


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="RSLaA 第二部分: 交通参数 / 协同感知评价 / 结构化汇总",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p):
        p.add_argument("--site", default="", help="单个/多个地点, 逗号分隔")
        p.add_argument("--sites", default="")
        p.add_argument("--results-dir", default=RESULTS_DIR)
        p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
        p.add_argument("--frame-step", type=int, default=FRAME_STEP)
        p.add_argument("--workers", type=int, default=1,
                       help="评价阶段的帧级并行仿真进程数 (>1 启用多进程 "
                            "GPU 并行, 结果与串行逐位一致)")
        p.add_argument("--force", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        return p

    p1 = _common(sub.add_parser("traffic", help="抽取交通参数"))
    p2 = _common(sub.add_parser("eval", help="运行 auto_evaluate"))
    p2.add_argument("--traffic", default="", help="如 traffic_01,traffic_02")
    p2.add_argument("--methods", default="",
                    help="static,mpc,marl_match,marl_mismatch 的子集")
    p3 = _common(sub.add_parser("summarize", help="汇总与分析数据集"))
    _common(sub.add_parser("all", help="traffic + eval + summarize"))

    return ap.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args(argv)
    if args.cmd == "traffic":
        return cmd_traffic(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "summarize":
        return cmd_summarize(args)
    if args.cmd == "all":
        return cmd_all(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

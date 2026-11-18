# -*- coding: utf-8 -*-
"""RSLaA 第二部分 —— 脚本 2：实验配置规划 + 动态试验自动运行.

------
1) 目录与命名
   Results/scene_XXX/traffic_YY/
       lidar_pose_init.json      每个交通场景唯一, MPC / MARL 共用
       lidar_pose_static.json    静态基线 (单窗口, 无 window_length_sec / t_range)
       lidar_pose_mpc_w{W}_pos{P}_a{A}.json
       lidar_pose_marl_w{W}_pos{P}_a{A}_m{M}.json
   (评估目录 eval_* 由脚本 3 生成)

2) 试验配置规划 (冻结为 JSON, 便于跨机器复用)
   Results/plan/experiment_plan.json + design_manifest.csv + balance_report.json
   * 10 地点 × 9 交通场景 = 90 个场景
   * 每个场景 3 个配置单元, w 覆盖 5/15/30 (单元内互不重复)
   * 自由度组合 (pos, ang) 在 6 种水平上轮换, 全场地均衡且与 w 解耦
   * 每单元 1 条 MPC + 2 条 MARL (匹配 m=w / 错配 m≠w) = 9 条动态
   * 传感器固定 2 台: hdl32 + vlp16 (与现有 3 个智能体模型严格一致)
   * 初始位姿约束: 离地 6~10 m 随机、0.6 m 球半径内无静态障碍物、
     横向不偏离交通流中心线 (默认 ≤6 m)、两台相距 8~45 m 且优先分置道路两侧
     动态车流净空 = 离地高度 - 5.0 m (本数据最高车辆模型 4.96 m) ≥ 1 m

3) 动态试验运行
   python -m rslaa.rslaa_stage2_experiment run [--site scene_001]
   逐地点 → 逐交通场景 → 逐配置单元 (MPC → MARL 匹配 → MARL 错配)
   一个地点的 9 个交通场景跑完后询问是否继续下一个地点 (Y/N);
   --yes 用于无人值守批量运行; 已存在的位姿文件默认跳过 (断点续跑).

用法
----
    python -m rslaa.rslaa_stage2_experiment plan  [--sites scene_001,scene_002] [--force]
    python -m rslaa.rslaa_stage2_experiment run   [--site scene_001] [--traffic traffic_01]
    python -m rslaa.rslaa_stage2_experiment run   --yes              # 10 地点全量
    python -m rslaa.rslaa_stage2_experiment status
"""
from __future__ import annotations

import argparse
import csv
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

from rslaa.rslaa_stage1_preprocess import (  # noqa: E402
    SITES as SITES_CFG, traffic_indices,
)


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
from rslaa.paths import CHECKPOINTS_DIR, MODELS_DIR, REPO_ROOT  # noqa: E402

MODELS_DIR = str(MODELS_DIR)
RESULTS_DIR = os.environ.get("RSLAA_RESULTS", str(REPO_ROOT / "Results"))

CKPT_ROOT = str(CHECKPOINTS_DIR / "n2_hdl32-vlp16")
CHECKPOINTS = {
    5: os.path.join(CKPT_ROOT, "w05", "policy_best.pt"),
    15: os.path.join(CKPT_ROOT, "w15", "policy_best.pt"),
    30: os.path.join(CKPT_ROOT, "w30", "policy_best.pt"),
}

# 只有 2 传感器模型 (hdl32 + vlp16), 顺序必须与训练时一致
PRESETS = ["hdl32", "vlp16"]
N_LIDAR = 2

WINDOWS = [5, 15, 30]                       # 试验窗长 w (min)
COMBOS = [(0.2, 5), (0.2, 10), (0.2, 15),   # (pos m, ang deg), 6 种水平
          (0.5, 5), (0.5, 10), (0.5, 15)]
METRIC = "J_mv"                             # 默认指标: 与感知指标正相关 (最大化)
                                            # MPC 优化指标 = MARL 训练奖励指标
FRAME_STEP = 2                              # 位姿生成阶段的 POG 帧步长
# 1->2: 单窗 POG 建图 20.5s->10.4s, POG 体素一致率 99.5%, J_dyn 变化 <0.01%
# (scene_002/traffic_01 w=5 A/B). 需要与旧结果严格对齐时用 --frame-step 1.
DESIGN_VERSION = "RSLaA-part2-v1.0"

POSE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
POSE_UNIT = "rad"

# 初始位姿几何约束
Z_ABOVE_GROUND = (6.0, 10.0)     # 离地高度随机区间 (m)
STATIC_CLEARANCE = 0.6           # 静态障碍物球半径 (m)
MAX_VEH_HEIGHT = 5.0             # 车辆模型最大高度 (m), 用于动态净空校核
MIN_PAIR_DIST = 8.0              # 两台 LiDAR 最小间距 (m)
MAX_PAIR_DIST = 45.0             # 最大间距 (m)

SITES = SITES_CFG            # 站点表唯一来源: 脚本 1 (含 skip 配置)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 命名规则 (与实验计划 3.7 节一致)
# ---------------------------------------------------------------------------
def tag(w: int, pos: float, ang: int) -> str:
    """w5 / pos05 / a10 形式的配置标记."""
    return f"w{int(w)}_pos{int(round(pos * 10)):02d}_a{int(ang):02d}"


def mpc_name(w, pos, ang) -> str:
    return f"lidar_pose_mpc_{tag(w, pos, ang)}.json"


def marl_name(w, pos, ang, m) -> str:
    return f"lidar_pose_marl_{tag(w, pos, ang)}_m{int(m)}.json"


def eval_dirname(pose_name: str) -> str:
    """lidar_pose_xxx.json -> eval_xxx."""
    stem = pose_name[:-5] if pose_name.endswith(".json") else pose_name
    return "eval_" + stem[len("lidar_pose_"):]


STATIC_NAME = "lidar_pose_static.json"
INIT_NAME = "lidar_pose_init.json"


def traffic_dir(results_dir, site_id, traffic_id) -> str:
    return os.path.join(results_dir, site_id, traffic_id)


def traj_file(site: dict, k: int) -> str:
    return os.path.join(site["traj_dir"], site["stem"].format(i=k) + ".txt")


# ---------------------------------------------------------------------------
# 轨迹几何工具 (与脚本 1 同口径)
# ---------------------------------------------------------------------------
def _load_traj_records(path: str) -> np.ndarray:
    rec = np.loadtxt(path, encoding="utf-8-sig")
    if rec.ndim == 1:
        rec = rec.reshape(1, -1)
    return rec


def _longest_track_centerline(records: np.ndarray, ds: float = 0.5):
    """取采样点最多的个体轨迹, 去重并按弧长重采样为参考中心线."""
    t, vid = records[:, 0], records[:, 3].astype(np.int64)
    fx, fy = records[:, 6], records[:, 7]
    best, best_n = None, -1
    for v in np.unique(vid):
        m = vid == v
        if m.sum() > best_n:
            best_n, best = int(m.sum()), v
    m = vid == best
    order = np.argsort(t[m], kind="stable")
    pts = np.stack([fx[m][order], fy[m][order]], 1)
    keep = np.ones(len(pts), bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3
    pts = pts[keep]
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / ds), 2)
    sq = np.linspace(0, s[-1], n)
    cl = np.stack([np.interp(sq, s, pts[:, 0]),
                   np.interp(sq, s, pts[:, 1])], 1)
    return dict(points=cl, s=sq, length=float(s[-1]), ref_vid=int(best))


def _centerline_frame(cl: dict):
    p = cl["points"]
    seg = np.gradient(p, axis=0)
    nrm = np.linalg.norm(seg, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    tang = seg / nrm
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    return p, tang, normal


def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _overlap_fraction(xy: np.ndarray, env_points: np.ndarray,
                      pad: float = 20.0) -> float:
    """轨迹点落在环境点云水平范围内的比例 (用于发现坐标系不一致)."""
    x0, x1 = env_points[:, 0].min(), env_points[:, 0].max()
    y0, y1 = env_points[:, 1].min(), env_points[:, 1].max()
    m = ((xy[:, 0] > x0 - pad) & (xy[:, 0] < x1 + pad) &
         (xy[:, 1] > y0 - pad) & (xy[:, 1] < y1 + pad))
    return float(m.mean())


# ---------------------------------------------------------------------------
# 初始位姿生成
# ---------------------------------------------------------------------------
def generate_init_poses(site: dict, traj_path: str, road_half_m: float,
                        env, gm, rng: np.random.Generator) -> dict:
    """在交通流走廊上方生成 2 个满足全部约束的初始位姿.

    返回 {"poses": (N,6), "meta": {...}}; poses 为用户序 [x y z roll pitch yaw],
    角度为弧度, z 为绝对高程.
    """
    rec = _load_traj_records(traj_path)
    cl = _longest_track_centerline(rec)
    if cl is None:
        raise RuntimeError(f"无法从轨迹构建中心线: {traj_path}")
    p, tang, normal = _centerline_frame(cl)

    # ---- 静态障碍物 KD 树 (非地面点) ----
    pts = np.asarray(env.points, np.float64)
    h_all = pts[:, 2] - gm.height(pts[:, :2])
    obs = pts[h_all > 0.8]
    obs = _voxel_downsample(obs, 0.25)
    from scipy.spatial import cKDTree
    tree = cKDTree(obs) if len(obs) else None

    # ---- 候选点: 中心线上按弧长采样 × 横向偏移 ----
    half = float(np.clip(road_half_m, 2.5, 6.0))
    offs = np.array([0.0, 0.5 * half, -0.5 * half, half, -half])
    ds = float(cl["s"][1] - cl["s"][0]) if len(cl["s"]) > 1 else 0.5
    step = max(1, int(round(3.0 / ds)))
    idx = np.arange(3, max(len(p) - 3, 4), step)
    if idx.size == 0:
        idx = np.array([len(p) // 2])

    xy = (p[idx][:, None, :] + offs[None, :, None] * normal[idx][:, None, :])
    xy = xy.reshape(-1, 2)
    n_cand = len(xy)
    hg = gm.height(xy)
    dz = rng.uniform(Z_ABOVE_GROUND[0], Z_ABOVE_GROUND[1], size=n_cand)
    z_abs = hg + dz

    # 环境点云 xy 包围盒内 (避免地面模型外插)
    x0, x1 = pts[:, 0].min() + 1.0, pts[:, 0].max() - 1.0
    y0, y1 = pts[:, 1].min() + 1.0, pts[:, 1].max() - 1.0
    inside = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)

    # 静态遮挡: 球半径 0.6 m
    xyz = np.stack([xy[:, 0], xy[:, 1], z_abs], 1)
    if tree is not None:
        dmin, _ = tree.query(xyz, k=1, workers=-1)
    else:
        dmin = np.full(n_cand, np.inf)
    ok = inside & (dmin >= STATIC_CLEARANCE)

    # 动态净空: 车辆最高 5.0 m, 离地高度须留 >= 1.0 m
    dyn_margin = dz - MAX_VEH_HEIGHT

    cand_idx = np.flatnonzero(ok)
    relaxation = "none"
    if cand_idx.size < 2:               # 放宽: 优先保留最远离障碍物的点
        ok2 = inside
        cand_idx = np.flatnonzero(ok2)
        cand_idx = cand_idx[np.argsort(-dmin[cand_idx])][:200]
        relaxation = "static_clearance_relaxed"
    if cand_idx.size < 2:
        raise RuntimeError(f"{site['id']}/{os.path.basename(traj_path)}: "
                           f"无可用初始位姿 (有效候选 {cand_idx.size})")

    xx = xy[cand_idx]
    ss = cl["s"][np.repeat(idx, len(offs))[cand_idx]]
    oo = offs[np.tile(np.arange(len(offs)), len(idx))[cand_idx]]
    hh_deg = np.degrees(np.arctan2(tang[idx][:, 1], tang[idx][:, 0]))
    hh_deg = np.repeat(hh_deg, len(offs))[cand_idx]

    # ---- 选两点: 间距 8~45 m, 优先分置道路两侧 ----
    target = float(np.clip(0.35 * cl["length"], 12.0, 35.0))
    order = rng.permutation(len(cand_idx))
    best_pair, best_score = None, -1e9
    for a_i in order[:min(len(order), 160)]:
        d = np.linalg.norm(xx - xx[a_i], axis=1)
        sep_ok = (d >= MIN_PAIR_DIST) & (d <= MAX_PAIR_DIST)
        if not sep_ok.any():
            sep_ok = (d >= 4.0) & (d <= 80.0)
            if not sep_ok.any():
                continue
        side_bonus = np.where(np.sign(oo) != np.sign(oo[a_i]), 6.0, 0.0)
        score = np.where(sep_ok, -np.abs(d - target) + side_bonus, -1e9)
        b_i = int(np.argmax(score))
        if score[b_i] > best_score:
            best_score = float(score[b_i])
            best_pair = (int(a_i), b_i)
    if best_pair is None:
        best_pair = (0, min(1, len(cand_idx) - 1))

    poses, metas = [], []
    for n, k in enumerate(best_pair):
        yaw = np.deg2rad(hh_deg[k]) if n == 0 else np.deg2rad(hh_deg[k]) + np.pi
        yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
        poses.append([float(xx[k, 0]), float(xx[k, 1]), float(z_abs[cand_idx[k]]),
                      0.0, 0.0, yaw])
        metas.append(dict(
            x=round(float(xx[k, 0]), 3), y=round(float(xx[k, 1]), 3),
            z_abs=round(float(z_abs[cand_idx[k]]), 3),
            height_above_ground_m=round(float(dz[cand_idx[k]]), 3),
            ground_z=round(float(hg[cand_idx[k]]), 3),
            static_clearance_m=round(float(dmin[cand_idx[k]]), 3),
            dynamic_clearance_m=round(float(dyn_margin[cand_idx[k]]), 3),
            lateral_offset_m=round(float(oo[k]), 3),
            s_along_m=round(float(ss[k]), 2),
            yaw_deg=round(float(np.rad2deg(yaw)), 2),
        ))

    pos_arr = np.asarray(poses, np.float64)
    pair_dist = float(np.linalg.norm(pos_arr[0, :2] - pos_arr[1, :2]))
    return dict(
        poses=pos_arr,
        meta=dict(
            source_traj=os.path.abspath(traj_path),
            ref_vid=cl["ref_vid"],
            centerline_length_m=round(cl["length"], 2),
            road_half_used_m=half,
            n_candidates=int(n_cand),
            n_valid=int(len(cand_idx)),
            relaxation=relaxation,
            pair_distance_m=round(pair_dist, 2),
            min_static_clearance_m=round(float(min(m["static_clearance_m"]
                                                   for m in metas)), 3),
            height_range_m=[min(m["height_above_ground_m"] for m in metas),
                            max(m["height_above_ground_m"] for m in metas)],
            dynamic_clearance_min_m=round(
                float(min(m["dynamic_clearance_m"] for m in metas)), 3),
            lidars=metas,
        ),
    )


# ---------------------------------------------------------------------------
# 试验配置规划
# ---------------------------------------------------------------------------
def _site_static_params(results_dir: str, site_id: str) -> dict:
    p = os.path.join(results_dir, "static_params", site_id + ".json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_plan(results_dir: str, sites: list, seed: int = 42) -> dict:
    from lidarsim.pointcloud import load_env, GroundModel

    plan = dict(
        design_version=DESIGN_VERSION,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=int(seed),
        n_lidar=N_LIDAR,
        presets=PRESETS,
        windows_min=WINDOWS,
        pos_levels=[0.2, 0.5],
        ang_levels=[5, 10, 15],
        dof_combos=[{"pos_m": c[0], "ang_deg": c[1]} for c in COMBOS],
        metric=METRIC,
        frame_step=FRAME_STEP,
        results_dir=os.path.abspath(results_dir),
        models_dir=os.path.abspath(MODELS_DIR),
        checkpoints={str(k): v for k, v in CHECKPOINTS.items()},
        constraint=dict(
            height_above_ground_m=list(Z_ABOVE_GROUND),
            static_clearance_m=STATIC_CLEARANCE,
            max_vehicle_height_m=MAX_VEH_HEIGHT,
            pair_distance_m=[MIN_PAIR_DIST, MAX_PAIR_DIST],
            lateral_offset_max_m=6.0,
        ),
        sites=[],
    )

    scene_i = 0
    for site in sites:
        log("=" * 72)
        log(f"[规划] {site['id']}  {site['name']}  [{site['family']}]")
        sp = _site_static_params(results_dir, site["id"])
        road = sp.get("road", {}) or {}
        road_half = None
        if road.get("road_width_m"):
            road_half = float(road["road_width_m"]) / 2.0

        env = load_env(site["env"])
        sub = np.asarray(env.points)[:: max(1, len(env.points) // 2_000_000)]
        gm = GroundModel(np.asarray(sub, np.float64), cell=1.0)
        log(f"  环境点云 {len(env.points)} 点; 道路半宽先验 = "
            f"{road_half if road_half else '缺省 5.0'} m")

        site_out = dict(id=site["id"], name=site["name"], family=site["family"],
                        env=os.path.abspath(site["env"]),
                        traj_dir=os.path.abspath(site["traj_dir"]),
                        stem=site["stem"],
                        static_params=os.path.join(results_dir, "static_params",
                                                   site["id"] + ".json")
                        if os.path.isfile(os.path.join(results_dir, "static_params",
                                                       site["id"] + ".json")) else None,
                        traffic=[])
        if site.get("skip"):
            log(f"  跳过交通场景序号 {site['skip']} (数据与点云不匹配, "
                f"见脚本 1 的 SITES 配置)")
        for k in traffic_indices(site):
            tfile = traj_file(site, k)
            if not os.path.isfile(tfile):
                raise FileNotFoundError(tfile)
            tid = f"traffic_{k:02d}"
            rng = np.random.default_rng(seed + 1000 * (SITES.index(site) + 1) + k)

            # 轨迹与点云范围一致性检查 (数据异常时跳过该场景, 不臆造位姿)
            rec = _load_traj_records(tfile)
            frac = _overlap_fraction(rec[:, 6:8], np.asarray(env.points))
            if frac < 0.5:
                site_out["traffic"].append(dict(
                    traffic=tid, traj_file=os.path.abspath(tfile),
                    status="no_overlap", overlap_fraction=round(frac, 4),
                    skip_reason=f"轨迹与点云水平范围不重合 (重合度 {frac:.1%}), "
                                f"请检查该轨迹文件的坐标系/路段",
                    init_poses=[], init_meta={},
                    static_stem=STATIC_NAME, units=[]))
                log(f"  {tid}: [跳过] 轨迹与点云不重合 (重合度 {frac:.1%})")
                continue

            try:
                init = generate_init_poses(site, tfile, road_half or 5.0,
                                           env, gm, rng)
            except Exception as e:
                site_out["traffic"].append(dict(
                    traffic=tid, traj_file=os.path.abspath(tfile),
                    status="failed", overlap_fraction=round(frac, 4),
                    skip_reason=f"初始位姿生成失败: {e}",
                    init_poses=[], init_meta={},
                    static_stem=STATIC_NAME, units=[]))
                log(f"  {tid}: [失败] {e}")
                continue

            units = []
            for j in range(3):
                w = WINDOWS[(scene_i + j) % 3]
                pos, ang = COMBOS[(scene_i + 2 * j) % 6]
                others = [x for x in WINDOWS if x != w]
                m_mis = int(others[rng.integers(0, 2)])
                tag_s = tag(w, pos, ang)
                mp = mpc_name(w, pos, ang)
                mm = marl_name(w, pos, ang, w)
                mx = marl_name(w, pos, ang, m_mis)
                units.append(dict(
                    unit=j + 1, w=int(w), pos_m=float(pos), ang_deg=int(ang),
                    runs=[
                        dict(method="mpc", m=None, mismatch=0, stem=mp),
                        dict(method="marl", m=int(w), mismatch=0, stem=mm),
                        dict(method="marl", m=int(m_mis), mismatch=1, stem=mx),
                    ],
                    tag=tag_s,
                ))
            site_out["traffic"].append(dict(
                traffic=tid,
                traj_file=os.path.abspath(tfile),
                status="ok", overlap_fraction=round(frac, 4), skip_reason="",
                init_poses=init["poses"].round(8).tolist(),
                init_meta=init["meta"],
                static_stem=STATIC_NAME,
                units=units,
            ))
            log(f"  {tid}: 有效候选 {init['meta']['n_valid']}, "
                f"间距 {init['meta']['pair_distance_m']} m, "
                f"离地 {init['meta']['height_range_m']} m, "
                f"最近静态遮挡 {init['meta']['min_static_clearance_m']} m")
            scene_i += 1
        plan["sites"].append(site_out)

    skipped = [(s["id"], t["traffic"], t.get("skip_reason", ""))
               for s in plan["sites"] for t in s["traffic"]
               if t.get("status") != "ok"]
    plan["n_traffic_scenes"] = scene_i
    plan["n_dynamic_runs"] = scene_i * 9
    plan["n_static_runs"] = scene_i
    plan["skipped"] = [dict(site=a, traffic=b, reason=c) for a, b, c in skipped]
    if skipped:
        log("\n[规划] 以下场景因数据问题被跳过, 需人工确认:")
        for a, b, c in skipped:
            log(f"  - {a}/{b}: {c}")
    return plan


def write_pose_files(results_dir: str, site: dict, traffic: dict,
                     overwrite: bool = False) -> tuple:
    """写出 lidar_pose_init.json / lidar_pose_static.json (内容以 plan 为准)."""
    tdir = traffic_dir(results_dir, site["id"], traffic["traffic"])
    os.makedirs(tdir, exist_ok=True)
    poses = np.asarray(traffic.get("init_poses") or [], np.float64)
    if poses.size == 0:
        return None, None, False

    init_path = os.path.join(tdir, INIT_NAME)
    static_path = os.path.join(tdir, STATIC_NAME)
    stale = False
    if overwrite or not _poses_match(init_path, poses, key="poses"):
        stale = os.path.isfile(init_path)
        obj = dict(n_lidar=int(len(poses)), presets=list(PRESETS),
                   poses=np.round(poses, 8).tolist(),
                   pose_order=POSE_KEYS, pose_unit=POSE_UNIT)
        with open(init_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    if overwrite or not _poses_match(static_path, poses, key="windows"):
        # 静态基线: 单窗口, 不写 window_length_sec / t_range (评估时覆盖整条轨迹)
        obj = dict(presets=list(PRESETS), pose_order=POSE_KEYS, pose_unit=POSE_UNIT,
                   windows=[dict(index=1, poses=np.round(poses, 8).tolist())])
        with open(static_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return init_path, static_path, stale


def _poses_match(path: str, poses: np.ndarray, key: str) -> bool:
    """已有位姿文件是否与规划一致 (不一致则视为过期, 需要重写)."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cur = obj["poses"] if key == "poses" else obj["windows"][0]["poses"]
        return np.allclose(np.asarray(cur, np.float64),
                           np.round(poses, 8), atol=1e-6)
    except Exception:
        return False


def write_plan_files(results_dir: str, plan: dict):
    pdir = os.path.join(results_dir, "plan")
    os.makedirs(pdir, exist_ok=True)
    plan_path = os.path.join(pdir, "experiment_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    # 设计表 (一行 = 一次运行; 静态行 w/pos/ang/m 留空)
    rows = []
    for site in plan["sites"]:
        for tr in site["traffic"]:
            if tr.get("status", "ok") != "ok":
                continue
            base = dict(site=site["id"], site_name=site["name"],
                        family=site["family"], traffic=tr["traffic"],
                        traj_file=tr["traj_file"])
            rows.append(dict(run_id=f"{site['id']}/{tr['traffic']}/static",
                             method="static", w="", m="", mismatch="",
                             pos="", ang="", stem=STATIC_NAME,
                             eval_dir=eval_dirname(STATIC_NAME), **base))
            for u in tr["units"]:
                for r in u["runs"]:
                    method = "mpc" if r["method"] == "mpc" else (
                        "marl_match" if r["mismatch"] == 0 else "marl_mismatch")
                    rows.append(dict(
                        run_id=f"{site['id']}/{tr['traffic']}/{r['stem'][:-5]}",
                        method=method, w=u["w"], m=r["m"] if r["m"] else "",
                        mismatch=r["mismatch"], pos=u["pos_m"], ang=u["ang_deg"],
                        stem=r["stem"], eval_dir=eval_dirname(r["stem"]), **base))
    cols = ["run_id", "site", "site_name", "family", "traffic", "traj_file",
            "method", "w", "m", "mismatch", "pos", "ang", "stem", "eval_dir"]
    mpath = os.path.join(pdir, "design_manifest.csv")
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # 均衡性核对
    def _count(key, src=rows):
        out = {}
        for r in src:
            v = r[key]
            if v == "":
                continue
            out[str(v)] = out.get(str(v), 0) + 1
        return out

    bal = dict(n_runs=len(rows),
               n_static=sum(1 for r in rows if r["method"] == "static"),
               by_method=_count("method"), by_w=_count("w"), by_pos=_count("pos"),
               by_ang=_count("ang"), by_m=_count("m"),
               by_w_mismatch={}, by_combo={})
    for r in rows:
        if r["method"] == "marl_mismatch":
            k = f"w{r['w']}->m{r['m']}"
            bal["by_w_mismatch"][k] = bal["by_w_mismatch"].get(k, 0) + 1
        if r["pos"] != "":
            k = f"pos{r['pos']}_a{r['ang']}"
            bal["by_combo"][k] = bal["by_combo"].get(k, 0) + 1
    with open(os.path.join(pdir, "balance_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(bal, f, ensure_ascii=False, indent=2)

    if plan.get("skipped"):
        with open(os.path.join(pdir, "skipped_scenes.json"), "w",
                  encoding="utf-8") as f:
            json.dump(plan["skipped"], f, ensure_ascii=False, indent=2)

    log(f"\n[规划] 已写出:\n  {plan_path}\n  {mpath}")
    log(f"[规划] 运行总数 {bal['n_runs']} (静态 {bal['n_static']}); "
        f"方法 {bal['by_method']}")
    log(f"[规划] w {bal['by_w']} | pos {bal['by_pos']} | ang {bal['by_ang']} | m {bal['by_m']}")
    return plan_path


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
def _pose_valid(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return len(obj.get("windows", [])) > 0
    except Exception:
        return False


def _run_cmd(cmd: list, log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        r = subprocess.run(cmd, cwd=_REPO, stdout=lf, stderr=subprocess.STDOUT,
                           env=env)
    return r.returncode


def _mpc_cmd(site, traffic, u, run, results_dir, args) -> list:
    tdir = traffic_dir(results_dir, site["id"], traffic["traffic"])
    cmd = [sys.executable, "-m", "rslaa.mpc_controller",
           "--pose-json", os.path.join(tdir, INIT_NAME),
           "--traj", traffic["traj_file"], "--env", site["env"],
           "--models", MODELS_DIR,
           "--out", os.path.join(tdir, run["stem"]),
           "--window-min", str(u["w"]),
           "--pos-margin", str(u["pos_m"]),
           "--ang-margin-deg", str(u["ang_deg"]),
           "--metric", METRIC, "--optimizer", "fast", "--n-cand", "64",
           "--device", args.device, "--frame-step", str(args.frame_step),
           "--workers", str(args.workers),
           "--quiet"]
    if args.max_windows > 0:
        cmd += ["--max-windows", str(args.max_windows)]
    return cmd


def _marl_cmd(site, traffic, u, run, results_dir, args) -> list:
    tdir = traffic_dir(results_dir, site["id"], traffic["traffic"])
    policy = CHECKPOINTS[int(run["m"])]
    if not os.path.isfile(policy):
        raise FileNotFoundError(policy)
    cmd = [sys.executable, "-m", "rslaa.marl_deploy",
           "--policy", policy,
           "--pose-json", os.path.join(tdir, INIT_NAME),
           "--traj", traffic["traj_file"], "--env", site["env"],
           "--models", MODELS_DIR,
           "--out", os.path.join(tdir, run["stem"]),
           "--window-min", str(u["w"]),
           "--pos-margin", str(u["pos_m"]),
           "--ang-margin-deg", str(u["ang_deg"]),
           "--device", args.device, "--frame-step", str(args.frame_step),
           "--workers", str(args.workers),
           "--seed", "0", "--quiet"]
    if args.max_windows > 0:
        cmd += ["--max-windows", str(args.max_windows)]
    return cmd


def run_site(site: dict, plan: dict, results_dir: str, args) -> dict:
    t_start = time.time()
    manifest_path = os.path.join(results_dir, "plan", "run_manifest.csv")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    new_file = not os.path.isfile(manifest_path)
    mf = open(manifest_path, "a", newline="", encoding="utf-8-sig")
    mw = csv.writer(mf)
    if new_file:
        mw.writerow(["time", "site", "traffic", "method", "w", "m", "pos", "ang",
                     "stem", "status", "seconds", "out", "log"])

    stats = dict(ok=0, skip=0, fail=0)
    for traffic in site["traffic"]:
        if traffic.get("status", "ok") != "ok":
            log(f"\n[{site['id']}/{traffic['traffic']}] 跳过: "
                f"{traffic.get('skip_reason', '')}")
            continue
        tdir = traffic_dir(results_dir, site["id"], traffic["traffic"])
        os.makedirs(tdir, exist_ok=True)
        _, _, stale = write_pose_files(results_dir, site, traffic, overwrite=False)
        log(f"\n[{site['id']}/{traffic['traffic']}] 目录就绪: {tdir}")
        if stale and not args.overwrite:
            log("  [注意] 规划位姿与本目录原有 init/static 不一致, 已按规划重写;"
                " 旧动态位姿需加 --overwrite 重跑")

        if args.only_static:
            continue

        for u in traffic["units"]:
            for run in u["runs"]:
                out = os.path.join(tdir, run["stem"])
                log_f = os.path.join(results_dir, "logs", site["id"],
                                     traffic["traffic"], run["stem"][:-5] + ".log")
                if _pose_valid(out) and not args.overwrite:
                    stats["skip"] += 1
                    log(f"  跳过 (已存在): {run['stem']}")
                    continue
                if args.dry_run:
                    log(f"  [dry-run] {run['method']} {u['w']}min "
                        f"pos{u['pos_m']} a{u['ang_deg']} m={run['m']}")
                    continue
                if run["method"] == "mpc":
                    cmd = _mpc_cmd(site, traffic, u, run, results_dir, args)
                else:
                    cmd = _marl_cmd(site, traffic, u, run, results_dir, args)

                t0 = time.time()
                log(f"  运行 {run['stem']} ...")
                rc = _run_cmd(cmd, log_f)
                dt = time.time() - t0
                ok = (rc == 0) and _pose_valid(out)
                stats["ok" if ok else "fail"] += 1
                mw.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), site["id"],
                             traffic["traffic"], run["method"], u["w"],
                             run["m"] if run["m"] else "", u["pos_m"],
                             u["ang_deg"], run["stem"],
                             "ok" if ok else f"fail(rc={rc})", f"{dt:.1f}",
                             out, log_f])
                mf.flush()
                log(f"    -> {'ok' if ok else 'FAILED'} ({dt:.1f}s)"
                    + ("" if ok else f"  日志: {log_f}"))

    mf.close()
    log(f"\n[{site['id']}] 完成: ok={stats['ok']} skip={stats['skip']} "
        f"fail={stats['fail']}, 用时 {(time.time() - t_start) / 60:.1f} min")
    return stats


def select_sites(plan: dict, want=None) -> list:
    if not want:
        return plan["sites"]
    out = []
    for s in plan["sites"]:
        if s["id"] in want or s["name"] in want:
            out.append(s)
    if not out:
        raise SystemExit(f"未匹配到地点: {want}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_plan(path: str) -> dict:
    if not os.path.isfile(path):
        raise SystemExit(f"未找到规划文件 {path}, 请先运行 plan 子命令")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _args_sites(args):
    if getattr(args, "site", None):
        return [s.strip() for s in args.site.split(",") if s.strip()]
    if getattr(args, "sites", None):
        return [s.strip() for s in args.sites.split(",") if s.strip()]
    return None


def cmd_plan(args):
    results_dir = args.results_dir
    os.makedirs(os.path.join(results_dir, "plan"), exist_ok=True)
    plan_path = os.path.join(results_dir, "plan", "experiment_plan.json")
    if os.path.isfile(plan_path) and not args.force:
        log(f"规划文件已存在: {plan_path}\n使用 --force 重新生成。")
        return 0
    sites = SITES
    if _args_sites(args):
        sites = [s for s in SITES if s["id"] in _args_sites(args)
                 or s["name"] in _args_sites(args)]
        if not sites:
            raise SystemExit("未匹配到地点")
    plan = build_plan(results_dir, sites, seed=args.seed)
    write_plan_files(results_dir, plan)
    # 预先建好全部场景目录与 init/static 位姿文件
    for site in plan["sites"]:
        for tr in site["traffic"]:
            if tr.get("status", "ok") == "ok":
                write_pose_files(results_dir, site, tr, overwrite=False)
    log("[规划] 已按规划建立场景目录并写出 init/static 位姿文件")
    return 0


def cmd_run(args):
    plan_path = os.path.join(args.results_dir, "plan", "experiment_plan.json")
    plan = _load_plan(plan_path)
    sites = select_sites(plan, _args_sites(args))
    traffic_want = ([t.strip() for t in args.traffic.split(",") if t.strip()]
                    if args.traffic else None)
    if traffic_want:
        for s in sites:
            s["traffic"] = [t for t in s["traffic"] if t["traffic"] in traffic_want]

    log(f"[运行] 共 {len(sites)} 个地点, "
        f"每地点 {len(sites[0]['traffic'])} 个交通场景 "
        f"(其中 {sum(1 for tr in sites[0]['traffic'] if tr.get('status') == 'ok')} "
        f"个可运行)")
    tty = sys.stdin.isatty()
    totals = dict(ok=0, skip=0, fail=0)
    for n, site in enumerate(sites):
        st = run_site(site, plan, args.results_dir, args)
        for k in totals:
            totals[k] += st[k]
        if n < len(sites) - 1:
            nxt = sites[n + 1]
            if args.yes:
                log(f"\n[自动继续] 下一个地点: {nxt['id']} ({nxt['name']})")
                continue
            if not tty:
                log(f"\n非交互环境, 已在 {site['id']} 结束后停止。"
                    f"\n继续下一个地点请加 --yes, 或 --site {nxt['id']} 单独运行。")
                break
            ans = input(f"\n{site['id']} 已全部完成。是否继续下一个地点 "
                        f"{nxt['id']}? [Y/N]: ").strip().lower()
            if ans not in ("y", "yes", "是"):
                log("已停止。")
                break
    log(f"\n[运行] 汇总: ok={totals['ok']} skip={totals['skip']} "
        f"fail={totals['fail']}")
    return 0 if totals["fail"] == 0 else 1


def cmd_status(args):
    plan_path = os.path.join(args.results_dir, "plan", "experiment_plan.json")
    plan = _load_plan(plan_path)
    log(f"{'site':<12}{'traffic':<12}{'init':>5}{'static':>7}"
        f"{'dynamic':>9}{'eval':>6}")
    tot = dict(init=0, static=0, dyn=0, dyn_need=0, ev=0, skipped=0)
    for site in plan["sites"]:
        for tr in site["traffic"]:
            tdir = traffic_dir(args.results_dir, site["id"], tr["traffic"])
            if tr.get("status", "ok") != "ok":
                log(f"{site['id']:<12}{tr['traffic']:<12}  [跳过] "
                    f"{tr.get('skip_reason', '')}")
                tot["skipped"] += 1
                continue
            n_init = int(os.path.isfile(os.path.join(tdir, INIT_NAME)))
            n_static = int(_pose_valid(os.path.join(tdir, STATIC_NAME)))
            stems = [r["stem"] for u in tr["units"] for r in u["runs"]]
            n_dyn = sum(_pose_valid(os.path.join(tdir, s)) for s in stems)
            n_ev = sum(os.path.isfile(os.path.join(tdir, eval_dirname(s),
                                                    "config.json"))
                       for s in [STATIC_NAME] + stems)
            log(f"{site['id']:<12}{tr['traffic']:<12}{n_init:>5}{n_static:>7}"
                f"{n_dyn:>6}/{len(stems)}{n_ev:>6}/{len(stems) + 1}")
            tot["init"] += n_init
            tot["static"] += n_static
            tot["dyn"] += n_dyn
            tot["dyn_need"] += len(stems)
            tot["ev"] += n_ev
    need_static = tot["dyn_need"] // 9
    log(f"\n合计: init {tot['init']}/{need_static}, "
        f"static {tot['static']}/{need_static}, "
        f"dynamic {tot['dyn']}/{tot['dyn_need']}, "
        f"eval {tot['ev']}/{tot['dyn_need'] + need_static}, "
        f"跳过场景 {tot['skipped']}")
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="RSLaA 第二部分: 配置规划 + 动态试验自动运行",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plan", help="生成/冻结试验配置")
    p1.add_argument("--sites", default="")
    p1.add_argument("--results-dir", default=RESULTS_DIR)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--force", action="store_true")

    p2 = sub.add_parser("run", help="按规划运行动态试验")
    p2.add_argument("--site", default="", help="单个/多个地点, 逗号分隔")
    p2.add_argument("--sites", default="", help="同 --site")
    p2.add_argument("--traffic", default="", help="如 traffic_01,traffic_02")
    p2.add_argument("--results-dir", default=RESULTS_DIR)
    p2.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p2.add_argument("--frame-step", type=int, default=FRAME_STEP)
    p2.add_argument("--workers", type=int, default=1,
                    help="帧级并行仿真进程数 (>1 启用多进程 GPU 并行, "
                         "结果与串行逐位一致; 显存约 0.5~1 GB/进程)")
    p2.add_argument("--max-windows", type=int, default=0,
                    help="仅前 N 窗 (0=全部, 试点用)")
    p2.add_argument("--yes", action="store_true",
                    help="地点间不再询问, 自动继续")
    p2.add_argument("--overwrite", action="store_true", help="重跑已存在的位姿文件")
    p2.add_argument("--only-static", action="store_true",
                    help="只准备 init/static 文件, 不跑动态试验")
    p2.add_argument("--dry-run", action="store_true")

    p3 = sub.add_parser("status", help="检查文件完成度")
    p3.add_argument("--results-dir", default=RESULTS_DIR)

    return ap.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "status":
        return cmd_status(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

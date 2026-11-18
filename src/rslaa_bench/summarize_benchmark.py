# -*- coding: utf-8 -*-
"""run_benchmark 产物的数据汇总与相关性分析.

功能:
  1. 递归汇集一个基准输出根目录下任意层级子文件夹中的全部
     scene_*/traffic_*/t_* 运行目录 (scene_SSS/traffic_TK/t_TT/config.json,
     data_*.json, lidar_para_*.json, traffic_*.txt), 例如
       root/run_A/scene_001/traffic_01/t_05/...      (多实验批次汇总)
       root/scene_001/traffic_01/t_05/...            (标准单根)
     输出三张结构化表 (CSV, 首列 source 记录 scene 所在子文件夹):
       summary_candidates.csv  候选级 (每个决策窗 6 行): 8 指标 + score +
                               自由度 + 窗长 + 交通参数 + 是否选中
       summary_windows.csv     窗口级 (每个时间窗 1 行, 含首窗): 配置/交通/选中方案
       summary_poses.csv       LiDAR 级长表: 每窗全部方案的逐雷达位姿参数
  2. 相关性分析: 5 个替代指标 (J_EIG/J_cov/J_mv/J_dyn/J_traj) 分别对
     score / IoU / Dev / Comp 计算 Pearson r 与 Spearman rho (含 p 值与样本数),
     输出 correlations_*.csv 与控制台矩阵.
  3. 绘制 4 x 5 相关性散点图 (行 = score/IoU/Dev/Comp, 列 = 5 个替代指标),
     支持按位置自由度/角度自由度/时间窗长度/场景筛选后重绘,
     例如只看位置自由度 ±0.4 m 的子集.

用法:
  # 全量汇总 + 相关性 + 图
  python -m rslaa_bench.summarize_benchmark --root bench_out

  # 子集: 位置自由度 ±0.4 m 且 5 min 时间窗
  python -m rslaa_bench.summarize_benchmark --root bench_out \\
      --pos-margin 0.4 --window-min 5

  # 仅看每个窗选中的方案
  python -m rslaa_bench.summarize_benchmark --root bench_out --chosen-only

说明:
  * J_dyn/J_traj 在历史不足时为 null (退化), 相关性按各指标对 pairwise 剔除.
  * Dev 越小越好, 因此 "好" 的替代指标与 Dev 应呈负相关.
  * 仅依赖 numpy/scipy/matplotlib (与主项目一致), 不需要 pandas.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# 常量 (与 run_benchmark 保持一致)
# ---------------------------------------------------------------------------
SURROGATES = ["J_EIG", "J_cov", "J_mv", "J_dyn", "J_traj"]
PERF = ["IoU", "Dev", "Comp"]
TARGETS = ["score"] + PERF
DATA_COLS = PERF + SURROGATES

CAND_COLS = (["source", "scene", "traffic", "window_min", "window", "scheme",
              "chosen", "pos_margin_m", "ang_margin_deg", "n_lidar", "seed"]
             + DATA_COLS + ["score", "mean_speed_mps", "volume_veh_per_h"])
WINDOW_COLS = ["source", "scene", "traffic", "window_min", "window", "t0",
               "t1", "n_schemes", "chosen_scheme", "pos_margin_m",
               "ang_margin_deg", "n_lidar", "mean_speed_mps",
               "volume_veh_per_h", "chosen_IoU", "chosen_Dev",
               "chosen_Comp", "chosen_score"]
POSE_COLS = ["source", "scene", "traffic", "window_min", "window", "scheme",
             "chosen", "lidar", "type", "x", "y", "z", "roll", "pitch", "yaw"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 读取与汇总
# ---------------------------------------------------------------------------
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_traffic(path):
    """traffic_*.txt -> {mean_speed_mps, volume_veh_per_h}."""
    out = {"mean_speed_mps": None, "volume_veh_per_h": None}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return out


def iter_run_dirs(root):
    """递归枚举 root 下任意层级子文件夹中的 scene_*/traffic_*/t_* 运行目录.

    支持:
      * root/scene_*/traffic_*/t_*                       (标准单根)
      * root/<任意层级子文件夹>/scene_*/traffic_*/t_*    (多批次/多实验汇总)
      * 旧两级结构 scene_*/t_*                           (traffic 记为 1)

    返回 [(tdir, source), ...]: source 为 scene_* 所在父目录相对 root 的
    路径 (root 直属时为 ""), 用于区分不同子文件夹中相同编号的 scene.
    """
    out = []
    root = os.path.abspath(root)
    for dirpath, dirnames, _ in os.walk(root):
        for d in sorted(dirnames):
            if not d.startswith("scene_"):
                continue
            sd = os.path.join(dirpath, d)
            tdirs = sorted(glob.glob(os.path.join(sd, "traffic_*", "t_*")))
            if not tdirs:                       # 旧版两级结构回退
                tdirs = sorted(glob.glob(os.path.join(sd, "t_*")))
            if not tdirs:
                continue
            source = os.path.relpath(dirpath, root)
            if source == ".":
                source = ""
            out.extend((td, source) for td in tdirs)
    return sorted(out)


def _traffic_of_dir(tdir):
    """由 t_TT 目录反推 traffic 编号; 旧两级结构返回 1."""
    parent = os.path.basename(os.path.dirname(tdir))
    if parent.startswith("traffic_"):
        return int(parent.split("_")[1])
    return 1


def aggregate(root):
    """汇集基准目录 -> (candidate_rows, window_rows, pose_rows).

    candidate_rows: 每个决策窗 (k>=2) 的每个候选方案一行.
    window_rows:    每个时间窗 (含首窗) 一行.
    pose_rows:      每个时间窗全部方案的逐 LiDAR 位姿一行 (长表).
    """
    cand_rows, win_rows, pose_rows = [], [], []
    run_dirs = iter_run_dirs(root)
    if not run_dirs:
        log(f"WARNING: no scene_*/traffic_*/t_* directory under {root} "
            f"(recursive)")
    for tdir, source in run_dirs:
        traffic = _traffic_of_dir(tdir)
        cfg_path = os.path.join(tdir, "config.json")
        cfg = _load_json(cfg_path) if os.path.isfile(cfg_path) else {}
        pos_m = cfg.get("pos_margin_m")
        ang_d = cfg.get("ang_margin_deg")
        n_lidar = cfg.get("n_lidar")
        seed = cfg.get("seed")
        cfg_wmin = cfg.get("window_min")

        lp_files = sorted(glob.glob(os.path.join(tdir, "lidar_para_*.json")))
        for lp_path in lp_files:
            lp = _load_json(lp_path)
            tag = os.path.basename(lp_path)[len("lidar_para_"):-len(".json")]
            scene = lp.get("scene", cfg.get("scene"))
            wmin = lp.get("window_min", cfg_wmin)
            k = lp["window"]
            t_range = lp.get("t_range", [None, None])
            chosen0 = lp.get("chosen", 0)           # 0 基索引
            n_schemes = lp.get("n_schemes", 1)

            # ---- LiDAR 位姿长表 ----
            for e in lp.get("entries", []):
                scheme, lid, ltype = int(e[0]), int(e[1]), str(e[2])
                pose_rows.append(dict(
                    source=source, scene=scene, traffic=traffic,
                    window_min=wmin, window=k, scheme=scheme,
                    chosen=(scheme == chosen0 + 1), lidar=lid, type=ltype,
                    x=float(e[3]), y=float(e[4]), z=float(e[5]),
                    roll=float(e[6]), pitch=float(e[7]), yaw=float(e[8])))

            # ---- 交通参数 ----
            tr = _parse_traffic(os.path.join(tdir, f"traffic_{tag}.txt"))

            # ---- 同窗 data (k>=2) ----
            data_path = os.path.join(tdir, f"data_{tag}.json")
            chosen_perf = dict(chosen_IoU=None, chosen_Dev=None,
                               chosen_Comp=None, chosen_score=None)
            if os.path.isfile(data_path):
                d = _load_json(data_path)
                cols = d.get("columns", DATA_COLS)
                data = d.get("data", [])
                scores = d.get("score", [None] * len(data))
                dchosen = int(d.get("chosen", chosen0))
                for a, row in enumerate(data):
                    vals = {c: (None if v is None else float(v))
                            for c, v in zip(cols, row)}
                    rec = dict(source=source, scene=scene, traffic=traffic,
                               window_min=wmin, window=k, scheme=a + 1,
                               chosen=(a == dchosen),
                               pos_margin_m=pos_m, ang_margin_deg=ang_d,
                               n_lidar=n_lidar, seed=seed,
                               mean_speed_mps=tr["mean_speed_mps"],
                               volume_veh_per_h=tr["volume_veh_per_h"],
                               score=(float(scores[a])
                                      if scores[a] is not None else None))
                    rec.update({c: vals.get(c) for c in DATA_COLS})
                    cand_rows.append(rec)
                if 0 <= dchosen < len(data):
                    cr = data[dchosen]
                    cmap = dict(zip(cols, cr))
                    chosen_perf = dict(
                        chosen_IoU=(None if cmap.get("IoU") is None
                                    else float(cmap["IoU"])),
                        chosen_Dev=(None if cmap.get("Dev") is None
                                    else float(cmap["Dev"])),
                        chosen_Comp=(None if cmap.get("Comp") is None
                                     else float(cmap["Comp"])),
                        chosen_score=(float(scores[dchosen])
                                      if scores[dchosen] is not None else None))

            win_rows.append(dict(
                source=source, scene=scene, traffic=traffic, window_min=wmin,
                window=k,
                t0=(t_range[0] if t_range else None),
                t1=(t_range[1] if t_range else None),
                n_schemes=n_schemes, chosen_scheme=chosen0 + 1,
                pos_margin_m=pos_m, ang_margin_deg=ang_d, n_lidar=n_lidar,
                mean_speed_mps=tr["mean_speed_mps"],
                volume_veh_per_h=tr["volume_veh_per_h"], **chosen_perf))

    return cand_rows, win_rows, pose_rows


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return v


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in cols})
    log(f"saved: {os.path.abspath(path)}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 筛选
# ---------------------------------------------------------------------------
def filter_rows(rows, pos_margin=None, ang_margin=None, window_min=None,
                scene=None, traffic=None, chosen_only=False):
    """按自由度/时间窗/场景/交通批次等条件筛选候选行."""
    out = rows
    if pos_margin is not None:
        out = [r for r in out if r.get("pos_margin_m") is not None
               and abs(float(r["pos_margin_m"]) - pos_margin) < 1e-6]
    if ang_margin is not None:
        out = [r for r in out if r.get("ang_margin_deg") is not None
               and abs(float(r["ang_margin_deg"]) - ang_margin) < 1e-6]
    if window_min is not None:
        out = [r for r in out if r.get("window_min") == window_min]
    if scene is not None:
        out = [r for r in out if r.get("scene") == scene]
    if traffic is not None:
        out = [r for r in out if r.get("traffic") == traffic]
    if chosen_only:
        out = [r for r in out if r.get("chosen")]
    return out


def filter_slug(pos_margin=None, ang_margin=None, window_min=None,
                scene=None, traffic=None, chosen_only=False):
    parts = []
    if pos_margin is not None:
        parts.append(f"pos{pos_margin:g}m")
    if ang_margin is not None:
        parts.append(f"ang{ang_margin:g}deg")
    if window_min is not None:
        parts.append(f"t{window_min:02d}")
    if scene is not None:
        parts.append(f"scene{scene:03d}")
    if traffic is not None:
        parts.append(f"traffic{traffic:02d}")
    if chosen_only:
        parts.append("chosen")
    return "_".join(parts) if parts else "all"


# ---------------------------------------------------------------------------
# 相关性分析
# ---------------------------------------------------------------------------
def _column(rows, key):
    return np.array([np.nan if r.get(key) is None else float(r[key])
                     for r in rows], dtype=np.float64)


def correlation_pair(x, y):
    """成对剔除 NaN 后计算 Pearson/Spearman; 常量或样本不足返回 nan."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    res = dict(n=n, pearson_r=np.nan, pearson_p=np.nan,
               spearman_rho=np.nan, spearman_p=np.nan)
    if n < 2:
        return res
    xx, yy = x[m], y[m]
    if np.std(xx) > 0 and np.std(yy) > 0:
        r, p = stats.pearsonr(xx, yy)
        res["pearson_r"], res["pearson_p"] = float(r), float(p)
        rho, ps = stats.spearmanr(xx, yy)
        if np.isfinite(rho):
            res["spearman_rho"], res["spearman_p"] = float(rho), float(ps)
    return res


def correlation_report(rows):
    """4 目标 x 5 替代指标的相关性, 返回行字典列表."""
    out = []
    for target in TARGETS:
        y = _column(rows, target)
        for s in SURROGATES:
            r = correlation_pair(_column(rows, s), y)
            r.update(target=target, surrogate=s)
            out.append(r)
    return out


def write_correlations(path, report):
    cols = ["target", "surrogate", "n", "pearson_r", "pearson_p",
            "spearman_rho", "spearman_p"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in report:
            w.writerow({k: ("" if isinstance(v, float) and np.isnan(v)
                            else v) for k, v in r.items()})
    log(f"saved: {os.path.abspath(path)}")


def print_report(report, tag=""):
    log(f"correlation matrix{(' [' + tag + ']') if tag else ''} "
        f"(rows: target, cols: surrogate):")
    head = "target\\sur  " + "".join(f"{s:>10s}" for s in SURROGATES)
    print("  " + head)
    for target in TARGETS:
        cells = []
        for s in SURROGATES:
            r = next(x for x in report
                     if x["target"] == target and x["surrogate"] == s)
            cells.append("   nan    " if np.isnan(r["pearson_r"])
                         else f"{r['pearson_r']:+6.3f}/{r['spearman_rho']:+5.2f}")
        print(f"  {target:>10s}  " + "".join(f"{c:>10s}" for c in cells))
    print("  (cell = Pearson r / Spearman rho)")


# ---------------------------------------------------------------------------
# 4 x 5 相关性散点图
# ---------------------------------------------------------------------------
def plot_corr_grid(rows, out_path, title_suffix="all candidates"):
    """行 = score/IoU/Dev/Comp, 列 = 5 替代指标; 散点+回归线+r/rho/n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_ALL, C_CHOSEN, C_FIT = "#4C78A8", "#E45756", "#333333"
    fig, axes = plt.subplots(4, 5, figsize=(22, 14), sharex="col")
    chosen = np.array([bool(r.get("chosen")) for r in rows])

    for i, target in enumerate(TARGETS):
        y_all = _column(rows, target)
        for j, s in enumerate(SURROGATES):
            ax = axes[i, j]
            x_all = _column(rows, s)
            m = np.isfinite(x_all) & np.isfinite(y_all)
            n = int(m.sum())
            xo, yo = x_all[m], y_all[m]
            mc = chosen[m]

            if n:
                ax.scatter(xo[~mc], yo[~mc], s=26, c=C_ALL, alpha=0.55,
                           edgecolors="none", label="candidate")
                if mc.any():
                    ax.scatter(xo[mc], yo[mc], marker="*", s=130,
                               c=C_CHOSEN, edgecolors="k", linewidths=0.4,
                               zorder=5, label="chosen")
            if n >= 2 and np.std(xo) > 0:
                coef = np.polyfit(xo, yo, 1)
                xs = np.linspace(xo.min(), xo.max(), 50)
                ax.plot(xs, np.polyval(coef, xs), ls="--", lw=1.4,
                        c=C_FIT, zorder=4)
            cp = correlation_pair(x_all, y_all)
            txt = (f"n={cp['n']}\n"
                   f"r={cp['pearson_r']:+.3f}\n"
                   f"rho={cp['spearman_rho']:+.3f}") \
                if np.isfinite(cp["pearson_r"]) else f"n={cp['n']}\nN/A"
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top",
                    ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#999999", alpha=0.85))
            ax.grid(alpha=0.3, ls="--")
            if i == 0:
                ax.set_title(s, fontsize=12, fontweight="bold")
            if i == 3:
                ax.set_xlabel(s)
            if j == 0:
                ylab = target + (" (m, lower better)" if target == "Dev"
                                 else "")
                ax.set_ylabel(ylab, fontsize=11, fontweight="bold")
            if i == 0 and j == 4:
                ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Surrogate metrics vs. perception performance — "
                 f"{title_suffix} (n candidates = {len(rows)})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log(f"saved: {os.path.abspath(out_path)}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    log(f"aggregating benchmark outputs under: {cfg.root}")
    cand, win, pose = aggregate(cfg.root)
    log(f"collected: {len(cand)} candidate rows, {len(win)} windows, "
        f"{len(pose)} lidar-pose rows")

    # ---- 结构化数据 (始终全量) ----
    write_csv(os.path.join(cfg.out_dir, "summary_candidates.csv"),
              cand, CAND_COLS)
    write_csv(os.path.join(cfg.out_dir, "summary_windows.csv"),
              win, WINDOW_COLS)
    write_csv(os.path.join(cfg.out_dir, "summary_poses.csv"),
              pose, POSE_COLS)

    if not cand:
        log("no decision-window data (data_*.json) found; nothing to analyze.")
        return

    # ---- 全量相关性 + 图 ----
    rep_all = correlation_report(cand)
    write_correlations(os.path.join(cfg.out_dir, "correlations_all.csv"),
                       rep_all)
    print_report(rep_all, "all")
    if not cfg.no_plot:
        plot_corr_grid(cand, os.path.join(cfg.out_dir, "corr_scatter_all.png"),
                       "all runs")

    # ---- 筛选子集 ----
    any_filter = any(v is not None for v in (cfg.pos_margin, cfg.ang_margin,
                                             cfg.window_min, cfg.scene,
                                             cfg.traffic)) \
        or cfg.chosen_only
    if any_filter:
        sub = filter_rows(cand, cfg.pos_margin, cfg.ang_margin,
                          cfg.window_min, cfg.scene, cfg.traffic,
                          cfg.chosen_only)
        slug = filter_slug(cfg.pos_margin, cfg.ang_margin, cfg.window_min,
                           cfg.scene, cfg.traffic, cfg.chosen_only)
        log(f"filter [{slug}]: {len(sub)} candidate rows")
        if sub:
            rep = correlation_report(sub)
            write_correlations(
                os.path.join(cfg.out_dir, f"correlations_{slug}.csv"), rep)
            print_report(rep, slug)
            if not cfg.no_plot:
                label = slug.replace("_", ", ").replace("pos", "pos ")
                plot_corr_grid(
                    sub, os.path.join(cfg.out_dir, f"corr_scatter_{slug}.png"),
                    label)
        else:
            log("WARNING: filter matched no rows; skipped.")
    log("done.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="run_benchmark 数据汇总 + 替代指标相关性分析")
    ap.add_argument("--root", required=True,
                    help="基准输出根目录 (递归搜索其下任意层级子文件夹中的 "
                         "scene_*/traffic_*/t_*)")
    ap.add_argument("--out-dir", default=None,
                    help="汇总结果输出目录 (默认 <root>/_summary)")
    ap.add_argument("--pos-margin", type=float, default=None,
                    help="按位置自由度筛选 (m), 如 0.4")
    ap.add_argument("--ang-margin", type=float, default=None,
                    help="按角度自由度筛选 (deg), 如 10")
    ap.add_argument("--window-min", type=int, default=None,
                    help="按时间窗长度筛选 (min), 如 5/15/30")
    ap.add_argument("--scene", type=int, default=None,
                    help="按场景编号筛选")
    ap.add_argument("--traffic", type=int, default=None,
                    help="按交通批次编号筛选")
    ap.add_argument("--chosen-only", action="store_true",
                    help="仅保留每个窗选中的方案")
    ap.add_argument("--no-plot", action="store_true",
                    help="只输出 CSV, 不绘制散点图")
    args = ap.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = os.path.join(args.root, "_summary")
    return args


if __name__ == "__main__":
    run(parse_args())

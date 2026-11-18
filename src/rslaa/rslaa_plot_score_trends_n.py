# -*- coding: utf-8 -*-
"""方案 B 口径下的感知收益趋势图（相对静态基线）—— 不覆盖原脚本。

方案 B 指标口径（全部基于 evals_raw.csv 的逐个体记录）
------------------------------------------------------
TP      : matched==1 且 IoU3D > 0.15 的个体-帧
IoU     : TP 上 IoU3D 的均值（已检出对象的定位质量）
Dev     : TP 上底面中心 XY 误差 dev_m 的中位数（m）
MR      : 1 - N_TP / N_matched，即“无效框率”（precision 的补，越低越好）
S_t     : 0.4*IoU + 0.3*(1 - clip(Dev,0,2)/2) + 0.3*(1 - MR)
          某帧无 TP 时，按最差情况记 S_t=0（IoU=0, Dev=2m, MR=1）。

逐帧 S_t 减去同场景 static 基线均值，即为该帧的感知收益 ΔS_t。

用法示例：
    python -m rslaa.rslaa_plot_score_trends_n --root "...\\Results\\scene_001"
    python -m rslaa.rslaa_plot_score_trends_n --root "...\\Results"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

METHOD_ORDER = ["mpc", "marl_match", "marl_mismatch"]
METHOD_LABEL = {
    "mpc": "MPC",
    "marl_match": "MARL (matched)",
    "marl_mismatch": "MARL (mismatched)",
}
METHOD_COLOR = {
    "mpc": "#74408F",
    "marl_match": "#E4765B",
    "marl_mismatch": "#FBB040",
}

RUN_MPC = re.compile(r"eval_mpc_w(\d+)_pos(\d+)_a(\d+)")
RUN_MARL = re.compile(r"eval_marl_w(\d+)_pos(\d+)_a(\d+)_m(\d+)")

# 方案 B 常量
TP_IOU = 0.15
DEV_CAP = 2.0
W_IOU, W_DEV, W_MR = 0.4, 0.3, 0.3
RAW_NAME = "evals_raw.csv"


def parse_run(eval_dir: Path):
    """解析评估目录名 -> (method, w, pos, ang)。"""
    name = eval_dir.name
    if name == "eval_static":
        return ("static", None, None, None)
    m = RUN_MPC.match(name)
    if m:
        w, pos, ang = m.groups()
        return ("mpc", int(w), int(pos) / 10.0, int(ang))
    m = RUN_MARL.match(name)
    if m:
        w, pos, ang, mm = m.groups()
        method = "marl_match" if int(w) == int(mm) else "marl_mismatch"
        return (method, int(w), int(pos) / 10.0, int(ang))
    return (None, None, None, None)


def _read_raw(eval_dir: Path) -> pd.DataFrame | None:
    """读取逐个体评估记录；缺列/空文件返回 None。"""
    raw = eval_dir / RAW_NAME
    if not raw.is_file():
        return None
    try:
        df = pd.read_csv(raw, encoding="utf-8-sig",
                         usecols=["t", "iou3d", "dev_m", "matched"])
    except (ValueError, pd.errors.EmptyDataError):
        return None
    if df.empty:
        return None
    return df


def _with_tp(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "t": df["t"].to_numpy(dtype=float),
        "iou3d": df["iou3d"].to_numpy(dtype=float),
        "dev_m": df["dev_m"].to_numpy(dtype=float),
        "matched": (df["matched"].to_numpy(dtype=float) > 0).astype(float),
    })
    d["tp"] = ((d["matched"] > 0) & (d["iou3d"] > TP_IOU)).astype(float)
    d["iou_tp"] = d["iou3d"].where(d["tp"] > 0)
    d["dev_tp"] = d["dev_m"].where(d["tp"] > 0)
    return d


def frame_metrics_b(eval_dir: Path) -> pd.DataFrame | None:
    """逐帧方案 B 指标。

    返回列：t, n_gt, n_matched, n_tp, iou_q, dev_q, mr_q, score。
    无 TP 的帧按最差情况填充（IoU=0, Dev=DEV_CAP, MR=1, score=0），
    这样漏检不会被“无定义帧剔除”掩盖。
    """
    df = _read_raw(eval_dir)
    if df is None:
        return None
    d = _with_tp(df)
    g = d.groupby("t", sort=False)
    out = g.agg(
        n_gt=("matched", "size"),
        n_matched=("matched", "sum"),
        n_tp=("tp", "sum"),
        iou_q=("iou_tp", "mean"),
        dev_q=("dev_tp", "median"),
    ).reset_index()

    out["mr_q"] = 1.0 - out["n_tp"] / out["n_matched"].replace(0, np.nan)
    no_tp = out["n_tp"] <= 0
    out.loc[no_tp, "iou_q"] = 0.0
    out.loc[no_tp, "dev_q"] = DEV_CAP
    out.loc[no_tp, "mr_q"] = 1.0
    out["score"] = (W_IOU * out["iou_q"]
                    + W_DEV * (1.0 - out["dev_q"].clip(0, DEV_CAP) / DEV_CAP)
                    + W_MR * (1.0 - out["mr_q"]))
    out.loc[no_tp, "score"] = 0.0
    return out


def run_metrics_b(eval_dir: Path) -> dict | None:
    """整条 run 的方案 B 标量指标（object-frame micro 汇总）。

    IoU 用 TP 上 IoU3D 的均值；Dev 用 TP 上 dev_m 的中位数；
    MR = 1 - N_TP/N_matched。无 TP 时按最差情况返回。
    """
    df = _read_raw(eval_dir)
    if df is None:
        return None
    d = _with_tp(df)
    n_matched = int(d["matched"].sum())
    n_tp = int(d["tp"].sum())
    n_gt = int(len(d))
    if n_tp == 0:
        return dict(iou_q=0.0, dev_q=DEV_CAP, mr_q=1.0, score=0.0,
                    n_tp=0, n_matched=n_matched, n_gt=n_gt)
    iou_q = float(d.loc[d["tp"] > 0, "iou3d"].mean())
    dev_q = float(d.loc[d["tp"] > 0, "dev_m"].median())
    mr_q = float(1.0 - n_tp / max(n_matched, 1))
    score = (W_IOU * iou_q
             + W_DEV * (1.0 - min(max(dev_q, 0.0), DEV_CAP) / DEV_CAP)
             + W_MR * (1.0 - mr_q))
    return dict(iou_q=iou_q, dev_q=dev_q, mr_q=mr_q, score=float(score),
                n_tp=n_tp, n_matched=n_matched, n_gt=n_gt)


def iter_eval_dirs(root: Path):
    """递归产出含 evals_raw.csv 的评估目录（跳过 back up）。"""
    for raw_path in sorted(root.rglob(RAW_NAME)):
        eval_dir = raw_path.parent
        if any("back up" in part for part in eval_dir.parts):
            continue
        yield eval_dir


def collect(root: Path):
    """读取全部评估目录，返回逐帧 S_t、方法/因素信息与静态基线。"""
    runs = []
    for eval_dir in iter_eval_dirs(root):
        method, w, pos, ang = parse_run(eval_dir)
        if method is None:
            continue
        fm = frame_metrics_b(eval_dir)
        if fm is None or fm.empty:
            continue
        runs.append({"method": method, "w": w, "pos": pos, "ang": ang,
                     "scenario": str(eval_dir.parent), "df": fm})
    if not runs:
        return None

    baseline = {}
    for run in runs:
        if run["method"] != "static":
            continue
        s = run["df"]["score"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size:
            baseline.setdefault(run["scenario"], []).append(float(s.mean()))
    baseline = {k: float(np.mean(v)) for k, v in baseline.items()}
    global_base = float(np.mean(list(baseline.values()))) if baseline else np.nan

    for run in runs:
        run["delta"] = (run["df"]["score"].to_numpy(dtype=float)
                        - baseline.get(run["scenario"], global_base))

    return {"runs": runs, "baseline": baseline, "global_base": global_base}


def series_for_factor(runs, factor):
    """按因素值分组，返回 {method: {x: {frames: [...], run_means: [...]}}}。"""
    out = {m: {} for m in METHOD_ORDER}
    for run in runs:
        if run["method"] not in out:
            continue
        x = run[factor]
        d = run["delta"]
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        slot = out[run["method"]].setdefault(x, {"frames": [], "run_means": []})
        slot["frames"].append(d)
        slot["run_means"].append(float(d.mean()))
    return out


def bootstrap_band(xs, run_means, degree=1, n_boot=1000, seed=0):
    """对每个因素值的 run 均值做 bootstrap，给出拟合趋势的 95% 置信带。"""
    rng = np.random.default_rng(seed)
    grid = np.linspace(min(xs), max(xs), 60)
    if len(xs) < 2:
        return grid, None, None
    preds = []
    for _ in range(n_boot):
        ys = [rng.choice(run_means[x], size=len(run_means[x]), replace=True).mean()
              for x in xs]
        coef = np.polyfit(xs, ys, degree)
        preds.append(np.polyval(coef, grid))
    preds = np.array(preds)
    return grid, np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def draw_factor_panel(ax, runs, factor, xlabel, letter, show_legend):
    data = series_for_factor(runs, factor)
    xs_all = sorted({x for m in data for x in data[m]})
    if not xs_all:
        return

    centers = {x: i for i, x in enumerate(xs_all)}
    lane_width = 0.24
    offsets = {m: (i - (len(METHOD_ORDER) - 1) / 2) * lane_width
               for i, m in enumerate(METHOD_ORDER)}

    lo_vals, hi_vals = [], []
    for method in METHOD_ORDER:
        if not data[method]:
            continue
        values = sorted(data[method])
        xs = np.array([centers[v] + offsets[method] for v in values])
        means, q25, q75 = [], [], []
        run_means = {}
        for v in values:
            frames = np.concatenate(data[method][v]["frames"])
            means.append(frames.mean())
            q25.append(np.percentile(frames, 25))
            q75.append(np.percentile(frames, 75))
            run_means[centers[v] + offsets[method]] = np.asarray(
                data[method][v]["run_means"])
        means, q25, q75 = map(np.asarray, (means, q25, q75))
        color = METHOD_COLOR[method]

        grid, band_lo, band_hi = bootstrap_band(xs, run_means)
        if band_lo is not None:
            ax.fill_between(grid, band_lo, band_hi, color=color, alpha=0.18,
                            linewidth=0, zorder=1)
            coef = np.polyfit(xs, means, 1)
            ax.plot(grid, np.polyval(coef, grid), color=color, linewidth=3, zorder=2)

        ax.errorbar(
            xs, means,
            yerr=[means - q25, q75 - means],
            fmt="o", markersize=19, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=1.4, ecolor="#8C8C8C", elinewidth=2.0,
            capsize=6, capthick=2.0, zorder=4,
        )
        lo_vals.extend(q25.tolist())
        hi_vals.extend(q75.tolist())

    ax.axhline(0.0, color="#7F7F7F", linestyle=(0, (4, 4)), linewidth=1.5, zorder=0)

    low = min(min(lo_vals), 0.0)
    high = max(max(hi_vals), 0.0)
    pad = 0.16 * (high - low) if high > low else 0.05
    ax.set_ylim(low - pad, high + pad)

    ax.set_xticks(list(range(len(xs_all))))
    ax.set_xticklabels([f"{x:g}" for x in xs_all], fontsize=22)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _pos: "0" if abs(v) < 1e-9 else f"{v:g}")
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta S_t$ vs. static baseline")
    ax.set_xlim(-0.55, len(xs_all) - 0.45)
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=26,
            fontweight="bold", va="bottom", ha="right")

    if show_legend:
        handles = [Line2D([], [], color=METHOD_COLOR[m], marker="o", markersize=10,
                          linewidth=3, label=METHOD_LABEL[m]) for m in METHOD_ORDER]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=17,
                  handlelength=2.2, borderaxespad=0.3)


def main():
    figures_dir = Path("figures")
    parser = argparse.ArgumentParser(
        description="Scheme-B S_t gain vs window / pos / angle margin")
    parser.add_argument("--root", default="Results/scene_001",
                        help="Results 根目录 / scene_00x / traffic_0x")
    parser.add_argument("--out", default=str(figures_dir / "fig_score_trends_n.png"),
                        help="输出图片路径")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"目录不存在: {root}")

    collected = collect(root)
    if collected is None:
        raise SystemExit(f"未在 {root} 下找到评估结果 ({RAW_NAME})")
    runs = collected["runs"]
    if not collected["baseline"]:
        raise SystemExit(f"{root} 中缺少 static 结果，无法计算收益基线")

    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "dejavusans",
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 17,
        "axes.linewidth": 1.4,
    })

    fig, axes = plt.subplots(1, 3, figsize=(24, 8.2))
    panels = [
        ("w", "Window length (min)", "a"),
        ("pos", "Position margin (m)", "b"),
        ("ang", "Angle margin (°)", "c"),
    ]
    for ax, (factor, xlabel, letter) in zip(axes, panels):
        draw_factor_panel(ax, runs, factor, xlabel, letter,
                          show_legend=(letter == "a"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"root: {root}")
    print(f"runs: {len(runs)}, scenarios with static baseline: "
          f"{len(collected['baseline'])}")
    for method in METHOD_ORDER:
        ds = [r["delta"][np.isfinite(r["delta"])] for r in runs
              if r["method"] == method]
        if ds:
            d = np.concatenate(ds)
            print(f"  {method:14s} frames={d.size:7d} "
                  f"delta_St_mean={d.mean():+.5f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

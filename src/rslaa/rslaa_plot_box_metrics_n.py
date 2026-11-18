# -*- coding: utf-8 -*-
"""方案 B 口径下逐帧相对静态基线的 IoU / MR / Dev 差值箱线组图 —— 不覆盖原脚本。

方案 B 指标（基于 evals_raw.csv 的逐个体记录，逐帧计算）
--------------------------------------------------------
TP      : matched==1 且 IoU3D > 0.15
IoU     : 该帧 TP 上 IoU3D 的均值
Dev     : 该帧 TP 上 dev_m 的中位数 (m)
MR      : 该帧 1 - N_TP / N_matched（无效框率，越低越好）
无 TP 的帧按最差情况填充（IoU=0, Dev=2m, MR=1），避免把漏检帧当作“无定义”剔除。

Δ = 逐帧指标 − 同场景 static 逐帧均值；箱体为各方法全部 run 的逐帧 Δ 汇总。

用法示例：
    python -m rslaa.rslaa_plot_box_metrics_n --root "...\\Results\\scene_001"
    python -m rslaa.rslaa_plot_box_metrics_n --root "...\\Results"
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rslaa.rslaa_plot_score_trends_n import (  # noqa: E402
    parse_run, frame_metrics_b, iter_eval_dirs,
)

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - scipy 缺失时退化为均匀抖动
    gaussian_kde = None

METRICS = [
    ("iou_q", r"$\Delta$IoU"),
    ("mr_q", r"$\Delta$IvR"),
    ("dev_q", r"$\Delta$Dev (m)"),
]
COLS = [c for c, _ in METRICS]

METHOD_ORDER = ["mpc", "marl_match", "marl_mismatch"]
METHOD_LABEL = {
    "mpc": "MPC",
    "marl_match": "MARL\n(match)",
    "marl_mismatch": "MARL\n(mismatch)",
}
METHOD_COLOR = {
    "mpc": "#74408F",
    "marl_match": "#E4765B",
    "marl_mismatch": "#FBB040",
}

CIRCLE_SIZE = 2000    # 中位数圆圈面积 (pt^2)，明显小于背景灰色散点云宽度
MEDIAN_HALF = 0.26    # 中位横线半宽 (x 轴数据单位)
CAP_HALF = 0.10       # 上下分位横线半宽，短线但更粗
WHISKER_WIDTH = 2.6   # 垂直须线宽度
CAP_WIDTH = 3.4       # 上下分位横线宽度


def collect(root: Path):
    """读取 root 下全部逐帧方案 B 指标，并按场景扣减 static 基线。"""
    raw: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {c: [] for c in COLS}
    )
    used: list[Path] = []

    for eval_dir in iter_eval_dirs(root):
        method, _w, _pos, _ang = parse_run(eval_dir)
        if method is None:
            continue
        fm = frame_metrics_b(eval_dir)
        if fm is None or fm.empty:
            continue
        scenario = str(eval_dir.parent)
        for col in COLS:
            raw[(method, scenario)][col].append(fm[col].to_numpy(dtype=float))
        used.append(eval_dir)

    static_by_scenario: dict[str, dict[str, float]] = {}
    global_static: dict[str, float] = {}
    for (method, scenario), cols in raw.items():
        if method != "static":
            continue
        static_by_scenario[scenario] = {}
        for col in COLS:
            arr = np.concatenate(cols[col]) if cols[col] else np.array([])
            arr = arr[np.isfinite(arr)]
            static_by_scenario[scenario][col] = (
                float(arr.mean()) if arr.size else np.nan)
    for col in COLS:
        vals = [static_by_scenario[s][col] for s in static_by_scenario
                if np.isfinite(static_by_scenario[s][col])]
        global_static[col] = float(np.mean(vals)) if vals else np.nan

    pooled: dict[str, dict[str, np.ndarray]] = {
        m: {c: np.array([]) for c in COLS} for m in METHOD_ORDER
    }
    missing_baseline = 0
    for method in METHOD_ORDER:
        for col in COLS:
            deltas = []
            for (m, scenario), cols in raw.items():
                if m != method or not cols[col]:
                    continue
                base = static_by_scenario.get(scenario, {}).get(col, np.nan)
                if not np.isfinite(base):
                    base = global_static[col]
                    missing_baseline += 1
                if not np.isfinite(base):
                    continue
                deltas.append(np.concatenate(cols[col]) - base)
            pooled[method][col] = (np.concatenate(deltas)
                                   if deltas else np.array([]))

    return pooled, used, static_by_scenario, missing_baseline


def violin_jitter(vals: np.ndarray, rng, half_width: float = 0.40) -> np.ndarray:
    """按局部密度给出抖动宽度，使散点呈小提琴状分布。"""
    n = vals.size
    if n < 10 or np.nanstd(vals) == 0 or gaussian_kde is None:
        return rng.uniform(-half_width, half_width, size=n)
    kde = gaussian_kde(vals, bw_method=0.25)
    dens = kde(vals)
    dens = np.clip(dens / np.nanmax(dens), 1e-3, None)
    return rng.uniform(-1.0, 1.0, size=n) * half_width * dens


def box_stats(vals: np.ndarray, whis: float = 1.5):
    """按 1.5 IQR 规则返回 (下须, Q1, 中位数, Q3, 上须)。"""
    q1, med, q3 = np.percentile(vals, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    inner = vals[(vals >= q1 - whis * iqr) & (vals <= q3 + whis * iqr)]
    lo = float(inner.min()) if inner.size else float(q1)
    hi = float(inner.max()) if inner.size else float(q3)
    return lo, float(q1), float(med), float(q3), hi


def draw_panel(ax, pooled, col, ylabel, letter, max_points, rng, show_legend):
    positions = np.arange(len(METHOD_ORDER))
    series = []

    for i, method in enumerate(METHOD_ORDER):
        vals = pooled[method][col]
        vals = vals[np.isfinite(vals)]
        series.append(vals)

        if vals.size:
            n = min(vals.size, max_points)
            idx = rng.choice(vals.size, size=n, replace=False)
            sub = vals[idx]
            jitter = violin_jitter(sub, rng)
            ax.scatter(i + jitter, sub, s=40, color="#C0C0C0", alpha=0.32,
                       linewidths=0, zorder=1)

            lo, _q1, med, _q3, hi = box_stats(vals)
            # 垂直须线
            ax.plot([i, i], [lo, hi], color="#4D4D4D", linewidth=WHISKER_WIDTH,
                    solid_capstyle="butt", zorder=3)
            # 上下分位横线：更短、更粗
            for y in (lo, hi):
                ax.plot([i - CAP_HALF, i + CAP_HALF], [y, y], color="#4D4D4D",
                        linewidth=CAP_WIDTH, solid_capstyle="butt", zorder=3)
            # 中心圆圈 + 中位横线
            ax.scatter([i], [med], s=CIRCLE_SIZE, color=METHOD_COLOR[method],
                       alpha=0.95, linewidths=0, zorder=4)
            ax.plot([i - MEDIAN_HALF, i + MEDIAN_HALF], [med, med], color="black",
                    linewidth=2.4, solid_capstyle="butt", zorder=5)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=2, zorder=2)

    all_vals = np.concatenate(series) if any(s.size for s in series) else np.array([0.0])
    lo, hi = np.percentile(all_vals, [1.0, 99.0])
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.10 * (hi - lo) if hi > lo else 0.5
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=28)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.75, len(METHOD_ORDER) - 0.25)
    ax.tick_params(direction="out", length=5, width=1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=28,
            fontweight="bold", va="bottom", ha="right")

    if show_legend:
        handle = Line2D([], [], color="black", linestyle="--", linewidth=2,
                        label="Static baseline")
        ax.legend(handles=[handle], loc="upper right", frameon=False, fontsize=24,
                  handlelength=2.4, borderaxespad=0.4)


def main():
    figures_dir = Path("figures")
    parser = argparse.ArgumentParser(
        description="Scheme-B frame-wise metric deltas vs static baseline")
    parser.add_argument("--root", default="Results/scene_001",
                        help="Results 根目录 / scene_00x / traffic_0x")
    parser.add_argument("--out",
                        default=str(figures_dir / "fig_box_delta_iou_mr_dev_n.png"),
                        help="输出图片路径")
    parser.add_argument("--max-points", type=int, default=4000,
                        help="每个箱体背景散点的最大抽样数量")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"目录不存在: {root}")

    pooled, used, static_by_scenario, missing_baseline = collect(root)
    if not used:
        raise SystemExit(f"未在 {root} 下找到评估结果 (evals_raw.csv)")
    if not static_by_scenario:
        raise SystemExit(f"{root} 中缺少 static 结果，无法计算差值基线")

    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "dejavusans",
        "font.size": 28,
        "axes.labelsize": 28,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 24,
        "axes.linewidth": 2,
    })

    fig, axes = plt.subplots(1, 3, figsize=(28, 8.5))
    rng = np.random.default_rng(42)

    for ax, (col, label), letter in zip(axes, METRICS, "abc"):
        draw_panel(ax, pooled, col, label, letter, args.max_points, rng,
                   show_legend=(col == "iou_q"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"root: {root}")
    print(f"eval dirs: {len(used)}, scenarios with static baseline: "
          f"{len(static_by_scenario)}, missing_baseline={missing_baseline}")
    for method in METHOD_ORDER:
        n = {c: int(np.isfinite(pooled[method][c]).sum()) for c in COLS}
        mean = {c: float(np.nanmean(pooled[method][c])) for c in COLS}
        shown = {c: round(mean[c], 4) for c in COLS}
        print(f"  {method:14s} n={n} delta_mean={shown}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""方案 B 口径下各方法相对静态基线的指标变化率柱状图 —— 不覆盖原脚本。

方案 B 指标（基于 evals_raw.csv 的逐个体记录，整条 run 汇总）
------------------------------------------------------------
TP      : matched==1 且 IoU3D > 0.15
IoU     : TP 上 IoU3D 的均值
Dev     : TP 上底面中心 XY 误差 dev_m 的中位数 (m)
MR      : 1 - N_TP / N_matched（无效框率，越低越好；不是漏检率）
S_t     : 0.4*IoU + 0.3*(1 - clip(Dev,0,2)/2) + 0.3*(1 - MR)

Δ% = (指标_动态 − 指标_静态基线) / 指标_静态基线 × 100
静态基线取同一交通场景内 static 运行的标量指标；误差棒为各 run 均值的 bootstrap 标准误。

用法示例：
    python -m rslaa.rslaa_plot_bar_relative_n --root "...\\Results\\scene_001"
    python -m rslaa.rslaa_plot_bar_relative_n --root "...\\Results"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rslaa.rslaa_plot_score_trends_n import (  # noqa: E402
    parse_run, run_metrics_b, iter_eval_dirs,
)

METRICS = [
    ("iou_q", "ΔIoU / IoU$_{static}$ (%)"),
    ("mr_q", "ΔIvR / IvR$_{static}$ (%)"),
    ("dev_q", "ΔDev / Dev$_{static}$ (%)"),
]
METHOD_ORDER = ["mpc", "marl_match", "marl_mismatch"]
METHOD_LABEL = {"mpc": "MPC", "marl_match": "MARL\n(matched)",
                "marl_mismatch": "MARL\n(mismatched)"}
METHOD_COLOR = {"mpc": "#74408F", "marl_match": "#E4765B",
                "marl_mismatch": "#FBB040"}


def collect(root: Path):
    """递归读取评估结果，返回每条 run 的方案 B 标量指标与所属场景。"""
    runs = []
    for eval_dir in iter_eval_dirs(root):
        method, w, pos, ang = parse_run(eval_dir)
        if method is None:
            continue
        met = run_metrics_b(eval_dir)
        if met is None:
            continue
        runs.append({"method": method, "scenario": str(eval_dir.parent), **met})
    return runs


def analyse(root: Path):
    """聚合各 run 指标并计算相对静态基线的变化率。"""
    runs = collect(root)
    if not runs:
        raise SystemExit(f"未在 {root} 下找到评估结果 (evals_raw.csv)")

    baseline = {}
    for run in runs:
        if run["method"] != "static":
            continue
        for col, _ in METRICS:
            baseline.setdefault(run["scenario"], {}).setdefault(col, []).append(
                float(run[col]))
    baseline = {sc: {c: float(np.mean(v)) for c, v in d.items()}
                for sc, d in baseline.items()}

    stats = {m: {} for m in METHOD_ORDER}
    rng = np.random.default_rng(0)
    for col, _ in METRICS:
        for method in METHOD_ORDER:
            abs_runs, base_runs = [], []
            for run in runs:
                if run["method"] != method:
                    continue
                base = baseline.get(run["scenario"], {}).get(col)
                if base is None or not np.isfinite(base):
                    continue
                val = float(run[col])
                if not np.isfinite(val):
                    continue
                base_runs.append(float(base))
                abs_runs.append(val - base)
            if not abs_runs:
                continue
            abs_arr = np.asarray(abs_runs)
            base_arr = np.asarray(base_runs)
            mean_abs = float(abs_arr.mean())
            mean_base = float(base_arr.mean())
            pct = (mean_abs / mean_base * 100.0
                   if abs(mean_base) > 1e-12 else np.nan)

            sem = 0.0
            if abs_arr.size > 1:
                idx = rng.integers(0, abs_arr.size, size=(400, abs_arr.size))
                ratios = (abs_arr[idx].mean(axis=1)
                          / base_arr[idx].mean(axis=1) * 100.0)
                sem = float(np.nanstd(ratios, ddof=1))

            stats[method][col] = {"pct": pct, "sem": sem, "abs": mean_abs,
                                  "base": mean_base, "n_runs": int(abs_arr.size)}
    return stats, len(runs), len(baseline)


def draw_panel(ax, stats, col, ylabel, letter):
    positions = np.arange(len(METHOD_ORDER))
    heights, errors = [], []
    for method in METHOD_ORDER:
        s = stats[method].get(col, {"pct": 0.0, "sem": 0.0, "abs": 0.0})
        heights.append(0.0 if not np.isfinite(s["pct"]) else s["pct"])
        errors.append(s["sem"])
    heights = np.asarray(heights, dtype=float)
    errors = np.asarray(errors, dtype=float)

    ax.bar(positions, heights, width=0.62,
           color=[METHOD_COLOR[m] for m in METHOD_ORDER],
           edgecolor="white", linewidth=1.0, alpha=0.92, zorder=2)
    ax.errorbar(positions, heights, yerr=errors, fmt="none", ecolor="#4C4C4C",
                elinewidth=3.0, capsize=7, capthick=3.0, zorder=3)
    ax.axhline(0.0, color="#7F7F7F", linewidth=1.2, zorder=1)

    for x, method, h, e in zip(positions, METHOD_ORDER, heights, errors):
        s = stats[method].get(col, {"abs": 0.0, "pct": np.nan})
        top = h + e if h >= 0 else h - e
        pct_txt = "NA" if not np.isfinite(s["pct"]) else f"{s['pct']:+.1f}%"
        ax.annotate(f"{s['abs']:+.4f}\n({pct_txt})",
                    xy=(x, top), xytext=(0, 8 if h >= 0 else -8),
                    textcoords="offset points", ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=19,
                    linespacing=1.15, color="#262626", zorder=4)

    lo = min(0.0, np.min(heights - errors))
    hi = max(0.0, np.max(heights + errors))
    pad = 0.35 * (hi - lo) if hi > lo else 1.0
    ax.set_ylim(lo - pad * 0.55, hi + pad)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=20)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, len(METHOD_ORDER) - 0.45)
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="bottom", ha="right")


def main():
    figures_dir = Path("figures")
    parser = argparse.ArgumentParser(
        description="Scheme-B relative metric change vs static baseline")
    parser.add_argument("--root", default="Results/scene_001",
                        help="Results 根目录 / scene_00x / traffic_0x")
    parser.add_argument("--out", default=str(figures_dir / "fig_bar_relative_change_n.png"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"目录不存在: {root}")

    stats, n_runs, n_sc = analyse(root)

    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "dejavusans",
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.linewidth": 1.4,
    })

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (col, label), letter in zip(axes, METRICS, "abc"):
        draw_panel(ax, stats, col, label, letter)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"root: {root}, runs={n_runs}, scenarios={n_sc}")
    labels = {"iou_q": "IoU", "mr_q": "IvR", "dev_q": "Dev"}
    for col, _ in METRICS:
        parts = []
        for m in METHOD_ORDER:
            s = stats[m].get(col)
            if s:
                parts.append(f"{m}: {s['abs']:+.4f} ({s['pct']:+.2f}% ± "
                             f"{s['sem']:.2f}, n={s['n_runs']})")
        print(f"  {labels[col]:8s} " + " | ".join(parts))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

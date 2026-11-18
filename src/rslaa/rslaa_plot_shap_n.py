# -*- coding: utf-8 -*-
"""方案 B 口径下的 SHAP 分析（XGBoost + TreeSHAP）—— 不覆盖原脚本。

因变量
------
每条 run 的平均 ΔS_t（方案 B）：
    TP   : matched==1 且 IoU3D > 0.15
    IoU  : TP 上 IoU3D 均值
    Dev  : TP 上 dev_m 中位数
    MR   : 1 - N_TP / N_matched（无效框率）
    S_t  : 0.4*IoU + 0.3*(1 - clip(Dev,0,2)/2) + 0.3*(1 - MR)
    ΔS_t = S_t(run) - S_t(同场景 static 基线)

SHAP
----
优先使用 XGBoost 的 TreeSHAP（booster.predict(pred_contribs=True)），
不依赖额外的 shap 包。若当前环境没有安装 xgboost，则回退到纯 NumPy
PolyRidge + 精确 Shapley 枚举，仅用于无依赖环境预览。

展示内容与原 rslaa_plot_shap.py 一致：左侧 SHAP 蜂群图 + 重要性条形，
右侧前 6 个关键特征的 SHAP 散点图。

用法示例：
    python -m rslaa.rslaa_plot_shap_n --root "...\\Results" --method marl_match
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rslaa.rslaa_plot_score_trends_n import collect  # noqa: E402

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:  # pragma: no cover - 无 xgboost 时回退
    xgb = None
    HAS_XGB = False

CMAP = LinearSegmentedColormap.from_list(
    "purple_coral_amber", ["#74408F", "#E4765B", "#FBB040"]
)

INTERSECTIONS = {
    "scene_001": 1, "scene_002": 0, "scene_003": 2, "scene_004": 2, "scene_005": 2,
    "scene_006": 2, "scene_007": 1, "scene_008": 1, "scene_009": 2, "scene_010": 2,
}

FEATURES = [
    ("road_width", "Road width (m)"),
    ("lane_count", "Lane count"),
    ("lane_width", "Lane width (m)"),
    ("is_curved", "Is curved"),
    ("n_intersections", "Intersections"),
    ("obs_mean_height", "Obstacle mean height (m)"),
    ("obs_lon_std", "Obstacle lon. std (m)"),
    ("obs_lon_iqr", "Obstacle lon. IQR (m)"),
    ("obs_lon_range", "Obstacle lon. range (m)"),
    ("traffic_volume", "Traffic volume (veh/h)"),
    ("mean_speed", "Mean speed (km/h)"),
    ("ratio_car_truck", "Car/Truck ratio"),
    ("ratio_motor_vru", "Motor/VRU ratio"),
    ("window", "Window length (min)"),
    ("pos_margin", "Position margin (m)"),
    ("ang_margin", "Angle margin (°)"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class PolyRidge:
    """无依赖回退模型：标准化 + 二次项的岭回归。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _design(self, X):
        Z = (X - self.mu) / self.sd
        return np.hstack([np.ones((Z.shape[0], 1)), Z, Z ** 2])

    def fit(self, X, y):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        A = self._design(X)
        reg = self.alpha * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
        return self

    def predict(self, X):
        return self._design(np.asarray(X, dtype=float)) @ self.coef

    def score(self, X, y):
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def build_dataset(root: Path, method: str):
    """返回特征矩阵、因变量（每 run 平均 ΔS_t）与特征说明。"""
    collected = collect(root)
    if collected is None:
        raise SystemExit(f"未在 {root} 下找到评估结果 (evals_raw.csv)")

    rows = []
    for run in collected["runs"]:
        if run["method"] != method:
            continue
        d = run["delta"][np.isfinite(run["delta"])]
        if d.size == 0:
            continue
        scenario = Path(run["scenario"])
        road_file = scenario / "road_params.json"
        traffic_file = scenario / "traffic_params.json"
        if not (road_file.exists() and traffic_file.exists()):
            continue
        road = load_json(road_file)
        traffic = load_json(traffic_file)["full"]
        scene = scenario.parent.name

        rows.append({
            "delta_score": float(d.mean()),
            "road_width": road["road"]["road_width_m"],
            "lane_count": road["road"]["lane_count"],
            "lane_width": road["road"]["lane_width_m"],
            "is_curved": int(bool(road["road"]["is_curved"])),
            "n_intersections": INTERSECTIONS.get(scene, np.nan),
            "obs_mean_height": road["roadside_obstacle"]["mean_height_m"],
            "obs_lon_std": road["roadside_obstacle"]["longitudinal_std_m"],
            "obs_lon_iqr": road["roadside_obstacle"]["longitudinal_iqr_m"],
            "obs_lon_range": road["roadside_obstacle"]["longitudinal_range_m"],
            "traffic_volume": traffic["volume_veh_per_h"],
            "mean_speed": traffic["mean_speed_kmh"],
            "ratio_car_truck": traffic["ratio_car_truck"],
            "ratio_motor_vru": traffic["ratio_motor_vru"],
            "window": run["w"],
            "pos_margin": run["pos"],
            "ang_margin": run["ang"],
        })
    if not rows:
        raise SystemExit(f"没有可用的 {method} 运行数据")

    df = pd.DataFrame(rows)
    names = [c for c, _ in FEATURES]
    df = df.dropna(subset=names)
    const = [c for c in names if df[c].nunique() <= 1]
    use = [c for c in names if c not in const]
    return df, use, const


def exact_shapley(model, X: np.ndarray) -> np.ndarray:
    """精确 Shapley（均值填充价值函数），仅用于无 xgboost 的回退路径。"""
    n = X.shape[1]
    mean = X.mean(axis=0)
    subsets = np.array(list(itertools.product([0, 1], repeat=n)), dtype=bool)
    codes = subsets.dot(1 << np.arange(n))
    code_to_idx = {int(c): i for i, c in enumerate(codes)}

    preds = np.zeros((X.shape[0], len(subsets)))
    for k, S in enumerate(subsets):
        Xs = np.tile(mean, (X.shape[0], 1))
        Xs[:, S] = X[:, S]
        preds[:, k] = model.predict(Xs)

    phi = np.zeros_like(X, dtype=float)
    for j in range(n):
        for k, S in enumerate(subsets):
            if S[j]:
                continue
            size = int(S.sum())
            w = factorial(size) * factorial(n - size - 1) / factorial(n)
            S2 = S.copy()
            S2[j] = True
            k2 = code_to_idx[int(S2.dot(1 << np.arange(n)))]
            phi[:, j] += w * (preds[:, k2] - preds[:, k])
    return phi


def fit_xgboost(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """原生 XGBoost Booster（不依赖 sklearn），参数针对小样本表格数据。"""
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    params = {
        "objective": "reg:squarederror",
        "eta": 0.03,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "lambda": 5.0,
        "tree_method": "hist",
        "seed": 0,
        "nthread": 4,
    }
    booster = xgb.train(params, dtrain, num_boost_round=150)
    return booster, dtrain


def xgb_tree_shap(booster, dtrain) -> np.ndarray:
    """XGBoost 内置 TreeSHAP：pred_contribs 的最后一列是 base value，舍弃。"""
    contribs = np.asarray(booster.predict(dtrain, pred_contribs=True))
    return contribs[:, :-1]


def r2_score_np(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    figures_dir = Path("figures")
    parser = argparse.ArgumentParser(
        description="Scheme-B SHAP analysis of perception gain (XGBoost TreeSHAP)")
    parser.add_argument("--root", default="Results/scene_001")
    parser.add_argument("--method", default="marl_match",
                        choices=["marl_match", "mpc", "marl_mismatch"])
    parser.add_argument("--out", default=str(figures_dir / "fig_shap_n.png"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.root)
    df, used, const = build_dataset(root, args.method)
    if const:
        print("常量特征（该数据集内无变化，已从 SHAP 中剔除）: " + ", ".join(const))

    X = df[used].to_numpy(dtype=float)
    y = df["delta_score"].to_numpy(dtype=float)
    labels = dict(FEATURES)

    if HAS_XGB:
        booster, dtrain = fit_xgboost(X, y, used)
        r2 = r2_score_np(y, booster.predict(dtrain))
        phi = xgb_tree_shap(booster, dtrain)
        model_name = f"XGBoost TreeSHAP ({xgb.__version__})"
    else:
        print("[警告] 未安装 xgboost，回退到 PolyRidge + 精确 Shapley；"
              "安装 xgboost 后可获得 TreeSHAP 结果。")
        model = PolyRidge(alpha=1.0).fit(X, y)
        r2 = float(model.score(X, y))
        phi = exact_shapley(model, X)
        model_name = "PolyRidge + exact Shapley (fallback)"

    mean_abs = np.abs(phi).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    share = mean_abs / mean_abs.sum() * 100.0

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 20,
        "axes.linewidth": 1.2,
    })

    fig = plt.figure(figsize=(26, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])

    norm = Normalize(vmin=0, vmax=1)
    ys = np.arange(len(order))
    for row, j in enumerate(order):
        ax.barh(row, mean_abs[j], height=0.62, color="#EDEDED", zorder=0)
        color_val = (X[:, j] - X[:, j].min()) / max(np.ptp(X[:, j]), 1e-12)
        jitter = np.random.default_rng(1).uniform(-0.22, 0.22, size=X.shape[0])
        ax.scatter(phi[:, j], row + jitter, c=CMAP(color_val), s=90,
                   edgecolors="white", linewidths=0.6, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([labels[used[j]] for j in order], fontsize=20)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#9E9E9E", linewidth=1.4, linestyle="--", zorder=1)
    ax.set_xlabel("SHAP value (impact on ΔScore)")
    ax.set_title("Mean |Shapley Value| (Feature Importance)", fontsize=22, pad=34)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4)

    x_lo, x_hi = phi.min(), phi.max()
    x_pad = 0.55 * (x_hi - x_lo)
    ax.set_xlim(x_lo - 0.08 * (x_hi - x_lo), x_hi + x_pad)

    for row, j in enumerate(order):
        ax.text(x_hi + 0.28 * x_pad, row - 0.12, f"{mean_abs[j]:.4f}",
                fontsize=17, va="center", ha="left", color="#262626")
        ax.text(x_hi + 0.28 * x_pad, row + 0.22, f"{share[j]:.2f}%",
                fontsize=17, va="center", ha="left", color="#E4765B")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cax = ax.inset_axes([1.03, 0.52, 0.022, 0.34])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([])
    cbar.set_label("Feature value\n(Low → High)", fontsize=14, labelpad=10)

    # 右侧：前 6 个关键特征
    right = gs[0, 1].subgridspec(2, 3, hspace=0.55, wspace=0.42)
    top = order[:6]
    for k, j in enumerate(top):
        axs = fig.add_subplot(right[k // 3, k % 3])
        xv = X[:, j]
        sv = phi[:, j]
        cv = (xv - xv.min()) / max(np.ptp(xv), 1e-12)
        axs.scatter(xv, sv, c=CMAP(cv), s=95, edgecolors="white",
                    linewidths=0.7, zorder=3)
        if np.unique(xv).size >= 3:
            deg = 2 if np.unique(xv).size >= 4 else 1
            coef = np.polyfit(xv, sv, deg)
            grid = np.linspace(xv.min(), xv.max(), 100)
            axs.plot(grid, np.polyval(coef, grid), color="#4C4C4C", linewidth=2.4,
                     zorder=2)
        med = float(np.median(xv))
        axs.axvline(med, color="#4C4C4C", linestyle="--", linewidth=1.6, zorder=1)
        axs.set_title(labels[used[j]], fontsize=16, pad=6)
        if k % 3 == 0:
            axs.set_ylabel("SHAP value", fontsize=16)
        axs.text(0.04, 0.96, f"Median: {med:g}", transform=axs.transAxes,
                 fontsize=13, va="top", ha="left", color="#262626",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="none", alpha=0.75))
        axs.axhline(0.0, color="#CFCFCF", linewidth=1.0, zorder=0)
        axs.tick_params(labelsize=14, direction="out", length=3)
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"root: {root}, method={args.method}, samples={len(df)}, "
          f"features={len(used)}, model={model_name}, R2(train)={r2:.3f}")
    for j in order:
        print(f"  {labels[used[j]]:28s} mean|SHAP|={mean_abs[j]:.5f} "
              f"({share[j]:5.2f}%)")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

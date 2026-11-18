# -*- coding: utf-8 -*-
"""论文组图: 同一场景 4 种 LiDAR 位姿配置 x 6 个时间帧的仿真点云 + 感知结果.

行: Static / MPC / MARL-M (matched window) / MARL-DM (mismatched window)
列: 6 个 MARL(匹配) 优于静态基线的时间帧

每幅子图: 仿真点云 (intensity, 灰度) + 感知 BBox (红) + GT BBox (绿),
倾斜鸟瞰正交投影, 所有子图视角/范围完全一致.

用法:
    python -m rslaa.rslaa_plot_frame_grid
    python -m rslaa.rslaa_plot_frame_grid --no-cache --dpi 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from rslaa.common import (build_builder_and_sim, load_lidar_pose_seq_json,
                               make_sim, make_specs)  # noqa: E402
from rslaa.sim_pool import frame_seed  # noqa: E402
from lidarsim.lshape import evaluate_frame  # noqa: E402

TDIR = os.path.join("Results", "scene_001", "traffic_01")
CACHE_DIR = os.path.join("Results", "cache", "frame_grid")
OUT_DIR = os.path.join(os.path.dirname(_REPO), "output", "frame_grid")

# 行定义: (缓存键, 显示名, pose 文件后缀)
ROWS = [
    ("static",  "Static",                     "static"),
    ("mpc",     "MPC",                        "mpc_w15_pos02_a15"),
    ("marl_m",  "MARL-M\n(matched)",          "marl_w15_pos02_a15_m15"),
    ("marl_dm", "MARL-DM\n(mismatched)",      "marl_w15_pos02_a15_m30"),
]

# 6 个时间帧: MARL(匹配) 逐帧 IoU 明显高于静态基线, 且失配窗口表现更差
FRAMES = [1032.0, 11933.0, 11997.0, 13213.0, 17125.0, 18205.0]

AZIM_DEG = -55.0          # 方位角 (0 = 从 +x 方向看)
ELEV_DEG = 40.0           # 俯仰角 (90 = 正俯视)
POINT_SIZE = 2.0
INTENSITY_VMAX = 60.0     # None -> 自动取全体 99 分位
GT_COLOR = "#00C000"
PERC_COLOR = "#FF1A1A"


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def pose_file(tdir: str, suffix: str) -> str:
    return os.path.join(tdir, f"lidar_pose_{suffix}.json")


def load_pose_windows(tdir: str, suffix: str) -> list[dict]:
    ps = load_lidar_pose_seq_json(pose_file(tdir, suffix))
    out = []
    for w in ps["windows"]:
        tr = w.get("t_range")
        out.append(dict(index=w.get("index", 0),
                        t_range=None if tr is None else (float(tr[0]), float(tr[1])),
                        poses=np.asarray(w["poses"], np.float64)))
    return ps["presets"], out


def window_for(windows: list[dict], t: float) -> dict:
    for w in windows:
        if w["t_range"] is None or (w["t_range"][0] <= t <= w["t_range"][1]):
            return w
    return windows[-1]


def panel_path(key: str, t: float) -> str:
    return os.path.join(CACHE_DIR, f"{key}_{int(round(t))}.npz")


def simulate_panel(builder, params, tdir: str, key: str, suffix: str, t: float):
    presets, windows = load_pose_windows(tdir, suffix)
    specs = make_specs(presets)
    win = window_for(windows, t)
    sim = make_sim(builder, specs, win["poses"], params, seed=0)
    sim.rng = np.random.default_rng(frame_seed(0, t))   # 与评价脚本逐位一致
    scene, clouds = sim.simulate_frame(float(t))
    ev = evaluate_frame(clouds, scene, min_points=5)

    xyz = np.vstack([c.xyz for c in clouds if len(c.xyz)]).astype(np.float32)
    inten = np.concatenate([c.intensity for c in clouds if len(c.xyz)]).astype(np.float32)
    gt = np.stack([ins.gt_box for ins in ev.instances]) if ev.instances else np.zeros((0, 8, 3))
    fit = np.stack([ins.fit_box for ins in ev.instances if ins.fit_box is not None]) \
        if any(ins.fit_box is not None for ins in ev.instances) else np.zeros((0, 8, 3))
    origins = np.stack([c.origin for c in clouds]).astype(np.float32)

    np.savez_compressed(panel_path(key, t), xyz=xyz, intensity=inten,
                        gt=gt.astype(np.float32), fit=fit.astype(np.float32),
                        origins=origins, iou=ev.mean_iou, dev=ev.mean_dev,
                        n_gt=ev.n_gt, n_matched=ev.n_matched, t=t)
    return dict(xyz=xyz, intensity=inten, gt=gt, fit=fit, origins=origins,
                iou=ev.mean_iou, dev=ev.mean_dev, n_gt=ev.n_gt,
                n_matched=ev.n_matched)


def load_panel(key: str, t: float) -> dict:
    z = np.load(panel_path(key, t))
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# 投影
# ---------------------------------------------------------------------------
def make_proj(azim_deg: float, elev_deg: float):
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    u = np.array([-np.sin(az), np.cos(az), 0.0])
    v = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
    d = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    return u, v, d


BOX_EDGES = [(0, 2), (2, 6), (6, 4), (4, 0),
             (1, 3), (3, 7), (7, 5), (5, 1),
             (0, 1), (2, 3), (4, 5), (6, 7)]


def box_segments(box: np.ndarray, u, v):
    p = np.asarray(box, np.float64)
    su, sv = p @ u, p @ v
    segs = []
    for a, b in BOX_EDGES:
        segs.append([(su[a], sv[a]), (su[b], sv[b])])
    return segs


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def pick_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Arial", "Liberation Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


def render(panels: dict, out_png: str, dpi: int, vmax: float,
           azim_deg: float = AZIM_DEG, elev_deg: float = ELEV_DEG,
           point_size: float = POINT_SIZE, frames: list | None = None):
    frames = list(FRAMES if frames is None else frames)
    font = pick_font()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans"],
        "font.size": 24,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    u, v, d = make_proj(azim_deg, elev_deg)

    # 统一投影范围: 点云 0.2~99.8 分位 + 全部 GT 框/雷达位置 (保证不被裁掉)
    su_pts, sv_pts = [], []
    su_key, sv_key = [], []
    for pnl in panels.values():
        p = pnl["xyz"]
        su_pts.append(p @ u)
        sv_pts.append(p @ v)
        for box in pnl["gt"]:
            su_key.append(box @ u)
            sv_key.append(box @ v)
        su_key.append(pnl["origins"] @ u)
        sv_key.append(pnl["origins"] @ v)
    su_pts, sv_pts = np.concatenate(su_pts), np.concatenate(sv_pts)
    su_key, sv_key = np.concatenate(su_key), np.concatenate(sv_key)
    su0 = min(np.percentile(su_pts, 0.2), float(su_key.min()))
    su1 = max(np.percentile(su_pts, 99.8), float(su_key.max()))
    sv0 = min(np.percentile(sv_pts, 0.2), float(sv_key.min()))
    sv1 = max(np.percentile(sv_pts, 99.8), float(sv_key.max()))
    padu, padv = 0.04 * (su1 - su0), 0.04 * (sv1 - sv0)
    su0, su1, sv0, sv1 = su0 - padu, su1 + padu, sv0 - padv, sv1 + padv

    # 面板宽高比 = 投影数据宽高比 -> 等比例且不留白边
    data_asp = (su1 - su0) / (sv1 - sv0)
    nrow, ncol = len(ROWS), len(frames)
    panel_h = 2.0
    panel_w = panel_h * data_asp
    left, right, top, bottom = 0.080, 0.998, 0.945, 0.070
    fig_w = panel_w * ncol / (right - left)
    fig_h = panel_h * nrow / (top - bottom)
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), dpi=dpi,
                             squeeze=False)
    fig.patch.set_facecolor("white")

    cmap = mpl.colormaps["gray_r"]          # 灰度: 强度越高越深, 白底可见
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    for i, (key, label, _suffix) in enumerate(ROWS):
        for j, t in enumerate(frames):
            ax = axes[i, j]
            pnl = panels[(key, t)]
            xyz = pnl["xyz"]

            su, sv, sd = xyz @ u, xyz @ v, xyz @ d
            order = np.argsort(sd)                 # 远 -> 近
            ax.scatter(su[order], sv[order], c=pnl["intensity"][order],
                       cmap=cmap, norm=norm, s=point_size, marker="o",
                       linewidths=0, alpha=0.95, rasterized=True, zorder=2)

            for box in pnl["gt"]:
                for seg in box_segments(box, u, v):
                    ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                            color=GT_COLOR, lw=2.0, solid_capstyle="round",
                            zorder=5)
            for box in pnl["fit"]:
                for seg in box_segments(box, u, v):
                    ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                            color=PERC_COLOR, lw=1.5, solid_capstyle="round",
                            zorder=6)

            for o in pnl["origins"]:
                ax.scatter([o @ u], [o @ v], s=110, marker="^", c="#0000FF",
                           edgecolors="white", linewidths=1.2, zorder=7)

            ax.set_xlim(su0, su1)
            ax.set_ylim(sv0, sv1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            ax.set_facecolor("white")
            if i == 0:
                ax.set_title(f"t = {int(t)} s", fontsize=22, pad=6)

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=0.018, hspace=0.035)

    # 行标签 (图左侧, 两行文字)
    for i, (_key, label, _suffix) in enumerate(ROWS):
        pos = axes[i, 0].get_position()
        fig.text(left * 0.45, (pos.y0 + pos.y1) * 0.5, label,
                 va="center", ha="center", fontsize=22, linespacing=1.05)

    handles = [
        Line2D([0], [0], color=PERC_COLOR, lw=2.2, label="Perception BBox"),
        Line2D([0], [0], color=GT_COLOR, lw=2.2, label="Ground-truth BBox"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#0000FF",
               markeredgecolor="white", markersize=9, label="LiDAR"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=24, handlelength=2.2, columnspacing=2.2,
               bbox_to_anchor=(0.5, 0.002))

    fig.savefig(out_png, dpi=dpi, facecolor="white")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=TDIR)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "fig_frames_4x6.png"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--frames", default="",
                    help="逗号分隔的时间帧 (默认内置 6 帧)")
    ap.add_argument("--azim", type=float, default=AZIM_DEG)
    ap.add_argument("--elev", type=float, default=ELEV_DEG)
    ap.add_argument("--point-size", type=float, default=POINT_SIZE)
    ap.add_argument("--vmax", type=float, default=0.0,
                    help="强度色标上限; 0 = 自动 (99 分位)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    tdir = args.results_dir
    frames = ([float(x) for x in args.frames.split(",") if x.strip()]
              if args.frames else list(FRAMES))

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    need_builder = args.no_cache or any(
        not os.path.isfile(panel_path(k, t)) for k, _l, _s in ROWS for t in frames)
    builder = params = None
    if need_builder:
        cfg = json.load(open(os.path.join(tdir, "eval_static", "config.json"),
                             encoding="utf-8"))
        print("[Sim] 构建场景 (一次) ...", flush=True)
        t0 = time.time()
        builder, params = build_builder_and_sim(
            cfg["traj_path"], cfg["env_path"], cfg["models_dir"],
            device="auto", verbose=False)
        print(f"[Sim] 场景就绪 {time.time() - t0:.1f}s", flush=True)

    panels = {}
    for key, label, suffix in ROWS:
        for t in frames:
            path = panel_path(key, t)
            if args.no_cache or not os.path.isfile(path):
                t0 = time.time()
                pnl = simulate_panel(builder, params, tdir, key, suffix, t)
                print(f"[Sim] {label:<30s} t={t:8.0f}  pts={len(pnl['xyz']):6d}  "
                      f"IoU={pnl['iou']:.3f}  Dev={pnl['dev']:.2f}  "
                      f"matched={pnl['n_matched']}/{pnl['n_gt']}  "
                      f"{time.time() - t0:.1f}s", flush=True)
            else:
                pnl = load_panel(key, t)
                print(f"[Cache] {label:<28s} t={t:8.0f}  IoU={float(pnl['iou']):.3f}  "
                      f"matched={int(pnl['n_matched'])}/{int(pnl['n_gt'])}", flush=True)
            panels[(key, t)] = pnl

    if args.vmax > 0:
        vmax = float(args.vmax)
    elif INTENSITY_VMAX is not None:
        vmax = float(INTENSITY_VMAX)
    else:
        vmax = float(np.percentile(
            np.concatenate([p["intensity"] for p in panels.values()]), 99.0))
    vmax = max(vmax, 30.0)
    print(f"[Plot] intensity vmax = {vmax:.1f}", flush=True)

    t0 = time.time()
    render(panels, args.out, args.dpi, vmax,
           azim_deg=args.azim, elev_deg=args.elev,
           point_size=args.point_size, frames=frames)
    print(f"[Plot] 已保存 {args.out}  ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

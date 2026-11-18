# -*- coding: utf-8 -*-
"""多 LiDAR 位姿配置替代指标评测 —— 测试基准构建 (v2 指标版).

与 run_benchmark.py 的差异:
  * 替代指标改用 rslaa_bench.metric_n (任务 ROI = 轨迹走廊, 体素级 w 对齐),
    避免原版 ROI 近全盘导致的指标方向反转.
  * 多候选指标计算采用 compute_all_fast + prep_context: 5 个指标共享一次
    soft_visibility, 且与位姿无关的量 (H/d/corridors/roi) 跨候选复用.

不修改 lidarsim 既有代码, 仅复用其仿真/评估 API. 输出目录结构、字段格式
与 run_benchmark.py 完全一致.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidarsim.trajectory import Trajectories
from lidarsim.pointcloud import load_env, make_flat_env
from lidarsim.meshes import ModelLibrary
from lidarsim.scene import SceneBuilder
from lidarsim.config import SimParams, make_lidar
from lidarsim.lidar_sim import LidarSimulator, LidarPose
from lidarsim.lshape import evaluate_frame, sequence_completeness

from rslaa_bench.metrics import OccGrid, pose_internal
from rslaa_bench.metric_n import (
    SURROGATES_N, build_task_roi, prep_context, compute_all_fast,
)

VOXEL = 0.4
DT_SAMPLE = 1.0
N_CAND = 6
MIN_PTS = 5
ROI_MARGIN = 10.0
Z_LO, Z_HI = -0.5, 6.0
POLE_H = 5.5
POSE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
ROAD_HALF_M = 5.0       # 任务 ROI 道路半宽 (m)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- 目录结构: 与 run_benchmark.py 完全一致 ---
def _env_id(env_path):
    if env_path:
        return os.path.basename(os.path.abspath(env_path))
    return "<flat_env>"


def _traj_id(traj_path):
    return os.path.basename(os.path.abspath(traj_path))


def _read_scene_info(scene_dir):
    path = os.path.join(scene_dir, "scene_info.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    for root, _, files in os.walk(scene_dir):
        if "config.json" in files:
            with open(os.path.join(root, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return {"scene": int(os.path.basename(scene_dir).split("_")[1]),
                    "env_name": _env_id(cfg.get("env")),
                    "env_path": cfg.get("env")}
    return None


def _read_traffic_info(traffic_dir):
    path = os.path.join(traffic_dir, "traffic_info.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    for root, _, files in os.walk(traffic_dir):
        if "config.json" in files:
            with open(os.path.join(root, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return {"traffic": int(os.path.basename(traffic_dir).split("_")[1]),
                    "traj_name": _traj_id(cfg.get("traj")),
                    "traj_path": cfg.get("traj")}
    return None


def window_frames(frame_times, win_s):
    t0, t1 = float(frame_times[0]), float(frame_times[-1])
    n_win = int((t1 - t0 + 1e-6) // win_s)
    windows = []
    for k in range(n_win):
        a = t0 + k * win_s
        want = a + np.arange(0, int(win_s), DT_SAMPLE)
        idx = np.searchsorted(frame_times, want)
        idx = np.clip(idx, 0, len(frame_times) - 1)
        got = frame_times[idx]
        ok = np.abs(got - want) <= 0.5 * DT_SAMPLE
        windows.append(got[ok])
    return [w for w in windows if len(w) > 0]


def traffic_stats(traj, t0, t1):
    mask = (traj.time >= t0) & (traj.time <= t1)
    vids = np.unique(traj.vid[mask])
    speeds = []
    for v in vids:
        tr = traj.tracks[int(v)]
        sel = (tr.t >= t0) & (tr.t <= t1)
        if sel.sum() < 2:
            continue
        tt = tr.t[sel]
        pos = 0.5 * (tr.front[sel] + tr.rear[sel])
        dt = np.diff(tt)
        dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        ok = dt > 0
        if ok.any():
            speeds.append(float((dp[ok] / dt[ok]).mean()))
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    volume = len(vids) / max((t1 - t0) / 3600.0, 1e-9)
    return mean_speed, volume


# --- POG 构建 ---
def make_grid(builder, voxel=VOXEL):
    traj = builder.traj
    x0, x1 = traj.front[:, 0].min(), traj.front[:, 0].max()
    y0, y1 = traj.front[:, 1].min(), traj.front[:, 1].max()
    x0, x1 = x0 - ROI_MARGIN, x1 + ROI_MARGIN
    y0, y1 = y0 - ROI_MARGIN, y1 + ROI_MARGIN
    gx = np.linspace(x0, x1, 40)
    gy = np.linspace(y0, y1, 20)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    gz = builder.ground.height(np.stack([GX.ravel(), GY.ravel()], 1))
    z0, z1 = float(gz.min()) + Z_LO, float(gz.max()) + Z_HI
    origin = np.array([x0, y0, z0], np.float64)
    shape = tuple(np.ceil((np.array([x1, y1, z1]) - origin) / voxel).astype(int))
    return origin, shape


def _geom_obs_mask(poses_user, specs, origin, shape, voxel):
    nx, ny, nz = shape
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ctr = origin + (np.stack([i.ravel(), j.ravel(), k.ravel()], 1) + 0.5) * voxel
    obs = np.zeros(len(ctr), bool)
    for pu, spec in zip(poses_user, specs):
        d = ctr - np.asarray(pu[:3])
        r2 = np.hypot(d[:, 0], d[:, 1])
        r3 = np.linalg.norm(d, axis=1)
        el = np.degrees(np.arctan2(d[:, 2], np.maximum(r2, 1e-6)))
        m = (r3 >= spec.range_min) & (r3 <= spec.range_max) \
            & (el >= spec.vfov_min_deg) & (el <= spec.vfov_max_deg)
        if spec.fov_h_deg < 360:
            az = np.degrees(np.arctan2(d[:, 1], d[:, 0]) - pu[5])
            az = (az + 180.0) % 360.0 - 180.0
            m &= np.abs(az) <= spec.fov_h_deg / 2
        obs |= m
    return obs.reshape(shape).astype(np.float64)


def build_pog(sim, times, origin, shape, voxel, poses_user, specs, frame_step=1):
    occ = np.zeros(shape, np.int32)
    n_used = 0
    for t in times[::frame_step]:
        scene, clouds = sim.simulate_frame(float(t))
        pts = [c.xyz for c in clouds if c.xyz is not None and len(c.xyz)]
        if not pts:
            continue
        pts = np.vstack(pts)
        ij = np.floor((pts[:, :3] - origin) / voxel).astype(np.int64)
        ok = np.all((ij >= 0) & (ij < np.asarray(shape)), axis=1)
        ij = ij[ok]
        flat = np.unique(ij[:, 0] * shape[1] * shape[2]
                         + ij[:, 1] * shape[2] + ij[:, 2])
        occ.reshape(-1)[flat] += 1
        n_used += 1
    p = occ / max(n_used, 1)
    o = _geom_obs_mask(poses_user, specs, origin, shape, voxel)
    log(f"  POG: {n_used} frames, occupied voxels {int((p > 0).sum())}, "
        f"observable {int(o.sum())}")
    return OccGrid(p, o, origin, voxel)


# --- 位姿生成 ---
def initial_poses(builder, n_lidar, rng, min_dist=15.0):
    traj = builder.traj
    x0, x1 = traj.front[:, 0].min(), traj.front[:, 0].max()
    y0, y1 = traj.front[:, 1].min(), traj.front[:, 1].max()
    cy = 0.5 * (y0 + y1)
    xs = np.arange(x0 + 4, x1 - 4, 8.0)
    poles = np.concatenate([
        np.stack([xs, np.full_like(xs, y0 - 3.0)], 1),
        np.stack([xs, np.full_like(xs, y1 + 3.0)], 1)], 0)
    env_pts = builder.env_sim.points[:, :3]
    clear = np.ones(len(poles), bool)
    for i, p in enumerate(poles):
        d = np.linalg.norm(env_pts[:, :2] - p, axis=1)
        near = env_pts[d < 1.5]
        if len(near):
            zg = float(builder.ground.height(p[None, :])[0])
            zr = zg + POLE_H
            if np.any(np.abs(near[:, 2] - zr) < 1.5):
                clear[i] = False
    poles = poles[clear]
    if len(poles) == 0:
        poles = np.concatenate([
            np.stack([xs, np.full_like(xs, y0 - 3.0)], 1),
            np.stack([xs, np.full_like(xs, y1 + 3.0)], 1)], 0)
    sel = [int(np.argmax(np.abs(poles[:, 1] - cy)))]
    while len(sel) < min(n_lidar, len(poles)):
        d = np.linalg.norm(poles[None, :, :] - poles[sel][:, None, :], axis=2)
        dmin = d.min(0)
        dmin[sel] = -1
        sel.append(int(np.argmax(dmin)))
    poles = poles[sel[:n_lidar]]
    out = []
    for p in poles:
        zg = float(builder.ground.height(p[None, :])[0])
        yaw = float(np.arctan2(cy - p[1], 0.5 * (x0 + x1) - p[0]))
        out.append([p[0], p[1], zg + POLE_H, 0.0, 0.0, yaw])
    return np.asarray(out, np.float64)


def sample_candidates(prev_pose, n_cand, pos_margin, ang_margin, rng):
    prev = np.asarray(prev_pose, np.float64)
    N = len(prev)
    out = np.repeat(prev[None], n_cand, 0)
    out[:, :, :3] += rng.uniform(-pos_margin, pos_margin, (n_cand, N, 3))
    out[:, :, 3] += rng.uniform(-ang_margin, ang_margin, (n_cand, N))
    out[:, :, 4] += rng.uniform(-ang_margin, ang_margin, (n_cand, N))
    return out


# --- 候选仿真与真值标签 ---
def eval_candidate(builder, poses_user, specs, times, seed=0):
    lidars = [LidarPose(spec=spec, pose=pose_internal(pu))
              for spec, pu in zip(specs, poses_user)]
    sim = LidarSimulator(builder, lidars, SimParams(device="auto", seed=seed))
    evals = []
    for t in times:
        scene, clouds = sim.simulate_frame(float(t))
        evals.append(evaluate_frame(clouds, scene, min_points=MIN_PTS))
    iou = float(np.mean([ev.mean_iou for ev in evals]))
    dev = float(np.mean([ev.mean_dev for ev in evals]))
    comp = sequence_completeness(evals)
    return iou, dev, comp


def score_of(iou, dev, comp):
    return 0.4 * iou + 0.3 * (1.0 - np.clip(dev, 0, 5) / 5.0) + 0.3 * comp


# --- 位姿参数存档 ---
def lidar_pose_records(poses_user, presets):
    arr = np.asarray(poses_user, np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    recs = []
    for a in range(arr.shape[0]):
        for i in range(arr.shape[1]):
            recs.append([a + 1, i + 1, str(presets[i])] +
                        [float(v) for v in arr[a, i]])
    return recs


def save_lidar_params(sdir, tag, scene, window_min, k, t_range, poses_user,
                      presets, chosen=None):
    arr = np.asarray(poses_user, np.float64)
    n_scheme = 1 if arr.ndim == 2 else int(arr.shape[0])
    obj = dict(scene=scene, window_min=window_min, window=k,
               t_range=[float(t_range[0]), float(t_range[1])],
               pose_order=POSE_KEYS, pose_unit="rad",
               n_schemes=n_scheme, n_lidar=len(presets),
               presets=list(presets),
               columns=["scheme", "lidar", "type", *POSE_KEYS],
               entries=lidar_pose_records(arr, presets))
    if chosen is not None:
        obj["chosen"] = int(chosen)
    path = os.path.join(sdir, f"lidar_para_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log(f"  lidar params saved: {os.path.basename(path)} "
        f"({len(obj['entries'])} records, {n_scheme} schemes)")


# --- 可视化 ---
def _viz_window(sdir, tag, prev_grid, cand, data, scores, best, surrogates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    pc = prev_grid.column_max()
    extent = [prev_grid.origin[1], prev_grid.origin[1] + pc.shape[1] * prev_grid.voxel,
              prev_grid.origin[0], prev_grid.origin[0] + pc.shape[0] * prev_grid.voxel]
    ax1.imshow(pc.T, origin="lower", extent=extent, cmap="hot", vmin=0, vmax=1,
               aspect="equal")
    for a in range(len(cand)):
        xy = cand[a][:, :2]
        col = "lime" if a == best else "cyan"
        mk = "*" if a == best else "o"
        ax1.scatter(xy[:, 1], xy[:, 0], marker=mk, s=80 if a == best else 40,
                    c=col, edgecolors="k", linewidths=0.5,
                    label=f"cand{a}" + (" (chosen)" if a == best else ""),
                    zorder=5)
    ax1.set_xlabel("y (m)")
    ax1.set_ylabel("x (m)")
    ax1.set_title(f"history POG + candidates ({tag})")
    ax1.legend(fontsize=7, loc="upper right")
    x = np.arange(len(cand))
    w = 0.2
    ax2.bar(x - 1.5 * w, data[:, 0], w, label="IoU")
    ax2.bar(x - 0.5 * w, 1 - np.clip(data[:, 1], 0, 5) / 5, w, label="1-Dev/5")
    ax2.bar(x + 0.5 * w, data[:, 2], w, label="Comp")
    ax2.bar(x + 1.5 * w, scores, w, label="score")
    ax2.set_xticks(x, [f"c{a}" for a in range(len(cand))])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("value")
    ax2.set_title(f"candidate labels & score (chosen: cand{best})")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(sdir, f"viz_{tag}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log(f"  viz saved: {out}")


# --- 主流程 ---
def resolve_run_dirs(cfg):
    env_id = _env_id(cfg.env)
    traj_id = _traj_id(cfg.traj)

    if cfg.scene is not None:
        scene = cfg.scene
    else:
        matched = []
        if os.path.isdir(cfg.out):
            for d in os.listdir(cfg.out):
                sd = os.path.join(cfg.out, d)
                if d.startswith("scene_") and os.path.isdir(sd):
                    info = _read_scene_info(sd)
                    if info and info.get("env_name") == env_id:
                        matched.append(int(d.split("_")[1]))
        if matched:
            scene = max(matched)
        else:
            all_sc = [int(d.split("_")[1]) for d in os.listdir(cfg.out)
                      if d.startswith("scene_")
                      and os.path.isdir(os.path.join(cfg.out, d))] \
                if os.path.isdir(cfg.out) else []
            scene = max(all_sc, default=0) + 1

    scene_dir = os.path.join(cfg.out, f"scene_{scene:03d}")
    os.makedirs(scene_dir, exist_ok=True)
    si_path = os.path.join(scene_dir, "scene_info.json")
    if not os.path.isfile(si_path):
        with open(si_path, "w", encoding="utf-8") as f:
            json.dump(dict(scene=scene, env_name=env_id,
                           env_path=(os.path.abspath(cfg.env) if cfg.env else None)),
                      f, ensure_ascii=False, indent=2)

    matched_t = []
    if os.path.isdir(scene_dir):
        for d in os.listdir(scene_dir):
            td = os.path.join(scene_dir, d)
            if d.startswith("traffic_") and os.path.isdir(td):
                info = _read_traffic_info(td)
                if info and info.get("traj_name") == traj_id:
                    matched_t.append(int(d.split("_")[1]))
    if matched_t:
        traffic = max(matched_t)
    else:
        all_t = [int(d.split("_")[1]) for d in os.listdir(scene_dir)
                 if d.startswith("traffic_")
                 and os.path.isdir(os.path.join(scene_dir, d))] \
            if os.path.isdir(scene_dir) else []
        traffic = max(all_t, default=0) + 1

    traffic_dir = os.path.join(scene_dir, f"traffic_{traffic:02d}")
    os.makedirs(traffic_dir, exist_ok=True)
    ti_path = os.path.join(traffic_dir, "traffic_info.json")
    if not os.path.isfile(ti_path):
        with open(ti_path, "w", encoding="utf-8") as f:
            json.dump(dict(traffic=traffic, traj_name=traj_id,
                           traj_path=os.path.abspath(cfg.traj)),
                      f, ensure_ascii=False, indent=2)

    sdir = os.path.join(traffic_dir, f"t_{cfg.window_min:02d}")
    os.makedirs(sdir, exist_ok=True)
    return scene, traffic, sdir


def run(cfg):
    rng = np.random.default_rng(cfg.seed)
    scene, traffic, sdir = resolve_run_dirs(cfg)
    SSS, TT, TK = scene, cfg.window_min, traffic

    pos_margin = cfg.pos_margin if cfg.pos_margin is not None \
        else float(rng.choice([0.2, 0.4, 0.6]))
    ang_margin = cfg.ang_margin if cfg.ang_margin is not None \
        else float(rng.choice([5.0, 10.0, 15.0]))
    ang_rad = float(np.radians(ang_margin))

    presets = (cfg.presets.split(",") * cfg.n_lidar)[:cfg.n_lidar] \
        if cfg.presets else ["vlp16"] * cfg.n_lidar
    specs = [make_lidar(p) for p in presets]
    conf = dict(scene=SSS, traffic=TK, window_min=TT, window_frames=int(TT * 60),
                dt_sample=DT_SAMPLE, voxel=VOXEL, n_lidar=cfg.n_lidar,
                presets=presets, pos_margin_m=pos_margin,
                ang_margin_deg=ang_margin, seed=cfg.seed,
                pose_order=POSE_KEYS, pose_unit="rad",
                n_candidates=N_CAND, min_points=MIN_PTS,
                score="0.4*IoU + 0.3*(1-Dev/5) + 0.3*Comp",
                surrogates=[n for n, _ in SURROGATES_N],
                metric_version="v2 (metric_n)",
                roi="trajectory_corridor",
                traj=os.path.abspath(cfg.traj), env=cfg.env, models=cfg.models)
    with open(os.path.join(sdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)
    log(f"scene {SSS} traffic {TK} t_{TT:02d}: pos_margin=±{pos_margin} m, "
        f"ang_margin=±{ang_margin} deg, lidars={presets}")

    log("loading data ...")
    traj = Trajectories.load(cfg.traj)
    env = load_env(cfg.env) if cfg.env else make_flat_env(traj)
    models = ModelLibrary(cfg.models)
    builder = SceneBuilder(traj, env, models, verbose=False)
    origin, shape = make_grid(builder, VOXEL)
    log(f"grid shape {shape}, voxel {VOXEL} m, "
        f"{int(np.prod(shape))} voxels")

    win_s = TT * 60.0
    windows = window_frames(traj.frame_times, win_s)
    if cfg.max_windows > 0:
        windows = windows[:cfg.max_windows]
    log(f"{len(windows)} windows x {TT} min")

    hist = []
    prev_pose = None
    for k, times in enumerate(windows, start=1):
        t0, t1 = float(times[0]), float(times[-1])
        tag = f"{SSS:03d}_t_{TT:02d}_{k:06d}"
        log(f"--- window {k}/{len(windows)} [{t0:.0f},{t1:.0f}] s ---")

        if k == 1:
            pose = initial_poses(builder, cfg.n_lidar, rng)
            log(f"  initial poses:\n{np.round(pose, 3)}")
        else:
            cand = sample_candidates(prev_pose, N_CAND, pos_margin, ang_rad, rng)
            # ---- v2 替代指标: 任务 ROI (轨迹走廊) + 批量高效计算 ----
            ref_grid = hist[-1]
            # ROI 由轨迹构造, 只依赖网格原点/体素, 与位姿无关
            roi_col = build_task_roi(ref_grid, traj.front[:, :2],
                                      road_half_m=ROAD_HALF_M)
            # 预计算与位姿无关的量, 跨候选复用
            ctx = prep_context(ref_grid, hist[-cfg.h:], roi_col)
            log(f"  task ROI: {int(roi_col.sum())} cols "
                f"({roi_col.sum() / roi_col.size:.1%} of BEV)")
            J = np.zeros((N_CAND, len(SURROGATES_N)))
            t0_m = time.time()
            for a in range(N_CAND):
                J[a] = compute_all_fast(list(cand[a]), specs, ref_grid,
                                        hist[-cfg.h:], roi_col, ctx=ctx,
                                        n_ray=cfg.n_ray)
                log(f"  cand {a}: " + " ".join(
                    f"{n}={v:.4f}" for (n, _), v in zip(SURROGATES_N, J[a])))
            log(f"  surrogate metrics for {N_CAND} candidates: "
                f"{time.time() - t0_m:.2f} s")
            # 高保真仿真真值标签
            data = np.zeros((N_CAND, 8))
            for a in range(N_CAND):
                iou, dev, comp = eval_candidate(builder, list(cand[a]),
                                                specs, times[::cfg.frame_step],
                                                seed=cfg.seed)
                data[a] = [iou, dev, comp, *J[a]]
                log(f"  cand {a}: IoU={iou:.3f} Dev={dev:.3f} "
                    f"Comp={comp:.3f}")
            scores = np.array([score_of(*r[:3]) for r in data])
            best = int(np.argmax(scores))
            pose = cand[best]
            data_json = [[None if np.isnan(v) else float(v) for v in row]
                         for row in data]
            with open(os.path.join(sdir, f"data_{tag}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(dict(scene=SSS, window_min=TT, window=k,
                               t_range=[t0, t1],
                               columns=["IoU", "Dev", "Comp"] +
                                       [n for n, _ in SURROGATES_N],
                               data=data_json,
                               poses_user=cand.tolist(),
                               score=scores.tolist(), chosen=best),
                          f, ensure_ascii=False, indent=2)
            log(f"  chosen cand {best} (score={scores[best]:.3f})")
            if cfg.viz:
                _viz_window(sdir, tag, hist[-1], cand, data, scores, best,
                            SURROGATES_N)

        all_poses = pose[None] if k == 1 else cand
        save_lidar_params(sdir, tag, SSS, TT, k, (t0, t1), all_poses,
                          presets, chosen=(0 if k == 1 else best))

        lidars = [LidarPose(spec=s, pose=pose_internal(pu))
                  for s, pu in zip(specs, pose)]
        sim = LidarSimulator(builder, lidars,
                             SimParams(device="auto", seed=cfg.seed))
        grid = build_pog(sim, times, origin, shape, VOXEL, list(pose), specs,
                         frame_step=cfg.frame_step)
        grid.save(os.path.join(sdir, f"pog_{tag}.npz"),
                  pose_user=pose, t_range=[t0, t1])
        hist.append(grid)
        hist = hist[-max(cfg.h, 1):]

        v, q = traffic_stats(traj, t0, t1)
        with open(os.path.join(sdir, f"traffic_{tag}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(f"window {k} t=[{t0:.1f},{t1:.1f}] s\n"
                    f"mean_speed_mps={v:.3f}\n"
                    f"volume_veh_per_h={q:.1f}\n")
        log(f"  traffic: speed={v:.2f} m/s, volume={q:.0f} veh/h")

        prev_pose = pose
    log("done.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="多 LiDAR 位姿替代指标测试基准构建 (v2 指标 + 任务 ROI)")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-min", type=int, choices=[5, 15, 30], required=True)
    ap.add_argument("--n-lidar", type=int, default=4)
    ap.add_argument("--presets", default=None)
    ap.add_argument("--pos-margin", type=float, default=None)
    ap.add_argument("--ang-margin", type=float, default=None)
    ap.add_argument("--scene", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--n-ray", type=int, default=48)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--viz", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

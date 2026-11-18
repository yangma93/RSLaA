# -*- coding: utf-8 -*-
"""基于 MPC 的协作式多 LiDAR 位姿参数自适应控制.

滚动时域控制流程:
  1. 从 lidar_pose.json 读取初始位姿 + 类型 + LiDAR 数量
  2. 指定时间窗长度 (5/15/30 min)、位置自由度 (默认 0.5m)、姿态自由度 (默认 10°)
  3. 逐个时间窗执行:
     k=1:  使用初始位姿仿真, 构建 POG
     k>=2: 以上一窗位姿为候选种子, 在**锚定初始位姿**的固定盒内用
           scipy.optimize 优化选定的替代指标
           (J_EIG/J_cov/J_mv/J_dyn/J_traj 之一) 作为目标;
           约束: 每个 pose 维度在 [init - margin, init + margin] 内
           优化得到的位姿 -> 仿真 -> 构建新 POG -> 记录到序列
  4. 全部窗口完成后, 写出 lidar_pose_seq.json (含窗口长度、索引、位姿序列)

可行域说明: 裕度始终相对**初始位姿** (lidar_pose.json 的 poses), 一旦初始位姿
确定, RSL 的可移动范围与可调姿态角就固定下来; 上一窗位姿只作为候选种子,
不会让可行域随窗口滚动而平移, 因此不会出现累积漂移.

设计要点:
  * 不修改 lidarsim / rslaa_bench 源码, 仅在上层组合调用
  * 优化目标函数重用 common.compute_surrogate (v2 指标 metric_n), 历史占用图来自已构建的 OccGrid
  * scipy.optimize: 默认候选采样 + Nelder-Mead 精修 (--optimizer lbfgsb 可切回精确慢速版)
  * 位姿自由度默认 ±0.5m / ±10°, 为相对初始位姿的固定盒半宽, 可通过 CLI 覆盖
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rslaa.common import (
    VOXEL, DT_SAMPLE, POSE_KEYS, POSE_UNIT,
    METRIC_NAMES, METRIC_FUNCS, DEFAULT_METRIC, LOWER_IS_BETTER,
    metric_sign, metric_key,
    load_lidar_pose_json, save_lidar_pose_seq_json, append_pose_window,
    window_frames, build_builder_and_sim, make_sim, make_specs,
    make_grid_origin_shape, build_pog_occgrid, task_roi,
    compute_surrogate, compute_all_surrogates, compute_surrogate_oriented,
)


# ---------------------------------------------------------------------------
# 快速替代指标目标 (柱级, 活动柱子集)
#
# 背景: metrics.py 中 J_EIG/J_cov/J_dyn 为体素级指标, soft_visibility 每次
# 调用需对全网格 (可达数百万体素) x N 雷达做向量运算; L-BFGS-B 数值梯度下
# 单窗优化需数千次评估 -> 小时级, 不可用.
#
# 加速策略 (仅用于优化指导; 每窗报告仍用 metric_n 的精确指标):
#   * 任务区域取"任务 ROI 柱": 轨迹走廊 (traj + 横向 5m 膨胀), 过多时均匀
#     抽样到 max_cols 以控制单次评估耗时; 与 metric_n 的归一化分母一致,
#     避免旧版"历史可观测区域 = 近全盘"导致的指标方向反转.
#   * 柱级可见度 q_col 直接复用 metrics._column_visibility (n_ray 可降采样);
#     J_EIG/J_cov/J_dyn 用柱级近似 (H/d 取柱内 max), J_mv/J_traj 本就是柱级.
# ---------------------------------------------------------------------------
from rslaa_bench.metrics import (
    _column_visibility, bern_entropy, _dyn_weight, extract_corridors,
)


class FastObjective:
    """单窗优化的快速目标函数.

    方向约定: J_EIG/J_mv 越大越好; J_cov/J_dyn/J_traj 越小越好.
    __call__ 统一返回"越大越好得分"的负值, 供 minimize 使用.
    """

    def __init__(self, ref_poses, specs, hist, metric: str,
                 pos_margin: float, ang_rad: float, n_ray: int = 24,
                 max_cols: int = 12000, seed: int = 0,
                 anchor_poses=None):
        """ref_poses: 候选种子 (上一窗位姿); anchor_poses: 盒约束锚点
        (初始位姿, 缺省 = ref_poses). 裕度始终相对锚点位姿, 与窗口无关."""
        self.specs = specs
        self.metric = metric
        self.n_ray = int(n_ray)
        self.n_lidar = len(ref_poses)
        self.anchor = np.asarray(
            ref_poses if anchor_poses is None else anchor_poses, np.float64)
        marg = np.tile([pos_margin, pos_margin, pos_margin,
                        ang_rad, ang_rad, ang_rad], self.n_lidar)
        self.lo = self.anchor.ravel() - marg
        self.hi = self.anchor.ravel() + marg
        # 参考位姿只作为候选种子; 若因数值误差落到盒外, 投影回可行域
        self.ref = np.clip(np.asarray(ref_poses, np.float64).ravel(),
                           self.lo, self.hi).reshape(self.n_lidar, 6)
        self.n_eval = 0

        grid = hist[-1]
        self.grid = grid
        self.p_col = grid.column_max()                      # (nx,ny)
        nx, ny = self.p_col.shape
        self.ny = ny
        ci, cj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        col_ij = np.stack([ci.ravel(), cj.ravel()], 1)
        col_xy_all = grid.origin[:2] + (col_ij + 0.5) * grid.voxel

        # 任务 ROI (道路走廊): v2 指标的评测区域, 也是优化的归一化分母
        roi_flat = np.asarray(task_roi(grid)).ravel()
        self.roi_col = roi_flat.reshape(nx, ny)
        act_idx = np.flatnonzero(roi_flat)
        # 列数过多时均匀抽样 (对 ROI 均值是无偏估计, 控制单次评估耗时)
        if len(act_idx) > max_cols:
            rng = np.random.default_rng(seed)
            act_idx = np.sort(rng.choice(act_idx, max_cols, replace=False))
        self.act_ij = col_ij[act_idx]
        self.act_xy = col_xy_all[act_idx]
        self.w_col = np.ones(len(self.act_xy), np.float64)

        # 各指标的柱级权重 (在 ROI 子集上取)
        H_col = bern_entropy(grid.p).max(axis=2).ravel()[act_idx]
        if metric == "J_EIG":
            self.w_task = H_col + 1e-6
        elif metric == "J_dyn":
            d = _dyn_weight(hist).max(axis=2).ravel()[act_idx]
            self.w_task = d if d.max() >= 1e-3 else H_col + 1e-6
        else:
            self.w_task = self.w_col

        # J_traj: 走廊柱与任务 ROI 求交
        self.corridor_cells = None
        if metric == "J_traj":
            cors = extract_corridors(grid)
            if cors:
                cells = np.unique(np.concatenate(
                    [c["s1"][1] for c in cors], axis=0), axis=0)
                keep = self.roi_col[cells[:, 0], cells[:, 1]]
                cells = cells[keep]
                if len(cells):
                    flat = cells[:, 0] * ny + cells[:, 1]
                    order = np.argsort(flat)
                    self.corridor_cells = (flat[order], cells[order])
            if self.corridor_cells is None:                 # 无走廊 -> 退化为覆盖
                self.metric = "J_cov"
        # 优化方向: +1 = 最大化 (J_cov/J_EIG/J_mv); -1 = 最小化 (J_dyn/J_traj)
        self.sign = metric_sign(self.metric)

    def _q_col(self, poses):
        """各 LiDAR 对活跃柱的软可见度 (C,N)."""
        C = len(self.act_xy)
        q = np.zeros((C, self.n_lidar), np.float64)
        for n, (pu, spec) in enumerate(zip(poses, self.specs)):
            q[:, n] = _column_visibility(
                pu, spec, self.act_xy, self.p_col,
                self.grid.origin, self.grid.voxel, self.n_ray, 0.9)
        return q

    def score(self, flat_poses) -> float:
        """返回指标原始近似值 (方向见 self.sign); 不可用时返回 -inf."""
        self.n_eval += 1
        poses = np.clip(np.asarray(flat_poses, np.float64),
                        self.lo, self.hi).reshape(self.n_lidar, 6)
        m = self.metric
        if m == "J_traj" and self.corridor_cells is not None:
            flat_cells, cells = self.corridor_cells
            xy = self.grid.origin[:2] + (cells + 0.5) * self.grid.voxel
            q = np.zeros((len(cells), self.n_lidar), np.float64)
            for n, (pu, spec) in enumerate(zip(poses, self.specs)):
                q[:, n] = _column_visibility(
                    pu, spec, xy, self.p_col,
                    self.grid.origin, self.grid.voxel, self.n_ray, 0.9)
            return float((1.0 - np.prod(1.0 - q, axis=1)).mean())

        q = self._q_col(poses)
        q_u = 1.0 - np.prod(1.0 - np.clip(q, 0, 1), axis=1)
        w = self.w_task
        base = float((w * q_u).sum() / (w.sum() + 1e-12))
        if m == "J_mv":
            lam = 0.5
            mv = 0.0
            for a in range(self.n_lidar):
                for b in range(a + 1, self.n_lidar):
                    va = self.act_xy - poses[a][:2]
                    vb = self.act_xy - poses[b][:2]
                    na = np.hypot(va[:, 0], va[:, 1])
                    nb = np.hypot(vb[:, 0], vb[:, 1])
                    sin = np.abs(va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]) \
                        / np.maximum(na * nb, 1e-6)
                    mv += float((w * q[:, a] * q[:, b] * sin).sum())
            base += lam * mv / (w.sum() + 1e-12)
        return base

    def __call__(self, flat_poses) -> float:
        s = self.score(flat_poses)
        s = self.sign * s
        return -s if np.isfinite(s) else 1e6


# 注意: scipy 优化的边界需要调用者保证, 这里用模块级变量传递 (避免闭包被 pickle 时出问题)
_pos_margin = 0.5
_ang_rad = np.deg2rad(10.0)


def optimize_poses_fast(ref_poses: np.ndarray, specs: list, hist: list,
                        metric: str = DEFAULT_METRIC,
                        pos_margin: float = 0.5, ang_margin_deg: float = 10.0,
                        n_ray: int = 24, n_cand: int = 256,
                        refine: bool = True, refine_maxfun: int = 120,
                        seed: int = 0, verbose: bool = True,
                        anchor_poses: np.ndarray = None) -> np.ndarray:
    """候选采样 + Nelder-Mead 短精修的快速优化.

    总评估次数 ≈ n_cand + refine_maxfun (默认 ~380 次, 毫秒级单次),
    对比 L-BFGS-B 多起点 (最多 ~4000 次, 秒级单次) 提速 3~4 个数量级.

    anchor_poses: 盒约束锚点. 主循环传入**初始位姿**, 使全部窗口共享同一个
        可行域 [init - margin, init + margin]; 缺省 = ref_poses (旧行为).
    """
    t0 = time.time()
    obj = FastObjective(ref_poses, specs, hist, metric,
                        pos_margin, np.deg2rad(ang_margin_deg), n_ray=n_ray,
                        anchor_poses=anchor_poses)
    N = obj.n_lidar
    dim = 6 * N
    rng = np.random.default_rng(seed)

    # ---- 1) 候选采样: 参考位姿 + 锚点位姿 + 盒角点 + 均匀随机 (盒内) ----
    cands = [obj.ref.ravel()]
    if anchor_poses is not None:
        cands.append(obj.anchor.ravel())                    # "不调整"基准
    cands.append(obj.lo); cands.append(obj.hi)              # 角点参考
    cands += [rng.uniform(obj.lo, obj.hi) for _ in range(n_cand)]
    vals = obj.sign * np.array([obj.score(c) for c in cands])
    vals[~np.isfinite(vals)] = -np.inf
    best_i = int(np.argmax(vals))
    best_x, best_v = cands[best_i].copy(), float(vals[best_i])
    if verbose:
        print(f"  [fast] 候选 {len(cands)} 个: best={best_v:.4f} "
              f"(参考位姿={vals[0]:.4f}), 活跃柱={len(obj.act_xy)}, "
              f"方向={'min' if obj.sign < 0 else 'max'}, "
              f"耗时 {time.time() - t0:.1f}s")

    # ---- 2) Nelder-Mead 短精修 (盒内 clip) ----
    if refine and np.isfinite(best_v):
        from scipy.optimize import minimize
        t1 = time.time()
        res = minimize(obj, best_x, method="Nelder-Mead",
                       options={"maxfev": refine_maxfun, "maxiter": refine_maxfun,
                                "xatol": 1e-3, "fatol": 1e-4})
        x = np.clip(res.x, obj.lo, obj.hi)
        v = obj.sign * obj.score(x)
        if np.isfinite(v) and v > best_v:
            best_x, best_v = x, v
        if verbose:
            print(f"  [fast] 精修后 best={best_v:.4f}, "
                  f"精修耗时 {time.time() - t1:.1f}s")

    if verbose:
        print(f"  [fast] 优化总耗时 {time.time() - t0:.1f}s, "
              f"目标评估 {obj.n_eval} 次")
    return best_x.reshape(N, 6)


def optimize_poses_scipy(ref_poses: np.ndarray, specs: list,
                          grid, hist, metric: str = DEFAULT_METRIC,
                          pos_margin: float = 0.5, ang_margin_deg: float = 10.0,
                          n_ray: int = 48, n_starts: int = 8,
                          bounds_extra: dict = None,
                          anchor_poses: np.ndarray = None) -> np.ndarray:
    """原始精确指标 + L-BFGS-B 多起点 (保留作对照, 速度很慢).

    目标统一转换为"越大越好": J_cov/J_dyn/J_traj 取负号.
    anchor_poses: 盒约束锚点 (初始位姿); 缺省 = ref_poses (旧行为).
    """
    from scipy.optimize import minimize

    global _pos_margin, _ang_rad
    _pos_margin = float(pos_margin)
    _ang_rad = float(np.deg2rad(ang_margin_deg))

    N = len(ref_poses)
    ref = np.asarray(ref_poses, np.float64)
    anchor = np.asarray(
        ref_poses if anchor_poses is None else anchor_poses, np.float64)
    flat_anchor = anchor.ravel()
    marg = np.tile([pos_margin] * 3 + [_ang_rad] * 3, N)
    lo, hi = flat_anchor - marg, flat_anchor + marg
    flat_ref = np.clip(ref.ravel(), lo, hi)
    bounds = list(zip(lo, hi))

    sign = metric_sign(metric)

    def obj(x):
        poses = np.clip(x.reshape(N, 6), lo.reshape(N, 6), hi.reshape(N, 6))
        score = compute_surrogate(poses, specs, grid, hist, metric=metric, n_ray=n_ray)
        return -sign * score if np.isfinite(score) else 1e6

    rng = np.random.default_rng(0)
    x0_list = [flat_ref]
    if anchor_poses is not None:
        x0_list.append(flat_anchor)
    x0_list += [rng.uniform(lo, hi)
                for _ in range(max(n_starts - len(x0_list), 0))]
    best_x, best_f = flat_ref, 1e6
    for x0 in x0_list:
        try:
            res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "maxfun": 500})
            if res.fun < best_f:
                best_f, best_x = res.fun, res.x
        except Exception as e:
            print(f"  [optimize] 起点失败: {e}")
    return np.clip(best_x, lo, hi).reshape(N, 6)


# ---------------------------------------------------------------------------
# 滚动时域主循环
# ---------------------------------------------------------------------------
def run_mpc(traj_path: str, env_path: str, models_dir: str,
            pose_json_path: str, out_seq_path: str,
            window_minutes: int = 5,
            pos_margin: float = 0.5, ang_margin_deg: float = 10.0,
            metric: str = DEFAULT_METRIC,
            voxel: float = VOXEL,
            n_ray: int = 48, max_history: int = 5,
            device: str = "auto", frame_step: int = 1,
            max_windows: int = 0, verbose: bool = True,
            optimizer: str = "fast", n_cand: int = 256,
            refine: bool = True, workers: int = 1):
    """MPC 滚动时域控制主函数.

    参数:
      traj_path / env_path / models_dir: 仿真场景数据
      pose_json_path: lidar_pose.json 初始位姿文件
      out_seq_path: 输出 lidar_pose_seq.json
      window_minutes: 时间窗长度 (分钟)
      pos_margin / ang_margin_deg: 位置/姿态自由度, 即相对**初始位姿**的
              固定盒半宽; 全部窗口都约束在 [init - margin, init + margin] 内
      metric: 替代指标 (J_EIG/J_cov/J_mv/J_dyn/J_traj), 默认 J_mv
              J_mv 与感知指标正相关 (越大越好), 目标按最大化处理;
              J_EIG 同样最大化; J_cov/J_dyn/J_traj 则越小越好
      voxel / n_ray: POG 体素/可见性射线采样数
      max_history: 历史 POG 保留窗口数
      device: auto/gpu/cpu
      frame_step: POG 帧采样步长
      max_windows: 仅前 N 窗 (0=全部, 调试用)
      workers: 帧级并行进程数 (>1 启用多进程 GPU 并行, 结果与串行逐位一致)
    """
    t_start = time.time()
    win_s = window_minutes * 60.0

    # ---- 1) 加载初始位姿 ----
    pose_obj = load_lidar_pose_json(pose_json_path)
    presets = pose_obj["presets"]
    initial_poses = pose_obj["poses"]
    n_lidar = pose_obj["n_lidar"]
    if verbose:
        print(f"[MPC] 读取初始位姿: {n_lidar} 台 LiDAR, 类型 {presets}")
        print(f"[MPC] 初始位姿 (用户序 [x y z roll pitch yaw]):")
        print(f"      {np.round(initial_poses, 3)}")

    # ---- 2) 加载仿真场景 ----
    # 并行池先启动 (异步构建), 与父进程的场景加载重叠, 减少启动开销
    pool = None
    if workers > 1:
        from rslaa.sim_pool import SimPool
        pool = SimPool(traj_path, env_path, models_dir, device=device,
                       workers=workers, verbose=verbose, wait=False)
    builder, params = build_builder_and_sim(traj_path, env_path, models_dir,
                                             device=device, verbose=verbose)
    specs = make_specs(presets)
    if pool is not None:
        pool.wait_ready()
    origin, shape = make_grid_origin_shape(builder, voxel)
    frame_times = builder.traj.frame_times
    windows = window_frames(frame_times, win_s)
    if max_windows > 0:
        windows = windows[:max_windows]
    if verbose:
        print(f"[MPC] 时间窗划分: {len(windows)} x {window_minutes} min")
        _mk = metric_key(metric)
        _dir = "越小越好 (min)" if LOWER_IS_BETTER.get(_mk, False) else "越大越好 (max)"
        print(f"[MPC] 优化目标: {_mk or metric} [{_dir}], "
              f"pos_margin=±{pos_margin} m, ang_margin=±{ang_margin_deg} deg")
        print(f"[MPC] 可行域: 全窗锚定初始位姿 (不随窗口平移), "
              f"位置 ±{pos_margin} m / 姿态 ±{ang_margin_deg}°")

    # ---- 3) 滚动时域 ----
    # 可行域锚点: 初始位姿固定不变, 所有窗口的位姿都必须落在
    # [anchor - margin, anchor + margin] 内 (避免逐窗累积漂移)
    anchor_poses = initial_poses.copy()
    pose_seq_windows = []   # 输出序列
    hist_grids = []         # 历史 POG 序列 (OccGrid)
    hist_traj = []          # 历史窗口时间范围

    for k, times in enumerate(windows, start=1):
        t0, t1 = float(times[0]), float(times[-1])
        if verbose:
            print(f"\n[MPC] === 窗口 {k}/{len(windows)} "
                  f"[{t0:.1f}, {t1:.1f}] s ===")

        if k == 1:
            # 首个窗: 直接用初始位姿
            cur_poses = anchor_poses.copy()
        else:
            # 滚动优化: 上一窗位姿仅作候选种子, 可行域始终锚定初始位姿
            ref_pose = pose_seq_windows[-1]["poses"]
            if verbose:
                print(f"  候选种子 (上一窗位姿):\n{np.round(ref_pose, 3)}")
            t_opt = time.time()
            if optimizer == "fast":
                cur_poses = optimize_poses_fast(
                    ref_pose, specs, hist_grids[-max_history:],
                    metric=metric, pos_margin=pos_margin,
                    ang_margin_deg=ang_margin_deg,
                    n_ray=min(n_ray, 24), n_cand=n_cand,
                    refine=refine, verbose=verbose,
                    anchor_poses=anchor_poses)
            else:
                cur_poses = optimize_poses_scipy(
                    ref_pose, specs, hist_grids[-1], hist_grids[-max_history:],
                    metric=metric, pos_margin=pos_margin,
                    ang_margin_deg=ang_margin_deg, n_ray=n_ray,
                    anchor_poses=anchor_poses)
            if verbose:
                print(f"  优化后位姿:\n{np.round(cur_poses, 3)}")
                print(f"  耗时 {time.time() - t_opt:.1f}s")

        # ---- 用本窗位姿仿真 + 构建 POG ----
        if verbose:
            print(f"  仿真 + POG 构建中...")
        sim = make_sim(builder, specs, cur_poses, params, seed=0)
        pog = build_pog_occgrid(sim, times, origin, shape, voxel,
                                cur_poses, specs, frame_step=frame_step,
                                pool=pool)
        hist_grids.append(pog)
        hist_grids = hist_grids[-max_history:]

        # ---- 记录窗口 ----
        append_pose_window(pose_seq_windows, k, [t0, t1], cur_poses)
        if verbose:
            j_all = compute_all_surrogates(cur_poses, specs, pog, hist_grids, n_ray=n_ray)
            names_str = " ".join(f"{n}={v:.3f}" if np.isfinite(v) else f"{n}=NaN"
                                 for n, v in zip(METRIC_NAMES, j_all))
            print(f"  本窗指标: {names_str}")
            mk = metric_key(metric)
            if mk is not None and np.isfinite(j_all[METRIC_NAMES.index(mk)]):
                d = "越小越好" if LOWER_IS_BETTER[mk] else "越大越好"
                print(f"  优化目标 {mk} = {j_all[METRIC_NAMES.index(mk)]:.4f} ({d})")

    # ---- 4) 写出 lidar_pose_seq.json ----
    os.makedirs(os.path.dirname(os.path.abspath(out_seq_path)) or ".", exist_ok=True)
    save_lidar_pose_seq_json(out_seq_path, presets, pose_seq_windows,
                             window_length_sec=win_s)
    if pool is not None:
        pool.close()
    if verbose:
        print(f"\n[MPC] 完成! 共 {len(pose_seq_windows)} 个窗口, "
              f"总耗时 {time.time() - t_start:.1f}s")
        print(f"[MPC] 位姿序列已保存: {out_seq_path}")

    return pose_seq_windows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-json", required=True,
                    help="初始位姿 lidar_pose.json")
    ap.add_argument("--traj", required=True, help="轨迹文件")
    ap.add_argument("--env", default=None, help="环境点云 (.ply/.h5, 可缺省)")
    ap.add_argument("--models", required=True, help="车辆模型目录")
    ap.add_argument("--out", required=True,
                    help="输出 lidar_pose_seq.json")
    ap.add_argument("--window-min", type=int, default=5,
                    choices=[5, 15, 20, 30],
                    help="时间窗长度 (分钟)")
    ap.add_argument("--pos-margin", type=float, default=0.5,
                    help="相对初始位姿的位置盒半宽 ±m (默认 0.5, 全窗固定)")
    ap.add_argument("--ang-margin-deg", type=float, default=10.0,
                    help="相对初始位姿的姿态盒半宽 ±deg (默认 10, 全窗固定)")
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    choices=list(METRIC_FUNCS.keys()),
                    help="优化目标指标")
    ap.add_argument("--voxel", type=float, default=VOXEL)
    ap.add_argument("--n-ray", type=int, default=48)
    ap.add_argument("--max-history", type=int, default=5)
    ap.add_argument("--device", default="gpu", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1,
                    help="帧级并行仿真进程数 (>1 启用多进程 GPU 并行, "
                         "结果与串行逐位一致; 显存约 0.5~1 GB/进程)")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--optimizer", default="fast", choices=["fast", "lbfgsb"],
                    help="fast=柱级近似指标+候选采样 (默认, 秒级); "
                         "lbfgsb=精确体素指标+L-BFGS-B (极慢, 仅对照)")
    ap.add_argument("--n-cand", type=int, default=64,
                    help="fast 优化器的随机候选数 (默认 64)")
    ap.add_argument("--no-refine", action="store_true",
                    help="fast 优化器禁用 Nelder-Mead 精修")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_mpc(
        traj_path=args.traj,
        env_path=args.env,
        models_dir=args.models,
        pose_json_path=args.pose_json,
        out_seq_path=args.out,
        window_minutes=args.window_min,
        pos_margin=args.pos_margin,
        ang_margin_deg=args.ang_margin_deg,
        metric=args.metric,
        voxel=args.voxel,
        n_ray=args.n_ray,
        max_history=args.max_history,
        device=args.device,
        frame_step=args.frame_step,
        max_windows=args.max_windows,
        verbose=not args.quiet,
        optimizer=args.optimizer,
        n_cand=args.n_cand,
        refine=not args.no_refine,
        workers=args.workers,
    )

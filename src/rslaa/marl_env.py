# -*- coding: utf-8 -*-
"""多 LiDAR 协同感知 — 多智能体环境 (Gym 接口).

每个 LiDAR 是一个独立智能体, 全部智能体同时执行动作 (位姿调整), 系统
进行点云仿真 -> POG 更新 -> 奖励 (替代指标) 计算.

观测 (o_i, 每个 agent, 全部相对 LiDAR#0 表征):
  [相对位姿 6 维 (在 LiDAR#0 局部坐标系: x,y,z,droll,dpitch,dyaw),
   局部 POG BEV 块 (k×k, 以该 LiDAR 为中心, 朝向与 LiDAR#0 对齐),
   全局特征编码 (以 LiDAR#0 为原点的低分辨率 BEV + 统计量)]

动作 (a_i, 每个 agent):
  6 维连续: [dx, dy, dz, droll, dpitch, dyaw] (归一化到 [-1,1]),
  含义是**相对本 episode 初始位姿的偏移**, 而不是相对上一窗的增量:
  位置偏移在初始 LiDAR#0 朝向的局部坐标系中解释 (与相对观测保持旋转一致),
  叠加到初始位姿后统一投影回 [初始位姿 ± 裕度] 的固定盒。
  因此裕度始终相对初始位姿, RSL 的可移动范围/可调姿态角在 reset 时即确定,
  任意窗口的位姿都不会随滚动累积漂移。

奖励 (全局共享):
  选定的替代指标值 (J_EIG/J_cov/J_mv/J_dyn/J_traj), 按历史 POG 计算,
  并按 common.metric_sign 统一成"越大越好"的奖励方向

终止条件: 时间窗用完; 最大步数
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rslaa.common import (
    VOXEL, DT_SAMPLE,
    METRIC_FUNCS, DEFAULT_METRIC, LOWER_IS_BETTER,
    metric_sign, metric_key,
    build_builder_and_sim, make_sim, make_specs,
    make_grid_origin_shape, build_pog_occgrid,
    compute_surrogate, compute_all_surrogates, compute_surrogate_oriented,
)


# 全局特征编码: 以 LiDAR#0 为原点的低分辨率 BEV (cells²) + 统计量 (extra)
GLOBAL_BEV_CELLS = 8
GLOBAL_BEV_CELL_M = 8.0
GLOBAL_ENC_EXTRA = 8
DEFAULT_GLOBAL_ENC_DIM = GLOBAL_BEV_CELLS ** 2 + GLOBAL_ENC_EXTRA

# 动作语义标记 (随权重写入 env_meta, 供部署/续训时做兼容性检查)
#   offset_from_init_pose: 动作为相对本轮初始位姿的偏移 (裕度盒锚定初始位姿)
#   旧版权重没有该字段, 其动作是相对上一窗的增量, 存在累积漂移
ACTION_SEMANTICS = "offset_from_init_pose"


def _wrap_angle(a):
    """角度差 wrap 到 [-pi, pi)."""
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


class MultiLidarEnv:
    """多 LiDAR 协同感知环境 (MAPPO 训练用).

    env.reset() -> obs_dict (agent_id -> obs_vec)
    env.step(actions_dict) -> obs_dict, global_reward, done, info
    """

    def __init__(self, traj_path, env_path=None, models_dir=None,
                 presets=None, initial_poses=None, n_lidar=2,
                 pos_margin=0.5, ang_margin_deg=10.0,
                 window_minutes=5, max_steps=None,
                 metric=DEFAULT_METRIC, voxel=VOXEL, device="auto",
                 local_bev_size=7, global_enc_dim=None,
                 seed=0, frame_step=1,
                 randomize_init_poses=True,
                 init_z_range=(5.0, 8.0),
                 init_road_jitter=4.0,
                 init_clear_radius=1.2,
                 init_min_sep=3.0, workers=1):
        """
        参数:
          traj_path/env_path/models_dir: 仿真场景
          presets: LiDAR 型号列表 (长度 = n_lidar)
          initial_poses: (N,6) 用户序初始位姿 (首轮使用; 之后每轮重采样)
          n_lidar: LiDAR 数量
          pos_margin / ang_margin_deg: 相对**初始位姿**的位置/姿态裕度,
                  即固定盒 [init - margin, init + margin] 的半宽
          window_minutes: 单次仿真时间窗 (分钟)
          max_steps: 每 episode 最大步数 (缺省 = 窗口数 - 1)
          metric: 奖励用替代指标 (默认 J_mv); J_mv/J_EIG 与感知指标正相关,
                  奖励直接取原值 (越大越好); J_cov/J_dyn/J_traj 取负值
          local_bev_size: 局部 POG BEV 块尺寸 (默认 7 => 7x7)
          global_enc_dim: 相对全局编码维度 (缺省 = 8x8 BEV + 8 统计量 = 72)
          frame_step: POG 帧采样步长
          randomize_init_poses: True 时每轮 reset (第 2 轮起) 重新随机采样
                  初始位姿 (道路区域上空, 附近无障碍, 离地 init_z_range)
          init_z_range: 初始位姿离地高度范围 (m), 默认 5~8 m
          init_road_jitter: 在轨迹点横向随机偏移的最大距离 (m)
          init_clear_radius: 障碍物净空检查半径 (m)
          init_min_sep: 同一轮内不同 LiDAR 的最小水平间距 (m)
          workers: 帧级并行仿真进程数 (>1 启用多进程 GPU 并行, 惰性创建)
        """
        self.traj_path = traj_path
        self.env_path = env_path
        self.models_dir = models_dir
        self.presets = presets or ["vlp16"] * n_lidar
        self.n_lidar = n_lidar
        self.pos_margin = float(pos_margin)
        self.ang_rad = float(np.deg2rad(ang_margin_deg))
        self.window_minutes = float(window_minutes)
        self.metric = metric
        self.voxel = voxel
        self.device = device
        self.local_bev_size = int(local_bev_size)
        self.global_enc_dim = int(global_enc_dim) if global_enc_dim else \
            DEFAULT_GLOBAL_ENC_DIM
        self.seed = seed
        self.frame_step = frame_step
        self.workers = max(1, int(workers))
        self._sim_pool = None

        # 随机初始位姿参数
        self.randomize_init_poses = bool(randomize_init_poses)
        self.init_z_range = (float(init_z_range[0]), float(init_z_range[1]))
        self.init_road_jitter = float(init_road_jitter)
        self.init_clear_radius = float(init_clear_radius)
        self.init_min_sep = float(init_min_sep)
        self._episode_idx = 0
        self._obs_tree = None

        # 构建仿真场景 (reset 时保持 builder 不变, 避免重复加载)
        self.builder, self.params = build_builder_and_sim(
            traj_path, env_path, models_dir, device=device, verbose=False)
        self.specs = make_specs(self.presets)
        self.origin, self.shape = make_grid_origin_shape(self.builder, voxel)

        # 时间窗
        frame_times = self.builder.traj.frame_times
        self.win_s = window_minutes * 60.0
        self.windows = []
        t0_all, t1_all = float(frame_times[0]), float(frame_times[-1])
        n_win = int((t1_all - t0_all + 1e-6) // self.win_s)
        for k in range(n_win):
            a = t0_all + k * self.win_s
            want = a + np.arange(0, int(self.win_s), DT_SAMPLE)
            idx = np.searchsorted(frame_times, want)
            idx = np.clip(idx, 0, len(frame_times) - 1)
            got = frame_times[idx]
            ok = np.abs(got - want) <= 0.5 * DT_SAMPLE
            if ok.any():
                self.windows.append(got[ok])

        self.max_steps = max_steps or max(len(self.windows) - 1, 1)

        # 随机数发生器 (先于初始位姿创建, 供每轮重采样使用)
        self.rng = np.random.default_rng(seed)

        # 初始位姿: 用户指定 (或自动生成) 作为"首轮/固定"位姿
        if initial_poses is not None:
            self._user_init_poses = np.asarray(initial_poses, np.float64).copy()
        else:
            # 自动生成: 道路两侧杆位
            self._user_init_poses = self._auto_initial_poses()
        self._initial_poses = self._user_init_poses.copy()

        # 内部状态
        self.cur_step = 0
        # 裕度盒锚点: 本轮初始位姿, 在 reset 时固定, 之后不再变化
        self._anchor_poses = self._initial_poses.copy()
        self.cur_poses = self._initial_poses.copy()
        self.history_grids = []
        self.all_poses = [self.cur_poses.copy()]

        # 观测/动作维度
        self.obs_dim_i = 6 + self.local_bev_size * self.local_bev_size + self.global_enc_dim
        self.action_dim_i = 6  # [dx, dy, dz, droll, dpitch, dyaw]

    # ------------------------------------------------------------------
    # 自动初始位姿生成 (道路两侧)
    # ------------------------------------------------------------------
    def _auto_initial_poses(self):
        traj = self.builder.traj
        x0, x1 = traj.front[:, 0].min(), traj.front[:, 0].max()
        y0, y1 = traj.front[:, 1].min(), traj.front[:, 1].max()
        cy = 0.5 * (y0 + y1)
        xs = np.arange(x0 + 4, x1 - 4, 8.0)
        poles = np.concatenate([
            np.stack([xs, np.full_like(xs, y0 - 3.0)], 1),
            np.stack([xs, np.full_like(xs, y1 + 3.0)], 1)], 0)
        sel = [0]
        while len(sel) < min(self.n_lidar, len(poles)):
            d = np.linalg.norm(poles[None, :, :] - poles[sel][:, None, :], axis=2)
            dmin = d.min(0)
            dmin[sel] = -1
            sel.append(int(np.argmax(dmin)))
        poles = poles[sel[:self.n_lidar]]
        out = []
        for p in poles:
            zg = float(self.builder.ground.height(p[None, :])[0])
            yaw = float(np.arctan2(cy - p[1], 0.5 * (x0 + x1) - p[0]))
            out.append([p[0], p[1], zg + 5.5, 0.0, 0.0, yaw])
        poses = np.asarray(out, np.float64)
        # 数量不够时复制
        while len(poses) < self.n_lidar:
            poses = np.vstack([poses, poses[len(poses) % len(out)][None]])
        return poses[:self.n_lidar]

    # ------------------------------------------------------------------
    # 随机初始位姿 (道路区域上空 + 附近无障碍 + 离地 init_z_range)
    # ------------------------------------------------------------------
    def _ensure_obs_tree(self):
        """惰性构建环境点云 XY KDTree, 用于障碍物净空检查."""
        if self._obs_tree is not None:
            return
        from scipy.spatial import cKDTree
        pts = np.asarray(self.builder.env_full.points)
        max_pts = 2_000_000
        if len(pts) > max_pts:
            step = int(np.ceil(len(pts) / max_pts))
            pts = pts[::step]
        self._obs_pts_xy = np.asarray(pts[:, :2], np.float64)
        self._obs_pts_z = np.asarray(pts[:, 2], np.float64)
        self._obs_tree = cKDTree(self._obs_pts_xy)

    def _is_clear(self, xy, zg):
        """xy 半径 init_clear_radius 内是否存在高于地面 1 m 的障碍点."""
        self._ensure_obs_tree()
        idx = self._obs_tree.query_ball_point(np.asarray(xy, np.float64),
                                              self.init_clear_radius)
        if not idx:
            return True
        zs = self._obs_pts_z[np.asarray(idx, np.int64)]
        return bool(np.all(zs <= zg + 1.0))

    def _road_point_and_tangent(self, rng):
        """在轨迹 (道路) 上随机取一点及其切向/法向 (XY 平面)."""
        front = np.asarray(self.builder.traj.front, np.float64)[:, :2]
        n = len(front)
        idx = int(rng.integers(0, n))
        i0, i1 = max(idx - 1, 0), min(idx + 1, n - 1)
        tang = front[i1] - front[i0]
        norm = float(np.linalg.norm(tang))
        if norm < 1e-6:
            tang = np.array([1.0, 0.0])
        else:
            tang = tang / norm
        normal = np.array([-tang[1], tang[0]])
        return front[idx], tang, normal

    def sample_random_initial_poses(self, max_try=200):
        """在道路区域上空随机采样一组初始位姿.

        约束:
          * 候选点位于轨迹附近 (横向抖动 init_road_jitter 以内), 且落在 POG 网格内;
          * 离地高度在 init_z_range (默认 5~8 m);
          * init_clear_radius 内无高于地面 1 m 的障碍物;
          * 同轮不同 LiDAR 水平间距 >= init_min_sep;
          * 朝向沿道路切向并叠加 ±30° 抖动, roll/pitch = 0.
        """
        rng = self.rng
        xlo = self.origin[0] + 2 * self.voxel
        xhi = self.origin[0] + self.shape[0] * self.voxel - 2 * self.voxel
        ylo = self.origin[1] + 2 * self.voxel
        yhi = self.origin[1] + self.shape[1] * self.voxel - 2 * self.voxel

        poses = []
        tries = 0
        while len(poses) < self.n_lidar and tries < max_try:
            tries += 1
            p, tang, normal = self._road_point_and_tangent(rng)
            xy = (p
                  + normal * rng.uniform(-self.init_road_jitter,
                                         self.init_road_jitter)
                  + tang * rng.uniform(-1.5, 1.5))
            if not (xlo <= xy[0] <= xhi and ylo <= xy[1] <= yhi):
                continue
            zg = float(self.builder.ground.height(xy[None, :])[0])
            if not self._is_clear(xy, zg):
                continue
            if any(np.linalg.norm(xy - np.asarray(q[:2])) < self.init_min_sep
                   for q in poses):
                continue
            z = zg + float(rng.uniform(*self.init_z_range))
            yaw = float(np.arctan2(tang[1], tang[0])) \
                + float(rng.uniform(-np.pi / 6, np.pi / 6))
            poses.append([xy[0], xy[1], z, 0.0, 0.0, yaw])

        if len(poses) < self.n_lidar:
            # 兜底: 用确定性布置补齐, 保证 reset 始终可用
            fallback = self._auto_initial_poses()
            poses.extend(fallback[len(poses):self.n_lidar].tolist())
        return np.asarray(poses[:self.n_lidar], np.float64)

    def meta(self):
        """记录训练环境的关键配置, 随权重一起保存 (供部署/续训一致性检查)."""
        return {
            "n_lidar": int(self.n_lidar),
            "presets": list(self.presets),
            "pos_margin": float(self.pos_margin),
            "ang_margin_deg": float(np.rad2deg(self.ang_rad)),
            "window_minutes": float(self.window_minutes),
            "local_bev_size": int(self.local_bev_size),
            "global_enc_dim": int(self.global_enc_dim),
            "global_bev_cells": GLOBAL_BEV_CELLS,
            "global_bev_cell_m": GLOBAL_BEV_CELL_M,
            "metric": self.metric,
            "voxel": float(self.voxel),
            "obs_dim": int(self.obs_dim_i),
            "state_dim": int(self.state_global_dim),
            "obs_frame": "relative_to_lidar0",
            "action_semantics": ACTION_SEMANTICS,
            "pose_bound": "anchored_to_initial_pose",
        }

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self):
        # 每轮 (第 2 轮起) 重采样初始位姿, 提升训练时的场景多样性
        self._episode_idx += 1
        if self.randomize_init_poses and self._episode_idx > 1:
            self._initial_poses = self.sample_random_initial_poses()
        else:
            self._initial_poses = self._user_init_poses.copy()

        self.cur_step = 0
        # 裕度盒锚点 = 本轮初始位姿: 整轮所有步都相对它偏移, 不随步数平移
        self._anchor_poses = self._initial_poses.copy()
        self.cur_poses = self._initial_poses.copy()
        self.history_grids = []
        self.all_poses = [self.cur_poses.copy()]

        # 首窗仿真 (初始位姿)
        if len(self.windows) >= 1:
            self._build_and_store_pog(0)
        return self._get_obs()

    def _decode_actions(self, actions_norm):
        """归一化动作 -> 本窗绝对位姿 (锚定初始位姿的固定盒内).

        * 位置: 局部系偏移 = a * pos_margin, 按**初始** LiDAR#0 朝向旋到世界系;
        * 角度: 直接在初始姿态上叠加 a * ang_rad;
        * 最后逐维投影到 [anchor - margin, anchor + margin], 硬保证裕度
          始终相对初始位姿, 且不随窗口累积.
        """
        anchor = self._anchor_poses
        new_poses = anchor.copy()
        yaw0 = float(anchor[0, 5])
        c0, s0 = np.cos(yaw0), np.sin(yaw0)
        for i in range(self.n_lidar):
            act = np.clip(np.asarray(actions_norm[i], np.float64), -1.0, 1.0)
            dx = act[0] * self.pos_margin
            dy = act[1] * self.pos_margin
            new_poses[i, 0] += c0 * dx - s0 * dy
            new_poses[i, 1] += s0 * dx + c0 * dy
            new_poses[i, 2] += act[2] * self.pos_margin
            new_poses[i, 3:] += act[3:] * self.ang_rad
        marg = np.array([self.pos_margin] * 3 + [self.ang_rad] * 3, np.float64)
        return np.clip(new_poses, anchor - marg, anchor + marg)

    def step(self, actions_norm):
        """actions_norm: dict[agent_id -> np.ndarray(6)] 范围 [-1,1].

        动作含义: 相对本 episode **初始位姿**的归一化偏移 (位置在初始
        LiDAR#0 局部系内解释), 解码后投影回 [初始位姿 ± 裕度] 固定盒.
        因此每个窗口的位姿都由"初始位姿 + 一次偏移"决定, 与上一窗无关,
        不会出现逐窗累积漂移.

        返回:
          obs_dict (agent_id -> np.ndarray(obs_dim))
          reward_global (float)
          done (bool)
          info (dict)
        """
        # 1) 解码动作 -> 相对初始位姿的偏移 -> 投影到固定盒内的位姿
        new_poses = self._decode_actions(actions_norm)

        # 2) 用新位姿构建 POG (下一窗)
        win_idx = self.cur_step + 1
        if win_idx < len(self.windows):
            self.cur_poses = new_poses
            self.all_poses.append(new_poses.copy())
            pog = self._build_and_store_pog(win_idx)
        else:
            # 窗口用完, 保持原位姿但仍存储
            self.cur_poses = new_poses
            self.all_poses.append(new_poses.copy())
            pog = None

        # 3) 计算奖励 (基于最新可用的历史 POG)
        # 奖励 = 方向统一后的指标得分 (越大越好):
        #   J_EIG/J_mv 直接取原值; J_cov/J_dyn/J_traj 取负值 (原值越小得分越高)
        grid = self.history_grids[-1] if self.history_grids else None
        if grid is not None and self.cur_step > 0:
            val = compute_surrogate(self.cur_poses, self.specs, grid,
                                    self.history_grids[-3:],
                                    metric=self.metric, n_ray=24)
            reward = metric_sign(self.metric) * val if np.isfinite(val) else 0.0
        else:
            reward = 0.0

        self.cur_step += 1
        done = (self.cur_step >= self.max_steps
                or self.cur_step >= len(self.windows))

        return self._get_obs(), float(reward), bool(done), {
            "step": self.cur_step, "poses": self.cur_poses.copy()}

    # ------------------------------------------------------------------
    # POG 构建辅助
    # ------------------------------------------------------------------
    def _build_and_store_pog(self, win_idx):
        times = self.windows[win_idx]
        sim = make_sim(self.builder, self.specs, self.cur_poses,
                       self.params, seed=self.seed)
        if self.workers > 1 and self._sim_pool is None:
            from rslaa.sim_pool import SimPool
            self._sim_pool = SimPool(self.traj_path, self.env_path,
                                     self.models_dir, device=self.device,
                                     workers=self.workers)
        pog = build_pog_occgrid(sim, times, self.origin, self.shape,
                                self.voxel, self.cur_poses, self.specs,
                                frame_step=self.frame_step,
                                pool=self._sim_pool, seed=self.seed)
        self.history_grids.append(pog)
        self.history_grids = self.history_grids[-5:]
        return pog

    def close(self):
        """释放并行仿真进程 (workers>1 时)."""
        if self._sim_pool is not None:
            self._sim_pool.close()
            self._sim_pool = None

    # ------------------------------------------------------------------
    # 观测构建
    # ------------------------------------------------------------------
    def _relative_pose(self, i):
        """agent i 相对 LiDAR#0 的位姿 (在 LiDAR#0 的局部坐标系下).

        返回 6 维 [x, y, z, droll, dpitch, dyaw]:
          * 平移 (dx,dy) 投影到 LiDAR#0 朝向坐标系 (x=前, y=左), z 取差值;
          * 角度取与 LiDAR#0 的差值并 wrap 到 [-pi, pi);
          * agent 0 恒为全 0.
        这样观测与场景的绝对位置/朝向无关, 便于跨场景泛化.
        """
        p0 = self.cur_poses[0]
        p = self.cur_poses[i]
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        c, s = np.cos(p0[5]), np.sin(p0[5])
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        return np.array([
            xr, yr, p[2] - p0[2],
            _wrap_angle(p[3] - p0[3]),
            _wrap_angle(p[4] - p0[4]),
            _wrap_angle(p[5] - p0[5]),
        ], np.float32)

    def _bev_sample(self, grid, center_xy, yaw, size, cell_m):
        """在 center_xy 处按 yaw 朝向采样 size×size 的 BEV 占用 (旋转对齐).

        网格坐标与 LiDAR#0 朝向对齐, 因此 patch 表征的是"相对传感器"的
        局部结构, 与场景绝对坐标无关 (平移/旋转一致).
        """
        p_col = grid.p.max(axis=2)  # (nx, ny)
        nx, ny = p_col.shape
        off = (np.arange(size) - (size - 1) * 0.5) * float(cell_m)
        OX, OY = np.meshgrid(off, off, indexing="ij")
        c, s = np.cos(yaw), np.sin(yaw)
        wx = center_xy[0] + c * OX - s * OY
        wy = center_xy[1] + s * OX + c * OY
        ix = np.floor((wx - self.origin[0]) / self.voxel).astype(np.int64)
        iy = np.floor((wy - self.origin[1]) / self.voxel).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros((size, size), np.float32)
        out[ok] = p_col[ix[ok], iy[ok]]
        return out

    def _local_bev_patch(self, grid, lidar_pose):
        """以该 LiDAR (x,y) 为中心、LiDAR#0 朝向对齐的 k×k BEV patch."""
        k = self.local_bev_size
        if grid is None:
            return np.zeros(k * k, np.float32)
        patch = self._bev_sample(grid, lidar_pose[:2], self.cur_poses[0, 5],
                                 k, self.voxel)
        return patch.ravel()

    def _global_pog_encoding(self, grid):
        """以 LiDAR#0 为原点的低分辨率 BEV + 统计量 (global_enc_dim 维).

        不再使用全局网格的绝对索引, 而是以 LiDAR#0 为中心采样固定物理范围
        (GLOBAL_BEV_CELLS × GLOBAL_BEV_CELL_M) 的 BEV, 再拼接与坐标无关的
        统计量. 换场景后编码分布保持一致, 有利于策略泛化.
        """
        out = np.zeros(self.global_enc_dim, np.float32)
        if grid is None:
            return out
        p = grid.p
        bev = self._bev_sample(grid, self.cur_poses[0, :2],
                               self.cur_poses[0, 5],
                               GLOBAL_BEV_CELLS, GLOBAL_BEV_CELL_M)
        stats = np.array([
            float(p.mean()), float(p.std()),
            float((p > 0.1).mean()), float((p > 0.5).mean()),
            float(bev.mean()), float(bev.std()),
            float((bev > 0.1).mean()), float((bev > 0.5).mean()),
        ], np.float32)
        flat = np.concatenate([bev.ravel(), stats])
        n = min(len(flat), self.global_enc_dim)
        out[:n] = flat[:n]
        return out

    def _get_obs(self):
        """为每个 agent 构造观测向量."""
        grid = self.history_grids[-1] if self.history_grids else None
        global_enc = np.zeros(self.global_enc_dim, np.float32)
        if grid is not None:
            global_enc = self._global_pog_encoding(grid)

        obs_dict = {}
        for i in range(self.n_lidar):
            # 相对 LiDAR#0 的位姿 (6)
            pose = self._relative_pose(i)
            # 局部 BEV 块 (k*k)
            local_patch = (self._local_bev_patch(grid, self.cur_poses[i])
                           if grid is not None
                           else np.zeros(self.local_bev_size ** 2, np.float32))
            obs = np.concatenate([pose, local_patch, global_enc])
            obs_dict[i] = obs
        return obs_dict

    # ------------------------------------------------------------------
    # 便捷访问
    # ------------------------------------------------------------------
    @property
    def state_global_dim(self):
        """中心化 Critic 的全局状态维度."""
        pose_dim = 6 * self.n_lidar
        grid_enc_dim = self.global_enc_dim
        return pose_dim + grid_enc_dim

    def get_global_state(self):
        """中心化 Critic 输入: 全部(相对)位姿 concat + 相对 POG 编码."""
        grid = self.history_grids[-1] if self.history_grids else None
        pose_flat = np.concatenate(
            [self._relative_pose(i) for i in range(self.n_lidar)])
        grid_enc = self._global_pog_encoding(grid) if grid is not None \
            else np.zeros(self.global_enc_dim, np.float32)
        return np.concatenate([pose_flat, grid_enc])

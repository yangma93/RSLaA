# -*- coding: utf-8 -*-
"""MAPPO 策略部署: 加载训好的 policy.pt, 在新场景上按时间窗滚动预测位姿.

流程 (与 MPC 框架一致):
  1. 加载 policy.pt (marl_trainer.py 输出)
  2. 加载初始位姿 lidar_pose.json 与新场景数据 (轨迹 + 环境 + 模型)
  3. 逐个时间窗:
     k=1:  使用初始位姿仿真, 构建 POG
     k>=2: 观测 o_i = f(当前 POG) -> Actor 输出"相对初始位姿"的归一化偏移
            -> 叠加到初始位姿并投影回 [初始位姿 ± 裕度] 固定盒 -> new_pose
           用 new_pose 仿真, 构建新 POG, 作为下窗历史
  4. 写出 lidar_pose_seq.json

关键差异 (vs MPC):
  * MPC: 优化替代指标 -> 最优位姿 (贪心, 无学习); 可行域锚定初始位姿
  * 部署: Actor 直接输出相对初始位姿的偏移 (基于学习到的策略), 同样锚定

裕度说明: pos_margin / ang_margin_deg 始终相对**初始位姿**, 所有窗口的可调
范围固定为 [init - margin, init + margin], 因此不会随窗口滚动产生漂移.
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
    VOXEL, DT_SAMPLE,
    METRIC_FUNCS, DEFAULT_METRIC, LOWER_IS_BETTER, metric_key,
    load_lidar_pose_json, save_lidar_pose_seq_json, append_pose_window,
    window_frames, build_builder_and_sim, make_sim, make_specs,
    make_grid_origin_shape, build_pog_occgrid,
    compute_surrogate, compute_all_surrogates, compute_surrogate_oriented,
)
from rslaa.marl_env import MultiLidarEnv, ACTION_SEMANTICS
from rslaa.marl_trainer import MAPPO


# ---------------------------------------------------------------------------
# 部署主循环
# ---------------------------------------------------------------------------
def run_deploy(policy_path: str, traj_path: str, env_path: str,
               models_dir: str, pose_json_path: str, out_seq_path: str,
               window_minutes: int = 5, voxel: float = VOXEL,
               device: str = "cuda", pos_margin: float = None,
               ang_margin_deg: float = None,
               global_enc_dim: int = None, local_bev_size: int = None,
               frame_step: int = 1, max_windows: int = 0,
               verbose: bool = True, seed: int = 0, workers: int = 1):
    """加载训好的策略, 在新场景上滚动推理生成位姿序列.

    参数:
      policy_path: policy.pt 路径
      traj_path/env_path/models_dir: 新场景数据
      pose_json_path: 初始位姿 lidar_pose.json
      out_seq_path: 输出 lidar_pose_seq.json
      window_minutes: 时间窗长度 (分钟)
      pos_margin / ang_margin_deg: 部署自由度, 即相对**初始位姿**的固定盒
          半宽; 缺省=训练值, 显式指定时可小于等于训练自由度, 超过会被裁剪并告警
      其他: 与训练时保持一致 (env 构建参数; 缺省取权重内记录的训练配置)
    """
    t_start = time.time()
    win_s = window_minutes * 60.0

    # ---- 1) 加载策略 (先读元信息, 用于确定自由度上限与观测配置) ----
    device_use = resolve_device(device)
    ckpt_meta = MAPPO.load_meta(policy_path)
    em = ckpt_meta.get("env_meta", {}) or {}
    if em.get("action_semantics") != ACTION_SEMANTICS and verbose:
        print("[Deploy] 警告: 权重内未记录锚定式动作语义 "
              f"(action_semantics={em.get('action_semantics')}), 该权重应是旧版"
              "'相对上一窗增量'训练的; 当前部署按'相对初始位姿偏移'解释动作, "
              "行为会不同, 建议用新版 marl_trainer 重新训练。")

    train_pos = em.get("pos_margin")
    train_ang = em.get("ang_margin_deg")
    if pos_margin is None:
        pos_margin = float(train_pos) if train_pos is not None else 0.5
    elif train_pos is not None and float(pos_margin) > float(train_pos) + 1e-9:
        if verbose:
            print(f"[Deploy] 指定位置自由度 {pos_margin} m 超过训练值 "
                  f"{train_pos} m, 已裁剪为训练值")
        pos_margin = float(train_pos)
    if ang_margin_deg is None:
        ang_margin_deg = float(train_ang) if train_ang is not None else 10.0
    elif train_ang is not None \
            and float(ang_margin_deg) > float(train_ang) + 1e-9:
        if verbose:
            print(f"[Deploy] 指定角度自由度 {ang_margin_deg}° 超过训练值 "
                  f"{train_ang}°, 已裁剪为训练值")
        ang_margin_deg = float(train_ang)

    if local_bev_size is None:
        local_bev_size = int(em.get("local_bev_size", 7))
    if global_enc_dim is None and em.get("global_enc_dim"):
        global_enc_dim = int(em["global_enc_dim"])

    model = MAPPO.load(policy_path, device=device_use)
    model.eval()
    if verbose:
        print(f"[Deploy] 策略已加载: {policy_path}")
        print(f"[Deploy] n_agent={model.n_agent}, obs_dim={model.obs_dim}, "
              f"act_dim={model.act_dim}, state_dim={model.state_dim}")
        _cap = (f"{train_pos} m / {train_ang}°"
                if train_pos is not None and train_ang is not None
                else "未知 (旧权重)")
        print(f"[Deploy] 自由度: 位置 ±{pos_margin} m, "
              f"角度 ±{ang_margin_deg}° (训练上限 {_cap})")

    # ---- 2) 加载初始位姿 ----
    pose_obj = load_lidar_pose_json(pose_json_path)
    presets = pose_obj["presets"]
    initial_poses = pose_obj["poses"]
    n_lidar = pose_obj["n_lidar"]
    if verbose:
        print(f"[Deploy] 初始位姿: {n_lidar} 台, 类型 {presets}")

    # ---- 3) 构建环境 (复用 MultiLidarEnv 的场景/窗口/POG 逻辑) ----
    env = MultiLidarEnv(
        traj_path=traj_path, env_path=env_path, models_dir=models_dir,
        presets=presets, initial_poses=initial_poses, n_lidar=n_lidar,
        window_minutes=window_minutes, max_steps=0,
        pos_margin=pos_margin, ang_margin_deg=ang_margin_deg,
        metric=DEFAULT_METRIC, voxel=voxel, device=device_use,
        local_bev_size=local_bev_size, global_enc_dim=global_enc_dim,
        seed=seed, frame_step=frame_step,
        randomize_init_poses=False, workers=workers,
    )

    if (model.n_agent != env.n_lidar
            or model.obs_dim != env.obs_dim_i
            or model.state_dim != env.state_global_dim):
        raise RuntimeError(
            "策略与当前环境不匹配: "
            f"policy(n_agent={model.n_agent}, obs={model.obs_dim}, "
            f"state={model.state_dim}) vs "
            f"env(n_agent={env.n_lidar}, obs={env.obs_dim_i}, "
            f"state={env.state_global_dim})。\n"
            "观测已改为以 LiDAR#0 为原点的相对表征, 请使用新版 "
            "marl_trainer 重新训练 (或用 --init-policy 微调) 后再部署。")

    if verbose:
        print(f"[Deploy] 时间窗: {len(env.windows)} x {window_minutes} min")
        print(f"[Deploy] 观测: 相对 LiDAR#0, obs_dim={env.obs_dim_i}, "
              f"state_dim={env.state_global_dim}")
        print(f"[Deploy] 可行域: 全窗锚定初始位姿, 位置 ±{pos_margin} m / "
              f"姿态 ±{ang_margin_deg}° (不随窗口漂移)")

    # ---- 4) 滚动推理 ----
    pose_seq_windows = []
    obs = env.reset()

    k = 1
    t0, t1 = float(env.windows[0][0]), float(env.windows[0][-1])
    append_pose_window(pose_seq_windows, k, [t0, t1], env.cur_poses)
    if verbose:
        print(f"\n[Deploy] === 窗口 {k} [{t0:.1f},{t1:.1f}] (初始位姿) ===")
        print(f"  poses:\n{np.round(env.cur_poses, 3)}")

    for k in range(2, len(env.windows) + 1):
        t0, t1 = float(env.windows[k - 1][0]), float(env.windows[k - 1][-1])
        if verbose:
            print(f"\n[Deploy] === 窗口 {k}/{len(env.windows)} "
                  f"[{t0:.1f},{t1:.1f}] ===")
            t0_k = time.time()

        # Actor 推理 (deterministic = False 可探索, 部署时默认 False 采样均值)
        actions_norm, _, _ = model.select_actions(obs, deterministic=True)
        obs, reward, done, info = env.step(actions_norm)
        append_pose_window(pose_seq_windows, k, [t0, t1], env.cur_poses)

        if verbose:
            print(f"  Actor 输入 obs_dim={len(obs[0])}, reward={reward:.4f}")
            print(f"  新位姿:\n{np.round(env.cur_poses, 3)}")
            print(f"  耗时 {time.time() - t0_k:.1f}s")

        if max_windows > 0 and k >= max_windows + 1:
            break

    # ---- 5) 写出 ----
    save_lidar_pose_seq_json(out_seq_path, presets, pose_seq_windows,
                             window_length_sec=win_s)
    env.close()
    if verbose:
        print(f"\n[Deploy] 完成! 共 {len(pose_seq_windows)} 个窗口, "
              f"总耗时 {time.time() - t_start:.1f}s")
        print(f"[Deploy] 序列已保存: {out_seq_path}")

    return pose_seq_windows


def resolve_device(device):
    """cuda 不可用时自动回退 cpu; 'auto' 解析成具体设备名 (torch.load 不支持 auto)."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if str(device) == "auto":
            return "cuda" if has_cuda else "cpu"
        if "cuda" in str(device) and not has_cuda:
            return "cpu"
        return device
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="policy.pt 路径")
    ap.add_argument("--pose-json", required=True, help="初始 lidar_pose.json")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True, help="输出 lidar_pose_seq.json")
    ap.add_argument("--window-min", type=int, default=5,
                    choices=[5, 15, 20, 30])
    ap.add_argument("--pos-margin", type=float, default=None,
                    help="相对初始位姿的位置盒半宽 (m); 缺省=训练值, "
                         "超过训练值自动裁剪")
    ap.add_argument("--ang-margin-deg", type=float, default=None,
                    help="相对初始位姿的姿态盒半宽 (°); 缺省=训练值, "
                         "超过训练值自动裁剪")
    ap.add_argument("--voxel", type=float, default=VOXEL)
    ap.add_argument("--global-enc-dim", type=int, default=None,
                    help="缺省取自权重内的训练配置")
    ap.add_argument("--local-bev-size", type=int, default=None,
                    help="缺省取自权重内的训练配置")
    ap.add_argument("--frame-step", type=int, default=2,
                    help="POG 帧步长, 与 rslaa_stage2_experiment.FRAME_STEP 保持一致")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1,
                    help="帧级并行仿真进程数 (>1 启用多进程 GPU 并行, "
                         "结果与串行逐位一致)")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_deploy(
        policy_path=args.policy,
        traj_path=args.traj,
        env_path=args.env,
        models_dir=args.models,
        pose_json_path=args.pose_json,
        out_seq_path=args.out,
        window_minutes=args.window_min,
        pos_margin=args.pos_margin,
        ang_margin_deg=args.ang_margin_deg,
        voxel=args.voxel,
        global_enc_dim=args.global_enc_dim,
        local_bev_size=args.local_bev_size,
        device=args.device,
        frame_step=args.frame_step,
        max_windows=args.max_windows,
        verbose=not args.quiet,
        seed=args.seed,
        workers=args.workers,
    )

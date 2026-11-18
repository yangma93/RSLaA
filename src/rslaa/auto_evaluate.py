# -*- coding: utf-8 -*-
"""多 LiDAR 自适应位姿配置下的 IoU / Dev / HA_diff / MR / Comp 自动化评价.

功能:
  * 导入 lidar_pose_seq.json (MPC 或 MARL 部署输出)
  * 自动识别时间窗长度与每窗位姿; 如无时间窗信息则视为静态单窗
  * 逐窗加载对应位姿 -> 构建仿真器 -> 逐帧 evaluate_frame
  * 五项指标逐帧保存 (口径见 rslaa.common.frame_metrics):
      IoU     全部 GT 个体的 3D IoU 均值 (未感知计 0)
      Dev     仅 IoU>0 个体, 感知/GT 底面中心 XY 距离均值 (m), 无可感知个体记 NaN
      HA_diff 仅 IoU>0 个体, 航向角绝对误差均值 (rad, pi/2 周期)
      MR      漏检率 = IoU<=0.15 的个体数 / 该帧 GT 个体数
      Comp    逐轨迹有效感知帧数 / 在场帧数 (IoU>0.15 视为有效感知)
  * 同时导出汇总 CSV + 完整 L-shape 评估 CSV (复用 lshape.save_evals_csv)

用法示例:
  python -m rslaa.auto_evaluate --traj traj.txt --env env.ply \
      --models models/ --pose-seq lidar_pose_seq.json --out out/eval_results
"""
from __future__ import annotations

import argparse
import csv
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
    VOXEL, DT_SAMPLE, MIN_PTS, DET_IOU_THR, METRIC_VERSION,
    frame_metrics, completeness, comp_per_instance, nanmean,
    load_lidar_pose_seq_json, is_static_pose_seq,
    build_builder_and_sim, make_sim, make_specs,
    eval_window,
)
from lidarsim.lshape import save_evals_csv


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_evaluate(traj_path: str, env_path: str, models_dir: str,
                 pose_seq_path: str, out_dir: str,
                 device: str = "auto", frame_step: int = 1,
                 min_pts: int = MIN_PTS, seed: int = 0,
                 verbose: bool = True, workers: int = 1):
    """自动化评价主函数.

    输出文件 (out_dir):
      iou_dev_series.csv   — 逐时间步 IoU / Dev 序列
      comp_summary.csv     — 逐个体 Comp 完整度
      evals_raw.csv        — 完整 L-shape 评估原始导出
      config.json          — 本次运行参数
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) 加载 pose seq ----
    ps = load_lidar_pose_seq_json(pose_seq_path)
    static = is_static_pose_seq(ps)
    presets = ps["presets"]
    windows_raw = ps["windows"]
    if verbose:
        print(f"[Eval] 读取 pose_seq: {len(windows_raw)} 个窗口, "
              f"static={static}, presets={presets}")

    # ---- 2) 加载场景 ----
    # 并行池先启动 (异步构建), 与父进程的场景加载重叠
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

    if static:
        # 静态模式: 单一位姿配置, 按窗口 t_range 过滤帧 (如有)
        frame_times = builder.traj.frame_times
        pose = windows_raw[0]["poses"]
        if "t_range" in windows_raw[0]:
            t0w, t1w = windows_raw[0]["t_range"]
            full_times = frame_times[(frame_times >= t0w - 1e-6)
                                     & (frame_times <= t1w + 1e-6)]
            if len(full_times) == 0:
                full_times = frame_times   # 过滤后空则回退全量
        else:
            full_times = frame_times
        if verbose:
            print(f"[Eval] 静态模式: pose shape={pose.shape}, "
                  f"帧数={len(full_times)}, "
                  f"t_range={windows_raw[0].get('t_range', 'all')}")
        all_windows = [(None, full_times, pose)]
    else:
        # 滚动模式: 每窗用对应位姿
        win_s = ps["window_length_sec"]
        frame_times = builder.traj.frame_times
        all_windows = []
        for w in windows_raw:
            t0w, t1w = w["t_range"]
            sel = frame_times[(frame_times >= t0w - 1e-6)
                              & (frame_times <= t1w + 1e-6)]
            if len(sel) == 0:
                continue
            all_windows.append((w["index"], sel, w["poses"]))
        if verbose:
            print(f"[Eval] 滚动模式: {len(all_windows)} 个窗口, "
                  f"window_length={win_s}s")

    # ---- 3) 逐窗评估 ----
    all_evals = []          # 全局帧评估列表
    series_rows = []        # 逐帧指标序列行
    for wi, (win_idx, times, poses_user) in enumerate(all_windows):
        t_start_win = time.time()
        if verbose:
            idx_str = f"win {win_idx}" if win_idx is not None else "static"
            t_range_str = f"[{float(times[0]):.0f},{float(times[-1]):.0f}]s"
            print(f"\n[Eval] === {idx_str} {t_range_str} ===")
            print(f"  poses:\n{np.round(poses_user, 3)}")

        sim = make_sim(builder, specs, poses_user, params, seed=seed)
        if pool is not None:
            win_evals = pool.evaluate(sim, times, poses_user, specs,
                                      min_pts=min_pts, frame_step=frame_step,
                                      seed=seed)
        else:
            from rslaa.sim_pool import evaluate_frames
            win_evals = evaluate_frames(sim, times[::frame_step], min_pts, seed)
        win_ms = []
        for ev in win_evals:
            m = frame_metrics(ev)
            win_ms.append(m)
            series_rows.append([
                float(ev.t),
                int(wi + 1) if win_idx is None else int(win_idx),
                float(m["iou_mean"]),
                float(m["dev_mean"]),
                float(m["ha_diff"]),
                float(m["mr"]),
                int(m["n_gt"]),
                int(m["n_detected"]),
                int(m["n_perceived"]),
                int(m["n_matched"]),
            ])

        all_evals.extend(win_evals)

        if verbose:
            print(f"  frames={len(win_evals)}, "
                  f"IoU={nanmean([m['iou_mean'] for m in win_ms], 0.0):.4f}, "
                  f"Dev={nanmean([m['dev_mean'] for m in win_ms]):.3f}m, "
                  f"HA={nanmean([m['ha_diff'] for m in win_ms]):.4f}rad, "
                  f"MR={nanmean([m['mr'] for m in win_ms]):.4f}, "
                  f"Comp={completeness(win_evals):.4f}")
            print(f"  耗时 {time.time() - t_start_win:.1f}s")

    if pool is not None:
        pool.close()

    # ---- 4) 汇总 (整段区间; Dev/HA/MR 均按"有定义的帧"平均) ----
    all_ms = [frame_metrics(ev) for ev in all_evals]
    global_iou = nanmean([m["iou_mean"] for m in all_ms], 0.0)
    global_dev = nanmean([m["dev_mean"] for m in all_ms])
    global_ha = nanmean([m["ha_diff"] for m in all_ms])
    global_mr = nanmean([m["mr"] for m in all_ms])
    global_comp = completeness(all_evals)
    comp_per_vid = comp_per_instance(all_evals)

    # ---- 5) 保存结果 ----
    # 逐帧指标序列
    iou_dev_path = os.path.join(out_dir, "iou_dev_series.csv")
    with open(iou_dev_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "window_idx", "iou_mean", "dev_mean",
                    "ha_diff_mae", "mr",
                    "n_gt", "n_detected", "n_perceived", "n_matched"])
        for row in series_rows:
            w.writerow(row)

    # Comp 逐个体
    comp_path = os.path.join(out_dir, "comp_summary.csv")
    with open(comp_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["vid", "vtype", "comp"])
        for vid, (vtype, comp) in sorted(comp_per_vid.items()):
            w.writerow([vid, vtype, f"{comp:.4f}"])
        w.writerow([])
        w.writerow(["global_mean_comp", f"{global_comp:.4f}"])

    # 完整 L-shape 原始导出 (逐个体逐帧; 帧级汇总已由上方五项指标文件给出,
    # 这里不再重复导出 lshape 旧口径的 mean_dev/mean_iou 以免口径混淆)
    raw_path = os.path.join(out_dir, "evals_raw.csv")
    save_evals_csv(all_evals, raw_path, extra_summary=False)

    # 配置 JSON
    cfg = dict(
        traj_path=os.path.abspath(traj_path),
        env_path=os.path.abspath(env_path) if env_path else None,
        models_dir=os.path.abspath(models_dir),
        pose_seq_path=os.path.abspath(pose_seq_path),
        static=static,
        n_windows=len(all_windows),
        presets=presets,
        device=device,
        frame_step=frame_step,
        workers=int(workers),
        min_pts=min_pts,
        metric_version=METRIC_VERSION,
        det_iou_thr=DET_IOU_THR,
        global_iou_mean=global_iou,
        global_dev_mean=_jf(global_dev),
        global_ha_diff_mae=_jf(global_ha),
        global_mr=_jf(global_mr),
        global_comp=float(global_comp),
    )
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n[Eval] 完成! 总耗时 {time.time() - t0:.1f}s")
        print(f"[Eval] 全局: IoU={global_iou:.4f}, Dev={global_dev:.3f}m, "
              f"HA_diff={global_ha:.4f}rad, MR={global_mr:.4f}, "
              f"Comp={global_comp:.4f}")
        print(f"[Eval] 结果已保存:")
        print(f"  {iou_dev_path}")
        print(f"  {comp_path}")
        print(f"  {raw_path}")

    return cfg


def _jf(x):
    """float -> JSON 可写值 (NaN/Inf 记 null)."""
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-seq", required=True,
                    help="lidar_pose_seq.json (MPC / MARL 部署输出)")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--frame-step", type=int, default=1,
                    help="帧采样步长 (默认 1 = 每帧)")
    ap.add_argument("--min-pts", type=int, default=MIN_PTS)
    ap.add_argument("--workers", type=int, default=1,
                    help="帧级并行仿真进程数 (>1 启用多进程 GPU 并行, "
                         "结果与串行逐位一致; 显存约 0.5~1 GB/进程)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_evaluate(
        traj_path=args.traj,
        env_path=args.env,
        models_dir=args.models,
        pose_seq_path=args.pose_seq,
        out_dir=args.out,
        device=args.device,
        frame_step=args.frame_step,
        min_pts=args.min_pts,
        seed=args.seed,
        verbose=not args.quiet,
        workers=args.workers,
    )

# -*- coding: utf-8 -*-
"""帧级并行仿真池: 多进程 x 独立 CUDA context, 用于点云仿真 / POG 累加 / 评价.

为什么用进程而不是线程:
  单帧仿真的 GPU 侧只有若干个小 kernel, 真正耗时的是 Python 发射与
  asnumpy 往返 (GIL 无法规避), 因此线程/多 stream 收益有限;
  按帧切给多个进程 (各自独立 CUDA context) 可并行掉这部分 CPU 时间.

实测 (RTX 3060 / i7-10700, scene_001, 300 帧窗口):
  2 进程 1.26x, 4 进程 1.49x (单帧 35.7 -> 24.0 ms).
  达不到线性: Windows 下多个 CUDA context 不能真正并发行 kernel,
  GPU 侧按时间片串行, 进程越多单帧越慢.
  另有启动开销 (每个进程重建 SceneBuilder 约 15~20 s), 只适合长任务.

精度约定 (关键):
  每帧随机数只由 (全局 seed, 帧时刻 t) 决定 —— 见 frame_seed.
  因此分块方式 / worker 数 / 是否并行都不会改变任何一帧的噪声与漏检,
  单进程与多进程结果逐位一致, 并行只影响耗时.

Windows 使用 spawn:
  子进程会各自重建 SceneBuilder, 故池对象应在一次任务开始前创建并复用,
  不要每窗重建; 显存占用约 0.5~1 GB/进程.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rslaa.common import build_builder_and_sim, make_sim
from lidarsim.lshape import evaluate_frame

# 子进程内共享的仿真场景 (每个进程一份, 初始化时构建一次)
_W = {}


def frame_seed(seed: int, t: float) -> np.random.SeedSequence:
    """帧级随机种子: 仅由 (全局 seed, 帧时刻) 决定, 与帧如何分块无关."""
    return np.random.SeedSequence([int(seed), int(round(float(t) * 1e6))])


def accumulate_occ(sim, times, origin, shape, voxel, seed: int = 0):
    """逐帧仿真并累加体素占用计数, 返回 (occ int32, n_used)."""
    occ = np.zeros(shape, np.int32)
    origin = np.asarray(origin, np.float64)
    lim = np.asarray(shape, np.int64)
    n_used = 0
    for t in np.asarray(times, np.float64):
        sim.rng = np.random.default_rng(frame_seed(seed, t))
        scene, clouds = sim.simulate_frame(float(t))
        pts = [c.xyz for c in clouds if c.xyz is not None and len(c.xyz)]
        if not pts:
            continue
        pts = np.vstack(pts)
        ij = np.floor((pts[:, :3] - origin) / voxel).astype(np.int64)
        ok = np.all((ij >= 0) & (ij < lim), axis=1)
        ij = ij[ok]
        flat = np.unique(ij[:, 0] * shape[1] * shape[2]
                         + ij[:, 1] * shape[2] + ij[:, 2])
        occ.reshape(-1)[flat] += 1
        n_used += 1
    return occ, n_used


def evaluate_frames(sim, times, min_pts: int = 5, seed: int = 0):
    """逐帧仿真 + L-shape 评价, 返回 FrameEval 列表 (按 t 升序)."""
    out = []
    for t in np.asarray(times, np.float64):
        sim.rng = np.random.default_rng(frame_seed(seed, t))
        scene, clouds = sim.simulate_frame(float(t))
        out.append(evaluate_frame(clouds, scene, min_points=min_pts))
    return out


# ---------------------------------------------------------------------------
# 子进程入口 (模块级, 供 spawn 导入)
# ---------------------------------------------------------------------------
def _init_worker(traj_path, env_path, models_dir, device, barrier):
    builder, params = build_builder_and_sim(traj_path, env_path, models_dir,
                                            device=device, verbose=False)
    _W["builder"] = builder
    _W["params"] = params
    barrier.wait(timeout=900)          # 等全部子进程就绪, 父进程据此排除启动耗时


def _pog_worker(job):
    times, poses, specs, origin, shape, voxel, seed = job
    sim = make_sim(_W["builder"], specs, poses, _W["params"], seed=seed)
    return accumulate_occ(sim, times, origin, shape, voxel, seed)


def _eval_worker(job):
    times, poses, specs, min_pts, seed = job
    sim = make_sim(_W["builder"], specs, poses, _W["params"], seed=seed)
    return evaluate_frames(sim, times, min_pts, seed)


# ---------------------------------------------------------------------------
# 池
# ---------------------------------------------------------------------------
class SimPool:
    """K 个常驻仿真进程; workers<=1 时不创建进程 (调用方自动走串行)."""

    def __init__(self, traj_path, env_path, models_dir, device="auto",
                 workers=4, verbose=False, wait=True):
        self.workers = max(1, int(workers))
        self._pool = None
        self._barrier = None
        self._verbose = verbose
        if self.workers > 1:
            ctx = mp.get_context("spawn")
            self._barrier = ctx.Barrier(self.workers + 1)
            self._pool = ctx.Pool(self.workers, initializer=_init_worker,
                                  initargs=(traj_path, env_path, models_dir,
                                            device, self._barrier))
            if verbose:
                print(f"[SimPool] 启动 {self.workers} 个仿真进程 "
                      f"(device={device}, 显存约 0.5~1 GB/进程) ...")
            if wait:
                self.wait_ready()

    def wait_ready(self):
        """等待全部子进程完成场景构建 (可与父进程自身的前置加载重叠)."""
        if self._pool is None or self._barrier is None:
            return
        t0 = time.time()
        self._barrier.wait(timeout=900)
        self._barrier = None
        if self._verbose:
            print(f"[SimPool] 全部仿真进程就绪 ({time.time() - t0:.1f}s)")

    # -- 按帧切块 -> 分发给子进程 --
    def _dispatch(self, fn, frames, make_job):
        k = min(self.workers, len(frames))
        bounds = np.linspace(0, len(frames), k + 1).astype(int)
        jobs = [make_job(frames[a:b]) for a, b in
                zip(bounds[:-1], bounds[1:]) if b > a]
        return self._pool.map(fn, jobs)

    def pog(self, sim, times, poses, specs, origin, shape, voxel,
            frame_step: int = 1, seed: int = 0):
        """并行构建占用计数; 返回 (occ int32, n_used).

        sim 仅用于串行回退 (帧过少 / 未启用并行).
        """
        frames = np.asarray(times, np.float64)[::max(1, int(frame_step))]
        if self._pool is None or len(frames) < 2 * self.workers:
            return accumulate_occ(sim, frames, origin, shape, voxel, seed)
        res = self._dispatch(
            _pog_worker, frames,
            lambda c: (c, poses, specs, origin, shape, voxel, seed))
        occ = np.zeros(shape, np.int32)
        n_used = 0
        for o, n in res:
            occ += o
            n_used += n
        return occ, n_used

    def evaluate(self, sim, times, poses, specs, min_pts: int = 5,
                 frame_step: int = 1, seed: int = 0):
        """并行评价; 返回按 t 升序的 FrameEval 列表."""
        frames = np.asarray(times, np.float64)[::max(1, int(frame_step))]
        if self._pool is None or len(frames) < 2 * self.workers:
            return evaluate_frames(sim, frames, min_pts, seed)
        res = self._dispatch(
            _eval_worker, frames,
            lambda c: (c, poses, specs, min_pts, seed))
        evals = [e for chunk in res for e in chunk]
        evals.sort(key=lambda e: e.t)
        return evals

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

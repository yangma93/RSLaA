# -*- coding: utf-8 -*-
"""rslaa 共享工具模块.

提供:
  * lidar_pose.json / lidar_pose_seq.json 的读写函数
  * 时间窗划分 (window_frames)
  * 仿真器构建 (build_sim)
  * POG 构建 (build_pog_occgrid, convert_to_metrics_grid)
  * 位姿内部序转换 (pose_internal) — 复用 metrics.pose_internal
  * 替代指标计算包装 (compute_surrogate)

所有代码不修改 lidarsim 既有源码, 仅在上层组合调用.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# 确保能导入 lidarsim 与 rslaa_bench
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lidarsim.trajectory import Trajectories
from lidarsim.pointcloud import load_env, make_flat_env
from lidarsim.meshes import ModelLibrary
from lidarsim.scene import SceneBuilder
from lidarsim.config import SimParams, make_lidar
from lidarsim.lidar_sim import LidarSimulator, LidarPose
from lidarsim.lshape import evaluate_frame, sequence_completeness, cumulative_completeness
from rslaa_bench.metrics import OccGrid, pose_internal
from rslaa.metric_n import (
    SURROGATES_N, METRIC_FUNCS_N, compute_all_n, build_task_roi,
    j_eig_n, j_cov_n, j_mv_n, j_dyn_n, j_traj_n,
)
from rslaa.paths import MODELS_DIR

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
VOXEL = 0.4             # POG 体素边长 (m)
DT_SAMPLE = 1.0         # 窗口采样间隔 (s)
MIN_PTS = 5             # L-shape 最少点数
ROI_MARGIN = 10.0       # ROI 轨迹包围盒外余量 (m)
ROAD_HALF_M = 5.0       # 任务 ROI 道路半宽 (m)
Z_LO, Z_HI = -0.5, 6.0  # z 向范围 (相对地面最低点)
POSE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]   # 用户序
POSE_UNIT = "rad"

METRIC_NAMES = [n for n, _ in SURROGATES_N]
METRIC_FUNCS = dict(METRIC_FUNCS_N)

# 指标方向约定: True 表示"数值越小, 感知效能越好".
# J_mv 与 IoU/Dev/Comp 等感知指标呈**正相关** (覆盖 + 多视角互补), 按"最大化"
# 处理, 并作为默认优化/奖励指标; J_EIG (信息增益) 同样最大化;
# J_cov / J_dyn / J_traj 与感知指标呈负相关, 按"最小化"处理.
LOWER_IS_BETTER = {"J_cov": True, "J_dyn": True, "J_traj": True,
                   "J_EIG": False, "J_mv": False}
DEFAULT_METRIC = "J_mv"


def metric_key(metric: str) -> str:
    """把大小写/别名统一成 METRIC_FUNCS 中的正式名称."""
    if metric in METRIC_FUNCS:
        return metric
    for name in METRIC_FUNCS:
        if name.lower() == str(metric).lower():
            return name
    return None


def metric_sign(metric: str) -> float:
    """优化方向: +1 = 越大越好 (最大化), -1 = 越小越好 (最小化)."""
    name = metric_key(metric)
    if name is None:
        return 1.0
    return -1.0 if LOWER_IS_BETTER.get(name, False) else 1.0


# ---------------------------------------------------------------------------
# JSON I/O: lidar_pose.json (静态输入) 与 lidar_pose_seq.json (时序输出)
# ---------------------------------------------------------------------------
def load_lidar_pose_json(path: str) -> dict:
    """读取 lidar_pose.json.

    返回 dict: n_lidar, presets, poses (N,6), pose_order, pose_unit.
    poses 为用户序 [x y z roll pitch yaw], 弧度.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    obj["poses"] = np.asarray(obj["poses"], np.float64)
    assert obj["poses"].ndim == 2 and obj["poses"].shape[1] == 6, \
        f"poses 应为 (N,6), 实际 {obj['poses'].shape}"
    if "presets" not in obj:
        obj["presets"] = ["vlp16"] * obj["poses"].shape[0]
    if "n_lidar" not in obj:
        obj["n_lidar"] = obj["poses"].shape[0]
    return obj


def load_lidar_pose_seq_json(path: str) -> dict:
    """读取 lidar_pose_seq.json.

    返回 dict: window_length_sec, windows (list[dict{index, t_range, poses}]),
    presets, pose_order, pose_unit. poses 已转为 np.ndarray.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    for w in obj["windows"]:
        w["poses"] = np.asarray(w["poses"], np.float64)
    return obj


def is_static_pose_seq(obj: dict) -> bool:
    """判断 pose_seq 是否为静态单窗 (无 window_length_sec 或仅一个窗口)."""
    return obj.get("window_length_sec") is None or len(obj.get("windows", [])) <= 1


def save_lidar_pose_seq_json(path: str, presets: list,
                             windows: list[dict], window_length_sec: float = None,
                             pose_order: list = None, pose_unit: str = None):
    """保存 lidar_pose_seq.json.

    windows: [{"index": int, "t_range": [t0, t1], "poses": np.ndarray(N,6)}, ...]
    如果 window_length_sec 为 None, 则保存为静态单窗口格式.
    """
    windows_out = []
    for w in windows:
        windows_out.append({
            "index": int(w["index"]),
            "t_range": [float(w["t_range"][0]), float(w["t_range"][1])],
            "poses": np.asarray(w["poses"], np.float64).tolist(),
        })
    obj = dict(
        presets=list(presets),
        pose_order=pose_order or POSE_KEYS,
        pose_unit=pose_unit or POSE_UNIT,
        windows=windows_out,
    )
    if window_length_sec is not None:
        obj["window_length_sec"] = float(window_length_sec)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_pose_window(windows: list, index: int, t_range: list,
                       poses: np.ndarray) -> list:
    """向 windows 列表追加一个窗口 (就地修改并返回)."""
    windows.append({
        "index": int(index),
        "t_range": [float(t_range[0]), float(t_range[1])],
        "poses": np.asarray(poses, np.float64).copy(),
    })
    return windows


# ---------------------------------------------------------------------------
# 时间窗划分
# ---------------------------------------------------------------------------
def window_frames(frame_times: np.ndarray, win_s: float,
                  dt_sample: float = DT_SAMPLE) -> list[np.ndarray]:
    """把帧时间划分为连续窗口, 每窗按 dt_sample Hz 采样.

    返回 list[np.ndarray]: 每个窗口的仿真时刻序列.
    """
    t0, t1 = float(frame_times[0]), float(frame_times[-1])
    n_win = int((t1 - t0 + 1e-6) // win_s)
    windows = []
    for k in range(n_win):
        a = t0 + k * win_s
        want = a + np.arange(0, int(win_s), dt_sample)
        idx = np.searchsorted(frame_times, want)
        idx = np.clip(idx, 0, len(frame_times) - 1)
        got = frame_times[idx]
        ok = np.abs(got - want) <= 0.5 * dt_sample
        windows.append(got[ok])
    return [w for w in windows if len(w) > 0]


# ---------------------------------------------------------------------------
# 仿真器 / 构建器 工厂
# ---------------------------------------------------------------------------
def build_builder_and_sim(traj_path: str, env_path: str = None,
                          models_dir: str = None, device: str = "auto",
                          verbose: bool = True):
    """加载轨迹/环境/模型, 构建 SceneBuilder + 初始 SimParams."""
    traj = Trajectories.load(traj_path)
    env = load_env(env_path) if env_path else make_flat_env(traj)
    if models_dir is None:
        models_dir = str(MODELS_DIR)
    models = ModelLibrary(models_dir)
    params = SimParams(device=device)
    builder = SceneBuilder(traj, env, models, verbose=verbose)
    return builder, params


def make_sim(builder, specs: list, poses_user: np.ndarray,
             params: SimParams = None, seed: int = 0) -> LidarSimulator:
    """构建多 LiDAR 仿真器 (poses_user 为用户序 (N,6))."""
    poses_user = np.asarray(poses_user, np.float64)
    lidars = []
    for spec, pu in zip(specs, poses_user):
        lidars.append(LidarPose(spec=spec, pose=pose_internal(pu)))
    p = params or SimParams(device="auto", seed=seed)
    return LidarSimulator(builder, lidars, p)


def make_specs(presets: list) -> list:
    """presets 字符串列表 -> LidarSpec 列表."""
    return [make_lidar(p) for p in presets]


# ---------------------------------------------------------------------------
# ROI / 网格尺寸
# ---------------------------------------------------------------------------
def make_grid_origin_shape(builder: SceneBuilder,
                           voxel: float = VOXEL,
                           roi_margin: float = ROI_MARGIN) -> tuple:
    """由轨迹包围盒 + 余量确定体素网格 origin 与 shape (用于 POG)."""
    traj = builder.traj
    x0, x1 = traj.front[:, 0].min(), traj.front[:, 0].max()
    y0, y1 = traj.front[:, 1].min(), traj.front[:, 1].max()
    x0, x1 = x0 - roi_margin, x1 + roi_margin
    y0, y1 = y0 - roi_margin, y1 + roi_margin
    gx = np.linspace(x0, x1, 40)
    gy = np.linspace(y0, y1, 20)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    gz = builder.ground.height(np.stack([GX.ravel(), GY.ravel()], 1))
    z0, z1 = float(gz.min()) + Z_LO, float(gz.max()) + Z_HI
    origin = np.array([x0, y0, z0], np.float64)
    shape = tuple(np.ceil((np.array([x1, y1, z1]) - origin) / voxel).astype(int))
    return origin, shape


# ---------------------------------------------------------------------------
# POG 构建 -> metrics.OccGrid (含几何可观测掩码)
# ---------------------------------------------------------------------------
def _geom_obs_mask(poses_user, specs, origin, shape, voxel) -> np.ndarray:
    """几何可观测掩码 (0/1): 体素中心在任一 LiDAR 的 FoV 与量程内."""
    nx, ny, nz = shape
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
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


def build_pog_occgrid(sim: LidarSimulator, times: np.ndarray,
                      origin: np.ndarray, shape: tuple,
                      voxel: float, poses_user: np.ndarray, specs: list,
                      frame_step: int = 1, pool=None,
                      seed: int = 0) -> OccGrid:
    """对窗口逐帧仿真, 累计体素占用 -> OccGrid (p, o, origin, voxel).

    p: 占用概率 (nx,ny,nz) float32, 0~1
    o: 几何可观测掩码 (nx,ny,nz) float64, 0/1
    pool: 可选的 SimPool (多进程帧级并行); None 表示串行.
          由于每帧随机数由 (seed, t) 决定, 串行与并行的占用计数逐位一致.
    """
    from rslaa.sim_pool import accumulate_occ
    frames = np.asarray(times, np.float64)[::max(1, int(frame_step))]
    if pool is not None:
        occ, n_used = pool.pog(sim, frames, poses_user, specs, origin, shape,
                               voxel, frame_step=1, seed=seed)
    else:
        occ, n_used = accumulate_occ(sim, frames, origin, shape, voxel, seed)
    p = (occ / max(n_used, 1)).astype(np.float32)
    o = _geom_obs_mask(poses_user, specs, origin, shape, voxel)
    grid = OccGrid(p, o, origin, voxel)
    # 任务 ROI (道路走廊) 随网格一起生成, 供 v2 指标直接使用
    grid.roi_col = task_roi(grid, sim.builder.traj.front)
    return grid


# ---------------------------------------------------------------------------
# 任务 ROI (修复 v2 指标的必要输入)
# ---------------------------------------------------------------------------
def task_roi(grid: OccGrid, traj_front=None) -> np.ndarray:
    """BEV 任务区域掩码 (nx,ny) bool.

    优先: 轨迹散点 + 横向膨胀 ROAD_HALF_M 形成道路走廊;
    无轨迹时回退: 活跃占用柱膨胀, 再回退到几何可观测区域.
    """
    if traj_front is not None and len(np.asarray(traj_front)):
        return build_task_roi(grid, np.asarray(traj_front)[:, :2])

    cached = getattr(grid, "roi_col", None)          # build_pog_occgrid 已挂载
    if cached is not None:
        return cached

    from scipy.ndimage import binary_dilation
    act = grid.column_max() > 0.02
    if act.sum() < 50:
        act = grid.o.max(axis=2) > 0
    if act.sum() < 10:
        act = np.ones(act.shape, bool)
    k = max(1, int(np.round(ROAD_HALF_M / grid.voxel)))
    return binary_dilation(act, np.ones((2 * k + 1, 2 * k + 1), bool))


# ---------------------------------------------------------------------------
# 替代指标计算包装
# ---------------------------------------------------------------------------
def compute_surrogate(poses_user: np.ndarray, specs: list,
                      grid: OccGrid, hist: list,
                      metric: str = DEFAULT_METRIC, n_ray: int = 48) -> float:
    """基于历史占用图 + 候选联合位姿计算指定指标值.

    metric: 可选 5 种之一 (大小写不敏感). 默认 J_mv.
    返回 float 为**原始指标值**; 其优化方向由 metric_sign() 给出
    (J_cov/J_dyn/J_traj 越小越好, J_EIG/J_mv 越大越好). 指标不适用时返回 NaN.
    """
    name_map = {k.lower(): v for k, v in METRIC_FUNCS.items()}
    key = metric if metric in METRIC_FUNCS else str(metric).lower()
    fn = METRIC_FUNCS.get(key) or name_map.get(key) or j_cov_n
    # ROI = 任务区域 (道路走廊), 优先取 build_pog_occgrid 挂载的 roi_col
    roi_col = getattr(grid, "roi_col", None)
    if roi_col is None:
        roi_col = task_roi(grid)
    return float(fn(list(poses_user), specs, grid, hist, roi_col, n_ray=n_ray))


def compute_all_surrogates(poses_user: np.ndarray, specs: list,
                           grid: OccGrid, hist: list,
                           n_ray: int = 48) -> list[float]:
    """一次计算五个替代指标 (v2)."""
    roi_col = getattr(grid, "roi_col", None)
    if roi_col is None:
        roi_col = task_roi(grid)
    return compute_all_n(list(poses_user), specs, grid, hist, roi_col, n_ray=n_ray)


def compute_surrogate_oriented(poses_user: np.ndarray, specs: list,
                               grid: OccGrid, hist: list,
                               metric: str = DEFAULT_METRIC,
                               n_ray: int = 48) -> float:
    """方向统一后的指标得分: **始终越大越好**.

    - J_EIG / J_mv: 直接返回原值 (越大越好);
    - J_cov / J_dyn / J_traj: 返回 -原值 (原值越小, 得分越高).
    """
    val = compute_surrogate(poses_user, specs, grid, hist,
                            metric=metric, n_ray=n_ray)
    return metric_sign(metric) * val


# ---------------------------------------------------------------------------
# 帧级评价指标 (metric_version = 2)
# ---------------------------------------------------------------------------
# 口径约定 (与 rslaa_stage3_evaluate 的读取/汇总保持一致):
#   IoU     : 该帧全部 GT 个体的 3D IoU 均值, 未感知个体计 0.
#   Dev     : 仅 IoU>0 的可感知个体, 感知框与 GT 框底面中心 XY 欧氏距离的均值 (m);
#             该帧没有任何可感知个体时记 NaN (不参与平均, 否则"全漏检"会伪装成高精度).
#   HA_diff : 仅 IoU>0 的可感知个体, 航向角绝对误差的均值 (rad).
#   MR      : 漏检率 = 未检出个体数 / 该帧 GT 个体数, 检出判据 IoU > DET_IOU_THR;
#             GT 个体数为 0 的帧无定义 -> NaN.
#   Comp    : 逐轨迹统计, 有效感知判据同为 IoU > DET_IOU_THR.
DET_IOU_THR = 0.15
METRIC_VERSION = 2


def nanmean(values, default=np.nan) -> float:
    """忽略 None / NaN 的均值; 全为空时返回 default."""
    v = np.asarray([x for x in values
                    if x is not None and np.isfinite(float(x))], np.float64)
    return float(v.mean()) if v.size else float(default)


def box_yaw(box: np.ndarray) -> float:
    """8 角点盒的航向角 (rad).

    角点顺序与 scene.VehicleInstance.obb() / lshape.obb_corners() 一致:
    角点 0 -> 4 为局部 x 轴 (车长方向), 故 yaw = atan2(Δy, Δx).
    """
    b = np.asarray(box, np.float64)
    return float(np.arctan2(b[4, 1] - b[0, 1], b[4, 0] - b[0, 0]))


def box_bottom_center_xy(box: np.ndarray) -> np.ndarray:
    """8 角点盒的底面中心 (x, y); 底面角点为 0/2/4/6."""
    b = np.asarray(box, np.float64)
    return b[[0, 2, 4, 6], :2].mean(axis=0)


def heading_error(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """两盒航向角差异绝对值 (rad), 以 pi/2 为周期.

    矩形绕中心旋转 90° 并交换长宽即为同一盒子, L-shape 拟合也只能把朝向
    确定到 90° (fit_lshape_2d 的 theta in [0, pi/2)), 因此差值折叠到
    [-pi/4, pi/4): 否则 0° 与 89.9° 这种等价朝向会被误判成粗差.
    """
    d = box_yaw(box_a) - box_yaw(box_b)
    return float(abs((d + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0))


def is_perceived(ins, iou_thr: float = DET_IOU_THR) -> bool:
    """个体是否被有效感知 (成功拟合且 IoU 超过检出阈值)."""
    return ins.fit_box is not None and float(ins.iou3d) > iou_thr


def frame_metrics(ev, iou_thr: float = DET_IOU_THR) -> dict:
    """单帧五项指标 (口径见文件头注释), 返回 dict."""
    insts = list(getattr(ev, "instances", []) or [])
    n_gt = int(getattr(ev, "n_gt", 0) or 0)

    ious = [float(ins.iou3d) for ins in insts]
    perceived = [ins for ins in insts
                 if ins.fit_box is not None and ins.gt_box is not None
                 and float(ins.iou3d) > 0.0]

    devs, has = [], []
    for ins in perceived:
        devs.append(float(np.linalg.norm(
            box_bottom_center_xy(ins.fit_box) - box_bottom_center_xy(ins.gt_box))))
        has.append(heading_error(ins.fit_box, ins.gt_box))

    n_detected = sum(1 for ins in insts if float(ins.iou3d) > iou_thr)
    return dict(
        iou_mean=nanmean(ious),
        dev_mean=nanmean(devs),
        ha_diff=nanmean(has),
        mr=(float(1.0 - n_detected / n_gt) if n_gt > 0 else np.nan),
        n_gt=n_gt,
        n_detected=int(n_detected),
        n_perceived=len(perceived),
        n_matched=int(getattr(ev, "n_matched", 0) or 0),
    )


def completeness(evals, iou_thr: float = DET_IOU_THR) -> float:
    """区间轨迹完整度均值 (0~1): 个体有效感知帧数 / 在场帧数.

    有效感知判据为 IoU > iou_thr (而非仅"点数足够"), 与 MR 口径一致.
    """
    pres, mat = {}, {}
    for ev in evals:
        for ins in ev.instances:
            pres[ins.vid] = pres.get(ins.vid, 0) + 1
            if is_perceived(ins, iou_thr):
                mat[ins.vid] = mat.get(ins.vid, 0) + 1
    if not pres:
        return 0.0
    return float(np.mean([mat.get(v, 0) / n for v, n in pres.items()]))


def comp_per_instance(evals, iou_thr: float = DET_IOU_THR) -> dict:
    """逐个体 Comp: {vid: (vtype, 有效感知帧数 / 在场帧数)}."""
    pres, mat, vtype_of = {}, {}, {}
    for ev in evals:
        for ins in ev.instances:
            pres[ins.vid] = pres.get(ins.vid, 0) + 1
            vtype_of.setdefault(ins.vid, ins.vtype)
            if is_perceived(ins, iou_thr):
                mat[ins.vid] = mat.get(ins.vid, 0) + 1
    return {vid: (vtype_of.get(vid, -1), mat.get(vid, 0) / n)
            for vid, n in pres.items()}


# ---------------------------------------------------------------------------
# IoU / Dev / Comp 评估辅助
# ---------------------------------------------------------------------------
def eval_window(builder, specs, poses_user, times,
                params=None, seed=0, min_pts=MIN_PTS):
    """对窗口内的多帧进行 IoU/Dev/Comp 评估 (新口径).

    返回 (iou_mean, dev_mean, comp, evals_list); dev_mean 为逐帧 Dev 的均值,
    无定义帧不计入 (全无定义时返回 NaN).
    """
    sim = make_sim(builder, specs, poses_user, params, seed=seed)
    evals = []
    for t in times:
        scene, clouds = sim.simulate_frame(float(t))
        evals.append(evaluate_frame(clouds, scene, min_points=min_pts))
    ms = [frame_metrics(ev) for ev in evals]
    iou = nanmean([m["iou_mean"] for m in ms], 0.0)
    dev = nanmean([m["dev_mean"] for m in ms])
    comp = completeness(evals)
    return iou, dev, comp, evals

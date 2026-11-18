# -*- coding: utf-8 -*-
"""LiDAR 扫描模式生成与传感器位姿.

位姿约定 [x y z roll yaw pitch] (世界坐标系, 单位 m / rad):
  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
旋转扫描: 方位角列在扫描周期内匀速旋转, 每列光束带独立时间戳,
对运动车辆按光束时刻插值位姿 -> 自然产生运动伪影 (rolling shutter).
"""
import numpy as np

from .config import LidarSpec


def pose_matrix(pose) -> np.ndarray:
    x, y, z, roll, yaw, pitch = [float(v) for v in pose]
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


def scan_pattern(spec: LidarSpec, pose):
    """生成一帧扫描的所有光束.

    返回 origins (R,3), dirs (R,3) 单位向量, t_frac (R,) 光束在扫描周期内的
    相对时刻 [0,1) (用于车辆位姿插值 -> 运动伪影).
    """
    R_l = pose_matrix(pose)
    origin = np.asarray(pose[:3], np.float64)
    n_az = int(round(spec.fov_h_deg / spec.h_res_deg)) if spec.fov_h_deg < 360 \
        else int(round(360.0 / spec.h_res_deg))
    if spec.fov_h_deg < 360:
        az = np.linspace(-spec.fov_h_deg / 2, spec.fov_h_deg / 2, n_az, endpoint=False)
    else:
        az = np.arange(n_az) * spec.h_res_deg  # [0,360)
    elev = np.linspace(spec.vfov_min_deg, spec.vfov_max_deg, spec.channels) * np.pi / 180
    az_r = az * np.pi / 180
    # 网格: 每列 (方位) x 每通道
    AZ, EL = np.meshgrid(az_r, elev, indexing="ij")     # (n_az, ch)
    ct = np.cos(EL)
    dirs = np.stack([ct * np.cos(AZ), ct * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
    dirs_w = dirs @ R_l.T
    # 光束时刻: 方位角在扫描周期内线性推进
    if spec.fov_h_deg < 360:
        t_frac = (AZ[:, 0] / (spec.fov_h_deg * np.pi / 180)).repeat(spec.channels)
    else:
        t_frac = (az_r / (2 * np.pi)).repeat(spec.channels)
    origins = np.broadcast_to(origin, dirs_w.shape).copy()
    return (origins.astype(np.float32), dirs_w.astype(np.float32),
            np.ascontiguousarray(t_frac, np.float32))

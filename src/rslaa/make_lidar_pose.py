# -*- coding: utf-8 -*-
"""生成 MPC 可直接导入的 lidar_pose.json.

输入: LiDAR 数量、每台类型、每台位姿 [x, y, z, roll, pitch, yaw]
输出: lidar_pose.json (字段 n_lidar / presets / poses / pose_order / pose_unit)

两种用法
--------
1) 命令行 (推荐):
   python -m rslaa.make_lidar_pose --out pose.json --n-lidar 2 \\
       --presets vlp16,vlp16 \\
       --poses "6355,1860,207,0,0,0; 6500,1965,207,0,0,180" --unit deg

2) 交互式 (不传 --poses 时逐项询问):
   python -m rslaa.make_lidar_pose --out pose.json

角度单位: --unit deg 时 roll/pitch/yaw 按角度输入并自动转弧度写入;
         --unit rad 时原样写入. 位置始终为米.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:                                   # 轻量导入, 仅读取型号表
    from lidarsim.config import LIDAR_PRESETS
    VALID_PRESETS = list(LIDAR_PRESETS.keys())
except Exception:                      # pragma: no cover
    VALID_PRESETS = ["vlp16", "hdl32", "os1_64", "pandar40", "rs128"]

POSE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
POSE_UNIT = "rad"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse_poses(text: str) -> np.ndarray:
    """解析位姿串: 行间用 ';' 或换行分隔, 每行 6 个数值, 逗号/空格分隔."""
    rows = []
    for chunk in text.replace(";", "\n").splitlines():
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [v for v in chunk.replace(",", " ").split() if v]
        if len(vals) != 6:
            raise ValueError(f"每行需 6 个数值 [x y z roll pitch yaw], 实际 {len(vals)}: {chunk}")
        rows.append([float(v) for v in vals])
    if not rows:
        raise ValueError("未解析到任何位姿")
    return np.asarray(rows, np.float64)


def load_poses_file(path: str) -> np.ndarray:
    """读取位姿文件 (.csv/.txt), 支持逗号或空白分隔."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return parse_poses(f.read())


def parse_presets(text: str, n_lidar: int) -> list:
    """解析型号串: 逗号分隔; 少于 n_lidar 时按最后一个循环补齐."""
    names = [s.strip() for s in text.replace(";", ",").split(",") if s.strip()]
    if not names:
        names = ["vlp16"]
    out = [names[i % len(names)] for i in range(n_lidar)]
    bad = sorted({n for n in out if n not in VALID_PRESETS})
    if bad:
        raise ValueError(f"未知型号 {bad}, 可选: {VALID_PRESETS}")
    return out


def interactive_input() -> tuple:
    """交互式采集 数量 / 类型 / 位姿 / 角度单位."""
    print("=" * 60)
    print("生成 lidar_pose.json (MPC 输入文件)")
    print(f"可用型号: {', '.join(VALID_PRESETS)}")
    print("=" * 60)

    n = int(input("LiDAR 数量: ").strip() or "1")
    unit = (input("角度单位 [deg/rad] (默认 deg): ").strip() or "deg").lower()
    if unit not in ("deg", "rad"):
        unit = "deg"

    presets, poses = [], []
    for i in range(n):
        print(f"\n--- LiDAR {i + 1}/{n} ---")
        p = input(f"  类型 (默认 vlp16): ").strip() or "vlp16"
        if p not in VALID_PRESETS:
            print(f"  未知型号, 回退 vlp16 (可选: {VALID_PRESETS})")
            p = "vlp16"
        presets.append(p)
        raw = input("  位姿 x,y,z,roll,pitch,yaw: ").strip()
        vals = [v for v in raw.replace(",", " ").split() if v]
        if len(vals) != 6:
            raise ValueError(f"需要 6 个数值, 实际输入 {len(vals)} 个")
        poses.append([float(v) for v in vals])

    return np.asarray(poses, np.float64), presets, unit


# ---------------------------------------------------------------------------
# 校验 + 写出
# ---------------------------------------------------------------------------
def normalize_poses(poses: np.ndarray, unit: str) -> np.ndarray:
    """角度统一为弧度; 位置保持米."""
    out = np.asarray(poses, np.float64).copy()
    if unit == "deg":
        out[:, 3:6] = np.deg2rad(out[:, 3:6])
    elif unit != "rad":
        raise ValueError(f"unit 需为 deg 或 rad, 实际 {unit}")
    return out


def validate(poses: np.ndarray, presets: list, n_lidar: int):
    """数量与数值合法性检查."""
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"poses 需为 (N,6), 实际 {poses.shape}")
    if len(presets) != len(poses):
        raise ValueError(f"presets 数量({len(presets)}) 与 poses 行数({len(poses)}) 不一致")
    if n_lidar is not None and n_lidar != len(poses):
        raise ValueError(f"--n-lidar({n_lidar}) 与位姿行数({len(poses)}) 不一致")
    if not np.all(np.isfinite(poses)):
        raise ValueError("位姿含 NaN/Inf")
    for i, p in enumerate(poses):
        if abs(p[3]) > np.pi or abs(p[4]) > np.pi:
            raise ValueError(f"LiDAR {i}: roll/pitch 超出 ±180°")


def save_lidar_pose_json(path: str, poses: np.ndarray, presets: list,
                         unit: str = "rad"):
    """写出符合 MPC 读取格式的 lidar_pose.json."""
    obj = {
        "n_lidar": int(len(poses)),
        "presets": list(presets),
        "poses": np.round(np.asarray(poses, np.float64), 8).tolist(),
        "pose_order": POSE_KEYS,
        "pose_unit": POSE_UNIT,
    }
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    # 回读自检: 确保 MPC 侧能正常解析
    from rslaa.common import load_lidar_pose_json
    chk = load_lidar_pose_json(path)
    assert chk["n_lidar"] == len(poses), "回读 n_lidar 不一致"
    assert chk["poses"].shape == poses.shape, "回读 poses 形状不一致"
    return obj


def print_summary(poses: np.ndarray, presets: list, unit_in: str):
    """打印位姿汇总表 (角度同时显示 deg)."""
    print()
    print(f"{'ID':>3} {'preset':<9} {'x(m)':>10} {'y(m)':>10} {'z(m)':>9} "
          f"{'roll(°)':>9} {'pitch(°)':>9} {'yaw(°)':>9}")
    print("-" * 76)
    for i, (p, nm) in enumerate(zip(poses, presets)):
        print(f"{i:>3} {nm:<9} {p[0]:>10.3f} {p[1]:>10.3f} {p[2]:>9.3f} "
              f"{np.rad2deg(p[3]):>9.2f} {np.rad2deg(p[4]):>9.2f} {np.rad2deg(p[5]):>9.2f}")
    print("-" * 76)
    print(f"角度输入单位: {unit_in} (已统一写入为弧度)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="生成 MPC 可导入的 lidar_pose.json",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="lidar_pose.json",
                    help="输出 JSON 路径 (默认 ./lidar_pose.json)")
    ap.add_argument("--n-lidar", type=int, default=None,
                    help="LiDAR 数量 (缺省由位姿行数推断)")
    ap.add_argument("--presets", default="vlp16",
                    help=f"型号, 逗号分隔; 少于数量时循环补齐. "
                         f"可选: {', '.join(VALID_PRESETS)}")
    ap.add_argument("--poses", default=None,
                    help="位姿串: 'x,y,z,roll,pitch,yaw; x,y,z,...'")
    ap.add_argument("--poses-file", default=None,
                    help="位姿文件 (.csv/.txt), 每行 6 个数值")
    ap.add_argument("--unit", default="deg", choices=["deg", "rad"],
                    help="roll/pitch/yaw 输入单位 (默认 deg)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- 采集输入 ----
    if args.poses_file:
        poses_in = load_poses_file(args.poses_file)
        n_lidar = args.n_lidar or len(poses_in)
        presets = parse_presets(args.presets, n_lidar)
        unit_in = args.unit
    elif args.poses:
        poses_in = parse_poses(args.poses)
        n_lidar = args.n_lidar or len(poses_in)
        presets = parse_presets(args.presets, n_lidar)
        unit_in = args.unit
    else:
        poses_in, presets, unit_in = interactive_input()
        n_lidar = len(poses_in)

    # ---- 校验 + 转换 ----
    validate(poses_in, presets, n_lidar)
    poses_rad = normalize_poses(poses_in, unit_in)

    # ---- 写出 ----
    obj = save_lidar_pose_json(args.out, poses_rad, presets, unit_in)

    if not args.quiet:
        print_summary(poses_rad, presets, unit_in)
        print(f"\n已写出: {os.path.abspath(args.out)}")
        print(f"  n_lidar = {obj['n_lidar']}")
        print(f"  presets = {obj['presets']}")
        print("\nMPC 调用示例:")
        print(f"  python -m rslaa.mpc_controller --pose-json \"{args.out}\" \\")
        print(f"      --traj <traj.txt> --env <env.ply> --models <models_dir> \\")
        print(f"      --out pose_seq.json --window-min 5 --metric J_mv")

    return obj


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""可视化: (1) 融合三维场景  (2) 多 LiDAR 仿真点云 + 真值 BBox  (3) 概率占用网格."""
import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

import os

import matplotlib
if os.name != "nt" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")           # 无显示环境 (服务器) 时仅可保存图片
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_VEH_COLORS = [
    [0.90, 0.25, 0.20], [0.20, 0.55, 0.95], [0.95, 0.70, 0.15], [0.35, 0.80, 0.35],
    [0.80, 0.35, 0.85], [0.20, 0.80, 0.80], [0.95, 0.45, 0.60], [0.60, 0.60, 0.20],
    [0.45, 0.30, 0.70], [0.95, 0.55, 0.25],
]
_LIDAR_COLORS = [
    [0.95, 0.20, 0.20], [0.15, 0.45, 0.95], [0.10, 0.80, 0.35], [0.95, 0.75, 0.10],
    [0.85, 0.25, 0.85], [0.10, 0.85, 0.85],
]


def _pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, np.float64))
    return pcd


def _obb_lines(obbs, color=(1, 0, 0)):
    geoms = []
    for obb in obbs:
        eb = np.asarray([
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]], np.int64)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(obb, np.float64))
        ls.lines = o3d.utility.Vector2iVector(eb.astype(np.int32))
        ls.paint_uniform_color(color)
        geoms.append(ls)
    return geoms


def _mesh_of(veh, color):
    # 先在车体局部系旋转, 再平移到车辆位置 (与 raycast 一致; 反序会绕世界原点旋转)
    c, s = np.cos(veh.yaw), np.sin(veh.yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    tv = veh.mesh.tri_verts @ R.T + veh.pos[None, None, :]
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tv.reshape(-1, 3)),
        o3d.utility.Vector3iVector(veh.mesh.tris))
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def _geom_bounds(geoms):
    lo, hi = np.inf * np.ones(3), -np.inf * np.ones(3)
    for g in geoms:
        if hasattr(g, "points") and len(g.points):
            p = np.asarray(g.points)
            lo = np.minimum(lo, p.min(0)); hi = np.maximum(hi, p.max(0))
        if hasattr(g, "vertices") and len(g.vertices):
            p = np.asarray(g.vertices)
            lo = np.minimum(lo, p.min(0)); hi = np.maximum(hi, p.max(0))
    return lo, hi


def _capture(geoms, save: str, width=1600, height=1000):
    """离屏渲染截图. 优先 OffscreenRenderer, 输出空白或异常时回退 legacy."""
    lo, hi = _geom_bounds(geoms)
    ctr = (lo + hi) * 0.5
    extent = float(np.linalg.norm(hi - lo)) + 1e-6
    eye = ctr + extent * np.array([0.55, -0.55, 0.45])
    try:
        import open3d.visualization.rendering as rendering
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        for i, g in enumerate(geoms):
            name = f"g{i}"
            mat = rendering.MaterialRecord()
            mat.point_size = 2.0
            if isinstance(g, o3d.geometry.LineSet):
                mat.shader = "defaultUnlit"
                mat.line_width = 2.0
            elif isinstance(g, o3d.geometry.TriangleMesh):
                mat.shader = "defaultLit"
            else:
                mat.shader = "defaultUnlit"
            renderer.scene.add_geometry(name, g, mat)
        renderer.setup_camera(55.0, ctr.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
        img = None
        for k in range(12):                 # FEngine 异步初始化期间渲染全黑, 多帧等待
            img = np.asarray(renderer.render_to_image())
            # 白色背景 + 非均匀内容才算有效 (初始化竞态会产生黑底/垃圾帧)
            if img.std() > 1.0 and img[:8, :8].mean() > 200:
                break
        ok = (img is not None and img.size and img.std() > 1.0
              and img[:8, :8].mean() > 200)
        del renderer
        if ok:
            o3d.io.write_image(save, o3d.geometry.Image(img))
            return
        print("[viz] OffscreenRenderer 输出空白, 回退 legacy 管线")
    except Exception as e:                                   # noqa: BLE001
        print(f"[viz] OffscreenRenderer 不可用 ({e}), 回退 legacy 管线")
    # ---- legacy 回退: 几何平移到原点 + 针孔相机 (CV 约定) ----
    import copy as _copy
    shifted = []
    for g in geoms:
        g2 = _copy.deepcopy(g)
        g2.translate(-ctr)
        shifted.append(g2)
    eye_o = extent * np.array([0.55, -0.55, 0.45])
    f = -eye_o
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    ext = np.eye(4)
    ext[:3, :3] = np.stack([r, -u, f], 0)
    ext[:3, 3] = -ext[:3, :3] @ eye_o
    fov_y = 55.0
    fy = (height * 0.5) / np.tan(np.radians(fov_y * 0.5))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fy, fy, width * 0.5, height * 0.5)
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = intrinsic
    param.extrinsic = ext
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for g in shifted:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    vis.poll_events(); vis.update_renderer()
    vis.get_view_control().convert_from_pinhole_camera_parameters(
        param, allow_arbitrary=True)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(save, do_render=True)
    vis.destroy_window()


def show_scene(scene, save: str = None, max_env_points: int = 1500000,
               show_vehicles: bool = True, show_bbox: bool = False):
    """(1) 融合三维场景: 环境点云 (灰) + 车辆网格模型 (按类别着色)."""
    if not HAS_O3D:
        raise RuntimeError("需要 open3d 进行三维可视化")
    env = scene.env_viz if scene.env_viz is not None else scene.env
    pts = env.points
    refl = env.reflectance if env.reflectance is not None else np.full(len(pts), 0.4)
    if len(pts) > max_env_points:
        sel = np.random.default_rng(0).choice(len(pts), max_env_points, replace=False)
        pts = pts[sel]
        refl = refl[sel]
    geoms = [_pcd(pts, np.stack([refl * 0.8] * 3, 1))]
    for i, veh in enumerate(scene.vehicles):
        col = _VEH_COLORS[i % len(_VEH_COLORS)]
        if show_vehicles:
            geoms.append(_mesh_of(veh, col))
        if show_bbox:
            geoms += _obb_lines([veh.obb()], (0, 1, 0))
    if save:
        _capture(geoms, save)
        print(f"[viz] 场景截图已保存: {save}")
    else:
        o3d.visualization.draw_geometries(geoms, window_name=f"Scene @ t={scene.t}s")


def show_sim_clouds(scene, clouds, save: str = None, show_env: bool = True,
                    max_env_points: int = 800000, show_bbox: bool = True):
    """(2) 多 LiDAR 仿真点云 (按雷达着色) + 真值 BBox (绿色, 可关闭)."""
    if not HAS_O3D:
        raise RuntimeError("需要 open3d 进行三维可视化")
    geoms = []
    if show_env and scene.env_viz is not None:
        pts = scene.env_viz.points
        if len(pts) > max_env_points:
            sel = np.random.default_rng(0).choice(len(pts), max_env_points, replace=False)
            pts = pts[sel]
        geoms.append(_pcd(pts, np.full((len(pts), 3), 0.55)))
    for i, c in enumerate(clouds):
        col = _LIDAR_COLORS[i % len(_LIDAR_COLORS)]
        if len(c.xyz):
            # 强度调制亮度
            base = np.asarray(col)
            k = (0.35 + 0.65 * c.intensity / 255.0)[:, None]
            geoms.append(_pcd(c.xyz, np.clip(base[None, :] * k, 0, 1)))
        m = o3d.geometry.TriangleMesh.create_sphere(radius=0.3, resolution=12)
        m.translate(c.origin)
        m.paint_uniform_color(col)
        geoms.append(m)
    if show_bbox:
        geoms += _obb_lines([v.obb() for v in scene.vehicles], (0, 1, 0))
    names = ", ".join(c.lidar_name for c in clouds)
    if save:
        _capture(geoms, save)
        print(f"[viz] 仿真点云截图已保存: {save}")
    else:
        o3d.visualization.draw_geometries(geoms, window_name=f"Sim clouds @ t={scene.t}s ({names})")


def save_clouds_npz(clouds, path: str):
    """仿真点云导出 (npz)."""
    out = {}
    for i, c in enumerate(clouds):
        p = f"lidar{i}_"
        out[p + "xyz"] = c.xyz
        out[p + "intensity"] = c.intensity
        out[p + "label"] = c.label
        out[p + "t_beam"] = c.t_beam
        out[p + "origin"] = c.origin
    np.savez_compressed(path, **out)


def save_clouds_mat(clouds, path: str):
    """多雷达仿真点云导出为 MATLAB 可读的 .mat (scipy.io.savemat).

    每个雷达一组变量 lidar{i}_xyz / _intensity / _label / _t_beam / _origin / _name.
    """
    from scipy.io import savemat
    out = {"n_lidars": np.array([[len(clouds)]], np.int32)}
    for i, c in enumerate(clouds):
        p = f"lidar{i}_"
        out[p + "xyz"] = np.asarray(c.xyz, np.float64)
        out[p + "intensity"] = np.asarray(c.intensity, np.float64)
        out[p + "label"] = np.asarray(c.label, np.int64)
        out[p + "t_beam"] = np.asarray(c.t_beam, np.float64)
        out[p + "origin"] = np.asarray(c.origin, np.float64).reshape(1, 3)
        out[p + "name"] = np.asarray(c.lidar_name)
    savemat(path, out, do_compression=True)


def save_clouds_ply(clouds, path: str):
    """多雷达仿真点云导出为二进制 PLY (binary_little_endian).

    逐点属性: x y z (float), intensity (float), label (int32: 0=环境,
    -1=天气杂波, >0=车辆vid), lidar_id (int32), red green blue (uchar,
    按类别着色: 环境灰 / 天气蓝灰 / 车辆按 vid 取色).
    """
    pts_l, inten_l, label_l, lidar_l, rgb_l = [], [], [], [], []
    for i, c in enumerate(clouds):
        n = len(c.xyz)
        if n == 0:
            continue
        pts_l.append(np.asarray(c.xyz, np.float32))
        inten_l.append(np.asarray(c.intensity, np.float32))
        lab = np.asarray(c.label, np.int32)
        label_l.append(lab)
        lidar_l.append(np.full(n, i, np.int32))
        rgb = np.zeros((n, 3), np.uint8)
        rgb[lab == 0] = (150, 150, 150)
        rgb[lab < 0] = (140, 170, 230)               # 天气杂波
        for vid in np.unique(lab[lab > 0]):
            col = _VEH_COLORS[int(vid) % len(_VEH_COLORS)]
            rgb[lab == vid] = (np.asarray(col) * 255).astype(np.uint8)
        rgb_l.append(rgb)
    if not pts_l:
        raise ValueError("点云为空, 无法导出 PLY")
    pts = np.vstack(pts_l)
    inten = np.concatenate(inten_l)
    label = np.concatenate(label_l)
    lidar_id = np.concatenate(lidar_l)
    rgb = np.vstack(rgb_l)
    n = len(pts)
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("intensity", "<f4"), ("label", "<i4"), ("lidar_id", "<i4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["intensity"] = inten
    arr["label"] = label
    arr["lidar_id"] = lidar_id
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment LidarSIM simulation export\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float intensity\n"
        "property int label\n"
        "property int lidar_id\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def show_pog(grid, save: str = None, thr: float = 0.7, slice_axis: int = 2,
             max_scatter: int = 120000, title: str = None):
    """(3) 概率占用网格: 3D 占用体素散点 + 俯视 (最大) 投影."""
    p = grid.probability()
    occ = p > thr
    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ii = np.argwhere(occ)
    if len(ii):
        if len(ii) > max_scatter:
            sel = np.random.default_rng(0).choice(len(ii), max_scatter, replace=False)
            ii = ii[sel]
        ctr = (ii + 0.5) * grid.delta + grid.roi_min
        prob = p[tuple(ii.T)]
        sc = ax.scatter(ctr[:, 0], ctr[:, 1], ctr[:, 2], c=prob,
                        cmap="turbo", s=1.5, vmin=thr, vmax=1.0)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="occupancy P")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(title or f"POG occupied voxels (P>{thr}), {grid.n_frames} frames, Δv={grid.delta} m")
    # 俯视投影
    ax2 = fig.add_subplot(1, 2, 2)
    proj2 = p.max(axis=slice_axis)
    ext = [grid.roi_min[0], grid.roi_max[0], grid.roi_min[1], grid.roi_max[1]]
    if slice_axis == 2:
        im = ax2.imshow(np.flipud(proj2.T), extent=ext, cmap="turbo", vmin=thr, vmax=1)
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    elif slice_axis == 1:
        im = ax2.imshow(proj2.T, extent=[grid.roi_min[0], grid.roi_max[0],
                                         grid.roi_max[2], grid.roi_min[2]],
                        cmap="turbo", vmin=thr, vmax=1)
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("z (m)")
    else:
        im = ax2.imshow(proj2, extent=[grid.roi_min[1], grid.roi_max[1],
                                       grid.roi_max[2], grid.roi_min[2]],
                        cmap="turbo", vmin=thr, vmax=1)
        ax2.set_xlabel("y (m)"); ax2.set_ylabel("z (m)")
    plt.colorbar(im, ax=ax2, label="max occupancy P")
    ax2.set_title("Top-view max projection")
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        plt.close(fig)
        print(f"[viz] POG 图已保存: {save}")
    else:
        plt.show()

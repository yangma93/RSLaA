# -*- coding: utf-8 -*-
"""lidarsim: multi-LiDAR ray-casting simulation for dynamic traffic scenes.

Sub-modules
-----------
config      sensor presets and simulation parameters
trajectory  Vissim / SUMO trajectory parsing
pointcloud  environment point-cloud loading and ground modelling
meshes      vehicle mesh library and model matching
scene       frame-level scene reconstruction (SceneBuilder)
scan        scan-pattern generation
raycast     GPU/CPU ray-casting backend
lidar_sim   multi-LiDAR frame simulator
lshape      detector-agnostic L-shape perception evaluation (IoU / Dev / Comp)
pog         probabilistic occupancy grid
conflict    traffic-conflict analysis
weather     rain / snow clutter and attenuation models
cav         connected-automated-vehicle sampling
viz         Open3D visualisation helpers
opencood_export  OpenCOOD-format export
"""
__version__ = "1.0.0"

from .config import LidarSpec, SimParams, LIDAR_PRESETS
from .trajectory import Trajectories
from .pointcloud import EnvCloud, load_env, GroundModel
from .meshes import Mesh, ModelLibrary
from .scene import SceneBuilder, SceneFrame, VehicleInstance
from .lidar_sim import LidarSimulator
from .pog import ProbabilisticOccupancyGrid

# -*- coding: utf-8 -*-
"""rslaa: adaptive pose configuration for roadside LiDARs.

Modules
-------
common              shared I/O, POG construction, simulator factories
metric_n            the five surrogate metrics (J_EIG, J_cov, J_mv, J_dyn, J_traj)
mpc_controller      receding-horizon (MPC) pose optimisation
marl_env            multi-agent environment used for MAPPO training
marl_trainer        MAPPO trainer (PyTorch)
marl_deploy         deployment of a trained MAPPO policy
auto_evaluate       detector-agnostic IoU / Dev / Comp evaluation
sim_pool            process pool and frame-level caching helpers
make_lidar_pose     initial pose generation and pose-sequence utilities
rslaa_stage1_preprocess   Stage 1: site cleaning and static parameters
rslaa_stage2_experiment   Stage 2: plan generation and pose-sequence experiments
rslaa_stage3_evaluate     Stage 3: high-fidelity replay and metric aggregation
rslaa_plot_*              figure generation utilities
rslaa_table_scene_metrics per-site table export
"""
__version__ = "0.1.0"

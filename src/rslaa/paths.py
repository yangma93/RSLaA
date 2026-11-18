# -*- coding: utf-8 -*-
"""Repository-level path resolution (works from a source checkout)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "mesh_models"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "Results"

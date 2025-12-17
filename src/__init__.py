"""
Audio-Based Navigation using k-Wave

A research-grade implementation for training agents to navigate 2D mazes
using acoustic reverberations simulated with the k-Wave library.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .cave_dataset import (
    MultiCaveDataset,
    ACTION_MAP,
    ACTION_NAMES,
    MIC_OFFSETS,
    compute_class_distribution,
    compute_class_weights,
)
from .lmdb_dataset import (
    LMDBAcousticDataset,
    compute_class_distribution_lmdb,
    compute_class_weights_lmdb,
)
from .models import (
    AcousticCNN1D,
    CompactAcousticNet,
    SpatialTemporalAcousticNet,
    FocalLoss,
)

__all__ = [
    "MultiCaveDataset",
    "LMDBAcousticDataset",
    "ACTION_MAP",
    "ACTION_NAMES",
    "MIC_OFFSETS",
    "compute_class_distribution",
    "compute_class_weights",
    "compute_class_distribution_lmdb",
    "compute_class_weights_lmdb",
    "AcousticCNN1D",
    "CompactAcousticNet",
    "SpatialTemporalAcousticNet",
    "FocalLoss",
]

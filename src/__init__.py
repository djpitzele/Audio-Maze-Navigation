"""
Audio-Based Navigation using k-Wave

A research-grade implementation for training agents to navigate 2D mazes
using acoustic reverberations simulated with the k-Wave library.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .simulation import AcousticSimulator
from .environment import MazeGenerator, Oracle
from .model import AudioNavCNN
from .dataset import AcousticGridDataset

__all__ = [
    "AcousticSimulator",
    "MazeGenerator",
    "Oracle",
    "AudioNavCNN",
    "AcousticGridDataset",
]

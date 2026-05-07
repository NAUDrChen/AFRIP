"""Detection head implementations."""

from .decoupled_head import DecoupledHead
from .dense_detection import DenseDetectionHead

__all__ = ["DecoupledHead", "DenseDetectionHead"]
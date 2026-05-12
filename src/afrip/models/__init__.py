"""Model-domain public entrypoints, registries, and reusable primitives."""

from .registry import (  # noqa: F401
    DETECTORS,
    ASSIGNERS,
    LOSSES,
    POSTPROCESSORS,
    PREPROCESSORS,
    TRACKERS,
    build_detector,
    build_assigner,
    build_loss,
    build_postprocessor,
    build_preprocessor,
    build_tracker,
)
from .detection import DetectionModel, assemble_detection_components  # noqa: F401
from . import detection as _detection  # noqa: F401
from . import tracking as _tracking  # noqa: F401

__all__ = [
    "DETECTORS",
    "ASSIGNERS",
    "LOSSES",
    "PREPROCESSORS",
    "POSTPROCESSORS",
    "TRACKERS",
    "build_detector",
    "build_assigner",
    "build_loss",
    "build_preprocessor",
    "build_postprocessor",
    "build_tracker",
    "DetectionModel",
    "assemble_detection_components",
]

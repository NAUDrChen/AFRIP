"""Detection-domain model components and builders."""

from .backbones import resnet as _resnet  # noqa: F401
from .necks import sppf as _sppf  # noqa: F401
from .necks import detection as _detection_necks  # noqa: F401
from .heads import decoupled_head as _decoupled_head  # noqa: F401
from .heads import dense_detection as _dense_detection  # noqa: F401
from .assigners import yolo_assigner as _yolo_assigner  # noqa: F401
from .assigners import simota_assigner as _simota_assigner  # noqa: F401
from .losses import detection_loss as _detection_loss  # noqa: F401
from .losses import detection_loss_v2 as _detection_loss_v2  # noqa: F401
from .preprocessors import tensor_preprocessor as _tensor_preprocessor  # noqa: F401
from .postprocessors import yolo_postprocessor as _yolo_postprocessor  # noqa: F401
from .detectors.configurable_detection import ConfigurableDetectionModel
from .builder import assemble_detection_components

__all__ = [
    "ConfigurableDetectionModel",
    "assemble_detection_components",
]
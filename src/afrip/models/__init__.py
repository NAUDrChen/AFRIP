"""Detection, tracking and shared model components."""
from .registry import (  # noqa: F401
    BACKBONES, NECKS, HEADS, DETECTORS, MATCHERS, LOSSES,
    build_detector, build_backbone, build_neck, build_head,
    build_matcher, build_loss,
)
from .backbones import resnet         # noqa: F401  触发注册
from .necks import sppf               # noqa: F401
from .heads import decoupled_head     # noqa: F401
from .detectors import yolort         # noqa: F401
from .matchers import yolo_matcher    # noqa: F401
from .losses import detection_loss    # noqa: F401

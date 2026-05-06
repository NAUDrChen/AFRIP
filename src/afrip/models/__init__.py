"""Detection, tracking and shared model components."""
from .common import (  # noqa: F401
    BACKBONES, NECKS, HEADS, DETECTORS, MATCHERS, LOSSES,
    COMMON_BLOCKS,
    build_detector, build_backbone, build_neck, build_head,
    build_matcher, build_loss, build_common_block, normalize_detector_config,
    DetectionBatch, DetectionDetections, DetectionModelOutput, DetectionTarget,
    normalize_detection_batch, normalize_detection_detections,
    normalize_detection_output, normalize_detection_targets,
    assemble_detection_components,
)
from .backbones import resnet         # noqa: F401  触发注册
from .necks import sppf               # noqa: F401
from .heads import decoupled_head     # noqa: F401
from .matchers import yolo_matcher    # noqa: F401
from .matchers import simota_matcher  # noqa: F401
from .losses import detection_loss    # noqa: F401
from .losses import detection_loss_v2 # noqa: F401

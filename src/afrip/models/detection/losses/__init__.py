"""Detection loss implementations."""

from .detection_loss import YoloRTCriterion
from .detection_loss_v2 import YoloRTv2Criterion

__all__ = ["YoloRTCriterion", "YoloRTv2Criterion"]
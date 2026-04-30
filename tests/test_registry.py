from afrip.core import BaseDetector, Registry, build_from_config


def test_registry_build() -> None:
    registry = Registry("demo")

    @registry.register()
    class DemoModule:
        def __init__(self, value: int) -> None:
            self.value = value

    instance = build_from_config({"type": "DemoModule", "value": 7}, registry)
    assert instance.value == 7


def test_base_detector_training_behavior() -> None:
    class DemoDetector(BaseDetector):
        def inference(self, x):
            return x

        def forward(self, x):
            return x

    detector = DemoDetector()
    assert detector.training_behavior_enabled is False

    detector.set_training_behavior(True)
    assert detector.training_behavior_enabled is True

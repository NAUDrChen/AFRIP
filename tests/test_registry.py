from afrip.core import Registry, build_from_config


def test_registry_build() -> None:
    registry = Registry("demo")

    @registry.register()
    class DemoModule:
        def __init__(self, value: int) -> None:
            self.value = value

    instance = build_from_config({"type": "DemoModule", "value": 7}, registry)
    assert instance.value == 7

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Registry:
    name: str
    _items: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            key = name or target.__name__
            if key in self._items:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._items[key] = target
            return target

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"{name} is not registered in {self.name}")
        return self._items[name]

    def list(self) -> list[str]:
        return sorted(self._items.keys())


def build_from_config(config: dict[str, Any], registry: Registry, **extra_kwargs: Any) -> Any:
    if "type" not in config:
        raise KeyError("config must contain a 'type' field")

    target_type = config["type"]
    kwargs = {key: value for key, value in config.items() if key != "type"}
    kwargs.update(extra_kwargs)
    return registry.get(target_type)(**kwargs)

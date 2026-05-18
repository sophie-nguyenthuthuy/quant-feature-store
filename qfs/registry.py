"""FeatureView registry — declare compute functions and versions.

Versions never collide: registering ("rsi", "v1") and ("rsi", "v2") keeps
both queryable. This is intentional. When you change a feature definition
you bump the version; old backtests stay reproducible against v1 while
new research runs against v2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl

ComputeFn = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass
class FeatureView:
    name: str
    version: str
    compute: ComputeFn
    inputs: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.name, self.version)


class Registry:
    def __init__(self) -> None:
        self._views: dict[tuple[str, str], FeatureView] = {}

    def register(self, view: FeatureView) -> FeatureView:
        if view.key in self._views:
            raise ValueError(f"duplicate registration for {view.key}")
        self._views[view.key] = view
        return view

    def get(self, name: str, version: str) -> FeatureView:
        return self._views[(name, version)]

    def all(self) -> list[FeatureView]:
        return list(self._views.values())


registry = Registry()


def feature_view(name: str, version: str, inputs: list[str] | None = None, description: str = ""):
    def decorator(fn: ComputeFn) -> FeatureView:
        view = FeatureView(
            name=name,
            version=version,
            compute=fn,
            inputs=inputs or [],
            description=description,
        )
        registry.register(view)
        return view

    return decorator

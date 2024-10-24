# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional

from ..component import Component

__all__ = ["MapComponents"]


class MapComponents(Component):
    component: Component | Sequence[Component]

    def __init__(self, component: Component | Sequence[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.component = component

    def call(
        self,
        inputs: tuple,
        /,
        tracing_level: int = 0,
        **kwargs
    ) -> tuple:
        if isinstance(self.component, Sequence):
            return tuple(
                component(
                    input,
                    tracing_level=tracing_level + 1,
                    **kwargs
                )
                for component, input in zip(self.component, inputs, strict=True)
            )
        else:
            return tuple(
                self.component(
                    input,
                    tracing_level=tracing_level + 1,
                    **kwargs
                )
                for input in inputs
            )

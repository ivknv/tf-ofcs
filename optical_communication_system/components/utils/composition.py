# -*- coding: utf-8 -*-

from typing import Any, Optional
from ..component import Component

__all__ = ["Composition"]


class Composition(Component):
    #: System components
    components: list[Component]

    def __init__(self, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    def call(self, inp: Any, /, tracing_level: int = 0, **kwargs) -> Any:
        result: Any = inp

        for component in self.components:
            result = component(result, tracing_level=tracing_level + 1, **kwargs)

        return result

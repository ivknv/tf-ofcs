# -*- coding: utf-8 -*-

from collections.abc import Callable
from typing import Any, Optional
from ..component import Component

__all__ = ["Lambda"]


class Lambda(Component):
    #: `Callable` that will be invoked
    function: Callable

    #: If `True`, keyword arguments will be passed to `function`
    allow_kwargs: bool

    def __init__(
        self,
        function: Callable,
        allow_kwargs: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.function = function
        self.allow_kwargs = allow_kwargs

    def call(self, inp: Any, /, **kwargs) -> Any:
        if self.allow_kwargs:
            return self.function(inp, **kwargs)

        return self.function(inp)

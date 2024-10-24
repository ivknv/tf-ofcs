# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Any, Optional
from ..component import Component

__all__ = ["Fork"]


class Fork(Component):
    shape: Sequence[int]

    def __init__(self, shape: Sequence[int], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.shape = shape

    def call(self, inp: Any, /, **kwargs) -> tuple:
        result = inp

        for i in self.shape:
            result = (result,) * i

        return result

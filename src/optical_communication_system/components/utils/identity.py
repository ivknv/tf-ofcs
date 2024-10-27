# -*- coding: utf-8 -*-

from typing import Any
from ..component import Component

__all__ = ["Identity"]


class Identity(Component):
    def call(self, inp: Any, /, **kwargs) -> Any:
        return inp

# -*- coding: utf-8 -*-

from collections.abc import Callable
from typing import Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .components import Signal

__all__ = [
    "Component",
    "SignalType",
    "FilterType"
]

Component = Callable[["Signal"], "Signal"]
SignalType = Union[Literal["digital"], Literal["electrical"], Literal["optical"]]
FilterType = Union[Literal["butterworth"], Literal["bessel"], Literal["chebyshev"]]

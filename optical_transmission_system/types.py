# -*- coding: utf-8 -*-

from typing import Literal, Union

__all__ = ["SignalType"]

SignalType = Union[Literal["digital"], Literal["electrical"], Literal["optical"]]

# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt
from typing import Literal, Union

__all__ = ["BitSequence", "SignalType"]

#: Sequence of bits (0 or 1)
BitSequence = npt.NDArray[np.bool]

#: Signal type
SignalType = Union[Literal["electrical"], Literal["optical"]]

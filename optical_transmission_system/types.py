# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt
from typing import Literal, Union

__all__ = ["BitSequence", "SignalType"]

BitSequence = npt.NDArray[np.bool]
SignalType = Union[Literal["electrical"], Literal["optical"]]

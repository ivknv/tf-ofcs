# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from ...datatypes import ElectricalSignal, OpticalSignal
from ..component import Component

__all__ = ["SplitSignals"]


class SplitSignals(Component):
    """
        Splits signal with shape (batch_size, parameter_dims, sample_count)
        into `n` tuples of shape (batch_size // n, parameter_dims, sample_count).

    """

    n: int

    def __init__(self, n: int = 0, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = n

    def call(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        **kwargs
    ) -> tuple[ElectricalSignal, ...] | tuple[OpticalSignal, ...]:
        Signal = signal.__class__

        batch_size = tf.shape(signal.samples)[0] // self.n

        return tuple(
            Signal(
                samples=signal.samples[i * batch_size:(i + 1) * batch_size],
                timing=signal.timing
            )
            for i in range(self.n)
        )

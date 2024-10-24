# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from ..datatypes import ElectricalSignal, OpticalSignal
from ..component import Component

__all__ = ["GatherSignals"]


class GatherSignals(Component):
    """
        Applies `tf.gather()` operation to the input signal, except only for the first
        two dimensions (`batch_size` and `parameter_dims`).
    """

    indices: tf.Variable  # shape=[None]; dtype=tf.int32

    def __init__(self, indices: Sequence[int] | tf.Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.indices = tf.Variable(tf.constant(indices))
        self.local_parameters = ["indices"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        indices: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        return Signal(samples=tf.gather(signal.samples, indices), timing=signal.timing)

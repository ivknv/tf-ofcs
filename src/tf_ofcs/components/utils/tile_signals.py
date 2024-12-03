# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from ...datatypes import ElectricalSignal, OpticalSignal
from ..component import Component

__all__ = ["TileSignals"]


class TileSignals(Component):
    """
        Applies `tf.tile()` operation to the input signal, but only the first
        two dimensions (`batch_size` and `parameter_dims`).
    """

    multiples: tf.Variable  # shape=[2]; dtype=tf.int32

    def __init__(self, multiples: tuple[int, int] | tf.Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.multiples = tf.Variable(tf.constant(multiples), name=f"{self.name}__multiples")
        self.local_parameters = ["multiples"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        multiples: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        return Signal(
            samples=tf.tile(signal.samples, tf.concat([multiples, [1]], 0)),
            timing=signal.timing
        )

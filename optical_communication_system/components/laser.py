# -*- coding: utf-8 -*-

import tensorflow as tf

from typing import Optional
from .datatypes import Timing, timing_spec, OpticalSignal

from .component import Component

__all__ = ["Laser"]


class Laser(Component):
    #: Output power in watts
    power: tf.Variable  # shape=[]; dtype=float32

    #: The output signal's shape is `(batch_size, tf.size(timing))`
    batch_size: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        *,
        power: float | tf.Tensor,
        batch_size: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.batch_size = tf.Variable(tf.constant(batch_size))
        self.parameters = ["power", "batch_size"]

    @tf.function(input_signature=(
        timing_spec,
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32)
    ))
    def implementation(self, timing: Timing, power: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        samples = tf.fill(
            [batch_size, tf.size(timing.time)],
            tf.cast(tf.math.sqrt(power), tf.complex64)
        )

        return OpticalSignal(timing=timing, samples=samples)

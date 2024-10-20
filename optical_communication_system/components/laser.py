# -*- coding: utf-8 -*-

from collections.abc import Sequence
import typing
import tensorflow as tf

from typing import Optional
from .datatypes import Timing, timing_spec, OpticalSignal

from .component import Component

__all__ = ["Laser"]


class Laser(Component):
    #: Output power in watts
    power: tf.Variable  # shape=[] or [None]; dtype=float32

    #: The output signal's shape is `(batch_size, parameter_dims, tf.size(time))`
    batch_size: tf.Variable # shape=[]; dtype=int32

    def __init__(
        self,
        power: float | Sequence[float] | tf.Tensor,
        batch_size: int | tf.Tensor = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.batch_size = tf.Variable(tf.constant(batch_size))
        self.parameters = ["power", "batch_size"]

    @tf.function(input_signature=(
        timing_spec,
        tf.TensorSpec(shape=typing.cast(tf.TensorShape, None), dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ))
    def implementation(
        self,
        timing: Timing,
        power: tf.Tensor,
        batch_size: tf.Tensor
    ) -> tf.Tensor:
        if tf.rank(power) == 0:
            power = tf.expand_dims(power, 0)

        result_shape = [batch_size, tf.size(power), timing.sample_count]

        samples = tf.reshape(
            tf.tile(
                tf.reshape(
                    tf.repeat(
                        tf.cast(tf.math.sqrt(power), tf.complex64),
                        timing.sample_count
                    ),
                    [tf.size(power), timing.sample_count]
                ),
                [batch_size, 1]
            ),
            result_shape
        )

        return OpticalSignal(timing=timing, samples=samples)

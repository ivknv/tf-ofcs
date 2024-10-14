# -*- coding: utf-8 -*-

from collections.abc import Sequence
import typing
import tensorflow as tf

from typing import Optional
from .datatypes import Timing, timing_spec, OpticalSignal

from .component import Component
from .utils import combine_shape

__all__ = ["Laser"]


class Laser(Component):
    #: Output power in watts
    power: tf.Variable  # shape=None; dtype=float32

    #: The output signal's shape (not taking into account the shape derived from parameters)
    batch_shape: tf.Variable  # shape=[None]; dtype=int32

    def __init__(
        self,
        *,
        power: float | tf.Tensor,
        batch_shape: Sequence[int] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.batch_shape = tf.Variable(tf.constant(batch_shape))
        self.parameters = ["power", "batch_shape"]

    @tf.function(input_signature=(
        timing_spec,
        tf.TensorSpec(shape=typing.cast(tf.TensorShape, None), dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ))
    def implementation(self, timing: Timing, power: tf.Tensor, batch_shape: tf.Tensor) -> tf.Tensor:
        result_shape = combine_shape([self.batch_shape, tf.shape(power), [tf.size(timing.time)]])

        repeated = tf.repeat(tf.cast(tf.math.sqrt(power), tf.complex64), tf.size(timing.time))
        samples = tf.reshape(
            tf.tile(repeated, tf.concat([[tf.math.reduce_prod(batch_shape)], tf.shape(repeated)[1:]], 0)),
            result_shape
        )

        return OpticalSignal(timing=timing, samples=samples)

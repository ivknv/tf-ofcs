# -*- coding: utf-8 -*-

from collections.abc import Sequence
import tensorflow as tf

from typing import Optional
from .datatypes import ElectricalSignal, Timing, OpticalSignal

from .component import Component

__all__ = ["Laser"]


class Laser(Component):
    #: Output power in watts
    power: tf.Variable  # shape=[] or [None]; dtype=float32

    #: The output signal's shape is `(batch_size, parameter_dims, tf.size(time))`
    batch_size: tf.Variable  # shape=[]; dtype=int32

    #: The output signal's shape is `(batch_size, parameter_dims, tf.size(time))`
    parameter_dims: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        power: float | Sequence[float] | tf.Tensor,
        batch_size: int | tf.Tensor = 1,
        parameter_dims: Optional[int | tf.Tensor] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.batch_size = tf.Variable(tf.constant(batch_size))

        if parameter_dims is None:
            self.parameter_dims = tf.Variable(tf.size(self.power))
        else:
            self.parameter_dims = tf.Variable(tf.constant(parameter_dims))

        self.global_parameters = ["batch_size", "parameter_dims"]
        self.local_parameters = [("power", 0), "batch_size", "parameter_dims"]

    def implementation(
        self,
        timing_or_signal: Timing | ElectricalSignal | OpticalSignal,
        /,
        power: tf.Tensor,
        batch_size: tf.Tensor,
        parameter_dims: tf.Tensor
    ) -> OpticalSignal:
        if isinstance(timing_or_signal, (ElectricalSignal, OpticalSignal)):
            timing = timing_or_signal.timing
        else:
            timing = timing_or_signal

        result_shape = [batch_size, parameter_dims, timing.sample_count]

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

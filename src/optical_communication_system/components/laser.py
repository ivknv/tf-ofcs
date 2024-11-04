# -*- coding: utf-8 -*-

from collections.abc import Sequence
import tensorflow as tf

from typing import Optional
from ..datatypes import ElectricalSignal, Timing, OpticalSignal

from .component import Component
from ..utils import timing_to_samples
from .. import constants

__all__ = ["Laser"]


class Laser(Component):
    """Outputs a monochromatic optical signal."""

    #: Output power in watts
    power: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Laser frequency offset (relative to the central frequency) in Hz
    frequency: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: The output signal's shape is `(batch_size, parameter_dims, tf.size(time))`
    batch_size: tf.Variable  # shape=[]; dtype=int32

    #: The output signal's shape is `(batch_size, parameter_dims, tf.size(time))`
    parameter_dims: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        power: float | Sequence[float] | tf.Tensor,
        frequency: float | Sequence[float] | tf.Tensor = 0.0,
        batch_size: int | tf.Tensor = 1,
        parameter_dims: Optional[int | tf.Tensor] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.frequency = tf.Variable(tf.constant(frequency))

        self.batch_size = tf.Variable(tf.constant(batch_size))

        if parameter_dims is None:
            self.parameter_dims = tf.Variable(tf.size(self.power))
        else:
            self.parameter_dims = tf.Variable(tf.constant(parameter_dims))

        self.global_parameters = ["batch_size", "parameter_dims"]
        self.local_parameters = [
            ("power", 0),
            ("frequency", 0),
            "batch_size",
            "parameter_dims"
        ]

    def implementation(
        self,
        timing_or_signal: Timing | ElectricalSignal | OpticalSignal,
        /,
        power: tf.Tensor,
        frequency: tf.Tensor,
        batch_size: tf.Tensor,
        parameter_dims: tf.Tensor
    ) -> OpticalSignal:
        if isinstance(timing_or_signal, (ElectricalSignal, OpticalSignal)):
            timing = timing_or_signal.timing
        else:
            timing = timing_or_signal

        result_shape = [batch_size, parameter_dims, timing.sample_count]
        time = timing_to_samples(timing)

        power = tf.expand_dims(power, -1)  # shape=[parameter_dims, 1]
        frequency = tf.expand_dims(frequency, -1)  # shape=[parameter_dims, 1]

        samples = tf.math.exp(
            tf.complex(
                tf.math.log(tf.math.sqrt(power)),
                2.0 * constants.pi * frequency * time
            )
        )

        samples = tf.reshape(
            tf.tile(
                tf.reshape(samples, [parameter_dims, timing.sample_count]),
                [batch_size, 1]
            ),
            result_shape
        )

        return OpticalSignal(timing=timing, samples=samples)

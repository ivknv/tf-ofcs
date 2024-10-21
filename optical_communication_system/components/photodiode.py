# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf
from .datatypes import ElectricalSignal, OpticalSignal, optical_signal_spec
from .component import Component

__all__ = ["Photodiode"]


class Photodiode(Component):
    #: Minimum power (in watts) that can be converted into a current
    sensitivity: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Maximum power the diode can convert (anything above will be clipped)
    maximum_power: tf.Variable  # shape=[] or [None]; dtype=float32

    def __init__(
        self,
        sensitivity: float | Sequence[float] | tf.Tensor = 1e-5,
        maximum_power: float | Sequence[float] | tf.Tensor = 2e-3,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sensitivity = tf.Variable(tf.constant(sensitivity))
        self.maximum_power = tf.Variable(tf.constant(maximum_power))

        self.local_parameters = [("sensitivity", 0), ("maximum_power", 0)]

    @tf.function(input_signature=(
        optical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ))
    def implementation(
        self,
        signal: OpticalSignal,
        sensitivity: tf.Tensor,
        maximum_power: tf.Tensor
    ) -> ElectricalSignal:
        sensitivity = tf.expand_dims(sensitivity, -1)
        maximum_power = tf.expand_dims(maximum_power, -1)

        samples = tf.cast(
            tf.clip_by_value(
                tf.abs(signal.samples)**2,
                sensitivity,
                maximum_power
            ),
            tf.float32
        )

        return ElectricalSignal(
            timing=signal.timing,
            samples=samples
        )

# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from .datatypes import ElectricalSignal, OpticalSignal, optical_signal_spec
from .component import Component

__all__ = ["Photodiode"]


class Photodiode(Component):
    #: Minimum power (in watts) that can be converted into a current
    sensitivity: tf.Variable  # shape=(1,); dtype=float32

    #: Maximum power the diode can convert (anything above will be clipped)
    maximum_power: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        sensitivity: float = 1e-5,
        maximum_power: float = 2e-3,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sensitivity = tf.Variable(tf.constant(sensitivity))
        self.maximum_power = tf.Variable(tf.constant(maximum_power))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> ElectricalSignal:
        samples = tf.cast(
            tf.clip_by_value(
                tf.reduce_sum(tf.abs(signal.samples)**2, axis=1),
                self.sensitivity,
                self.maximum_power
            ),
            tf.float32
        )

        return ElectricalSignal(
            timing=signal.timing,
            samples=samples
        )

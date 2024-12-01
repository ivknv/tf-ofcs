# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from .component import Component
from ..datatypes import OpticalSignal, optical_signal_spec

__all__ = ["OpticalGain"]

class OpticalGain(Component):
    #: Gain coefficient (unitless)
    gain: tf.Variable  # shape=[] or [None]; dtype=float32

    def __init__(self, gain: float | Sequence[float] | tf.Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.gain = tf.Variable(tf.constant(gain), name=f"{self.name}__gain")
        self.local_parameters = [("gain", 0)]

    @tf.function(input_signature=(
        optical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ))
    def implementation(self, input_signal: OpticalSignal, gain: tf.Tensor) -> OpticalSignal:
        gain = tf.expand_dims(gain, -1)

        return OpticalSignal(
            timing=input_signal.timing,
            samples=input_signal.samples * tf.cast(gain ** 0.5, tf.complex64)
        )

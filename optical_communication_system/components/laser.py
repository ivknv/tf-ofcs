# -*- coding: utf-8 -*-

import tensorflow as tf

from typing import Optional
from .datatypes import Timing, timing_spec, OpticalSignal

from .utils import fftfreq, gaussian_distribution
from .component import Component

__all__ = ["Laser"]


class Laser(Component):
    #: Output power in watts
    power: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        *,
        power: float = 2e-3,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))

    @tf.function(input_signature=(timing_spec,))
    def __call__(self, timing: Timing) -> OpticalSignal:
        samples = tf.fill(tf.shape(timing.time), tf.cast(tf.math.sqrt(self.power), tf.complex64))

        return OpticalSignal(timing=timing, samples=samples)

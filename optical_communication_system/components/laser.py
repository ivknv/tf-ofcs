# -*- coding: utf-8 -*-

import tensorflow as tf

from typing import Optional
from .datatypes import Timing, timing_spec, OpticalSignal

from .utils import gaussian_distribution

__all__ = ["Laser"]


class Laser(tf.Module):
    #: Output power in watts
    power: tf.Variable  # shape=(1,); dtype=float32

    #: Central frequency of the of the emitted light in Hz
    central_frequency: tf.Variable  # shape=(1,); dtype=float32

    #: Line width of the signal, full width at half-maximum (FWHM), Hz
    linewidth: tf.Variable  # shape=(1,); dtype=float32

    #: Number of frequency components in the signal
    frequency_count: tf.Variable  # shape=(1,); dtype=int32

    def __init__(
        self,
        *,
        power: float = 2e-3,
        central_frequency: float = 193.414e12,
        linewidth: float = 1e6,
        frequency_count: int = 128,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.central_frequency = tf.Variable(tf.constant(central_frequency))
        self.linewidth = tf.Variable(tf.constant(linewidth))
        self.frequency_count = tf.Variable(tf.constant(frequency_count))

    @tf.function(input_signature=(timing_spec,))
    def __call__(self, timing: Timing) -> OpticalSignal:
        gauss_variance = 1.0 / (2.0 * tf.math.log(2.0)) ** 0.5 * self.linewidth

        frequencies = tf.linspace(
            self.central_frequency - tf.multiply(self.linewidth, 10.0),
            self.central_frequency + tf.multiply(self.linewidth, 10.0),
            self.frequency_count
        )

        distribution = gaussian_distribution(
            frequencies,
            self.central_frequency,
            gauss_variance
        )
        distribution /= tf.reduce_sum(distribution)

        samples = tf.reshape(
            tf.tile(
                tf.cast(tf.math.sqrt(self.power) * distribution, tf.complex64),
                [tf.size(timing.time)],
            ),
            [tf.size(timing.time), self.frequency_count]
        )

        return OpticalSignal(
            timing=timing,
            frequencies=frequencies,
            samples=samples,
            central_frequency=self.central_frequency,
            power=self.power
        )

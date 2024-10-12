# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from .utils import fftfreq
from .component import Component

from .datatypes import Timing, timing_spec, ElectricalSignal, OpticalSignal

__all__ = ["BandPassFilter"]


class BandPassFilter(Component):
    #: Minimum frequency in Hz
    min_frequency: tf.Variable  # shape=(); dtype=float32

    #: Maximum frequency in Hz
    max_frequency: tf.Variable  # shape=(); dtype=float32

    def __init__(self, min_frequency: float, max_frequency: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.min_frequency = tf.Variable(tf.constant(min_frequency))
        self.max_frequency = tf.Variable(tf.constant(max_frequency))
        self.parameters = ["min_frequency", "max_frequency"]

    @tf.function
    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        SignalClass = signal.__class__

        filtered_samples = self._implementation(signal.timing, signal.samples, min_frequency, max_frequency)

        if isinstance(signal, ElectricalSignal):
            filtered_samples = tf.cast(tf.math.real(filtered_samples), tf.float32)

        return SignalClass(timing=signal.timing, samples=filtered_samples)

    @tf.function(input_signature=(
        timing_spec,
        tf.TensorSpec([None, None]),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32)
    ))
    def _implementation(
        self,
        timing: Timing,
        samples: tf.Tensor,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> tf.Tensor:
        fft = tf.signal.fft(tf.cast(samples, tf.complex64))

        freqs = tf.abs(fftfreq(tf.shape(fft)[1], 1.0 / timing.sampling_rate))
        pass_mask = tf.logical_and(tf.math.greater_equal(freqs, min_frequency), tf.less_equal(freqs, max_frequency))
        filtered_fft = tf.where(pass_mask, fft, tf.complex(0.0, 0.0))

        filtered_samples = tf.signal.ifft(filtered_fft)

        return filtered_samples

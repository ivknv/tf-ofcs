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

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[2], dtype=tf.complex64),
        tf.TensorSpec([2], tf.float32),
    ))
    def _transfer_function(self, fft_and_freq: tf.Tensor, freq_range: tf.Tensor) -> tf.Tensor:
        fft = fft_and_freq[0]
        freq = tf.abs(fft_and_freq[1])

        min_frequency = freq_range[0]
        max_frequency = freq_range[1]

        if tf.greater(freq, max_frequency) or tf.less(freq, min_frequency):  # type: ignore
            return tf.complex(0.0, 0.0)
        else:
            return fft

    @tf.function
    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        SignalClass = signal.__class__
        return SignalClass(
            timing=signal.timing,
            samples=self._implementation(signal.timing, signal.samples, min_frequency, max_frequency)
        )

    @tf.function(input_signature=(
        timing_spec, tf.TensorSpec([None]), tf.TensorSpec([], tf.float32), tf.TensorSpec([], tf.float32)
    ))
    def _implementation(
        self,
        timing: Timing,
        samples: tf.Tensor,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> tf.Tensor:
        fft = tf.signal.fft(tf.cast(samples, tf.complex64))
        freqs = tf.cast(
            fftfreq(tf.size(fft), 1.0 / timing.sampling_rate),
            tf.complex64
        )

        filtered_fft = tf.vectorized_map(
            fn=lambda fft_and_freq: self._transfer_function(fft_and_freq, [min_frequency, max_frequency]),
            elems=tf.transpose(tf.stack([fft, freqs]))
        )

        filtered_samples = tf.cast(tf.math.real(tf.signal.ifft(filtered_fft)), tf.float32)

        return filtered_samples

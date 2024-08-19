# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from .utils import fftfreq

from .datatypes import (
    ElectricalSignal, electrical_signal_spec
)

__all__ = ["ElectricalLowPassFilter"]


class ElectricalLowPassFilter(tf.Module):
    #: Filter cutoff frequency in Hz
    cutoff_frequency: tf.Variable  # shape=(1,); dtype=float32

    def __init__(self, cutoff_frequency: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.cutoff_frequency = tf.Variable(tf.constant(cutoff_frequency))

    @tf.function(input_signature=(tf.TensorSpec(shape=[2], dtype=tf.complex64),))
    def _transfer_function(self, fft_and_freq: tf.Tensor) -> tf.Tensor:
        fft = fft_and_freq[0]
        freq = tf.cast(tf.abs(fft_and_freq[1]), tf.float32)

        if tf.greater(freq, self.cutoff_frequency):  # type: ignore
            return tf.cast(tf.constant(0.0), tf.complex64)
        else:
            return fft

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, signal: ElectricalSignal) -> ElectricalSignal:
        fft = tf.signal.fft(tf.cast(signal.samples, tf.complex64))
        freqs = fftfreq(tf.size(fft), 1.0 / signal.timing.sampling_rate)

        filtered_fft = tf.vectorized_map(
            fn=self._transfer_function,
            elems=tf.transpose(tf.stack([fft, tf.cast(freqs, tf.complex64)]))
        )

        filtered_samples = tf.signal.ifft(filtered_fft)

        return ElectricalSignal(
            timing=signal.timing,
            samples=filtered_samples
        )

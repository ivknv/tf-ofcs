# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from .utils import fftfreq
from .component import Component

from .datatypes import (
    ElectricalSignal, electrical_signal_spec
)

__all__ = ["ElectricalLowPassFilter"]


class ElectricalLowPassFilter(Component):
    #: Filter cutoff frequency in Hz
    cutoff_frequency: tf.Variable  # shape=(); dtype=float32

    def __init__(self, cutoff_frequency: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.cutoff_frequency = tf.Variable(tf.constant(cutoff_frequency))
        self.parameters = ["cutoff_frequency"]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[2], dtype=tf.complex64),
        tf.TensorSpec([], tf.float32)
    ))
    def _transfer_function(self, fft_and_freq: tf.Tensor, cutoff_frequency: tf.Tensor) -> tf.Tensor:
        fft = fft_and_freq[0]
        freq = tf.abs(fft_and_freq[1])

        if tf.greater(freq, cutoff_frequency):  # type: ignore
            return tf.complex(0.0, 0.0)
        else:
            return fft

    @tf.function(input_signature=(electrical_signal_spec, tf.TensorSpec([], tf.float32)))
    def implementation(self, signal: ElectricalSignal, cutoff_frequency: tf.Tensor) -> ElectricalSignal:
        fft = tf.signal.fft(tf.cast(signal.samples, tf.complex64))
        freqs = tf.cast(
            fftfreq(tf.size(fft), 1.0 / signal.timing.sampling_rate),
            tf.complex64
        )

        filtered_fft = tf.vectorized_map(
            fn=lambda fft_and_freq: self._transfer_function(fft_and_freq, cutoff_frequency),
            elems=tf.transpose(tf.stack([fft, freqs]))
        )

        filtered_samples = tf.cast(tf.math.real(tf.signal.ifft(filtered_fft)), tf.float32)

        return ElectricalSignal(
            timing=signal.timing,
            samples=filtered_samples
        )

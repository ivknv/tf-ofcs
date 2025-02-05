# -*- coding: utf-8 -*-
# Copyright (C) 2024 Ivan Konovalov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Sequence
from typing import Optional, Protocol
import tensorflow as tf

from ..utils import fftfreq
from .component import Component

from ..datatypes import Timing, timing_spec, ElectricalSignal, OpticalSignal

__all__ = ["BandPassFilter", "FFTFilter"]


class _FrequencyResponseFunction(Protocol):
    def __call__(self, frequencies: tf.Tensor, parameter_dims: tf.Tensor, **kwargs) -> tf.Tensor:
        ...


class FFTFilter(Component):
    """
        This component performs a Fourier transform, multiplies the signal
        by the frequency response defined by a function, and then performs
        the inverse Fourier transform.
    """

    #: Function that defines the filter's frequency response
    #: Signature: (frequencies: tf.Tensor, parameter_dims: tf.Tensor, **kwargs) -> tf.Tensor.
    #: Must accept a tensor of frequencies (in Hz) and output a tensor with
    #: shape=[parameter_dims, n], dtype=tf.complex64, where n is the number of
    #: frequencies. `**kwargs` can be used to pass arbitrary additional arguments
    #: from components
    frequency_response_function: _FrequencyResponseFunction

    def __init__(
        self,
        frequency_response_function: _FrequencyResponseFunction,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.frequency_response_function = frequency_response_function

    def implementation(self, signal: ElectricalSignal | OpticalSignal, /) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        parameter_dims = tf.shape(signal.samples)[1]

        fft = tf.signal.fft(tf.cast(signal.samples, tf.complex64))
        freqs = tf.abs(fftfreq(tf.shape(fft)[-1], 1.0 / signal.timing.sampling_rate))

        frequency_response = self.frequency_response_function(freqs, parameter_dims)
        filtered_samples = tf.signal.ifft(frequency_response * fft)

        if isinstance(signal, ElectricalSignal):
            filtered_samples = tf.cast(tf.math.real(filtered_samples), tf.float32)

        return Signal(timing=signal.timing, samples=filtered_samples)


class BandPassFilter(Component):
    #: Minimum frequency in Hz
    min_frequency: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Maximum frequency in Hz
    max_frequency: tf.Variable  # shape=[] or [None]; dtype=float32

    def __init__(
        self,
        min_frequency: float | Sequence[float] | tf.Tensor,
        max_frequency: float | Sequence[float] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.min_frequency = tf.Variable(tf.constant(min_frequency), name=f"{self.name}__min_frequency")
        self.max_frequency = tf.Variable(tf.constant(max_frequency), name=f"{self.name}__max_frequency")
        self.local_parameters = [("min_frequency", 0), ("max_frequency", 0)]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
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
        tf.TensorSpec([None, None, None]),
        tf.TensorSpec([None], tf.float32),
        tf.TensorSpec([None], tf.float32)
    ))
    def _implementation(
        self,
        timing: Timing,
        samples: tf.Tensor,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> tf.Tensor:
        fft = tf.signal.fft(tf.cast(samples, tf.complex64))

        min_frequency = tf.expand_dims(min_frequency, -1)
        max_frequency = tf.expand_dims(max_frequency, -1)

        freqs = tf.abs(fftfreq(tf.shape(fft)[-1], 1.0 / timing.sampling_rate))
        pass_mask = tf.logical_and(tf.math.greater_equal(freqs, min_frequency), tf.less_equal(freqs, max_frequency))
        filtered_fft = tf.where(pass_mask, fft, tf.complex(0.0, 0.0))

        filtered_samples = tf.signal.ifft(filtered_fft)

        return filtered_samples

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
from typing import Optional
import tensorflow as tf

from ..utils import fftfreq
from .component import Component

from ..datatypes import (
    Timing, timing_spec, ElectricalSignal, OpticalSignal, FrequencyResponseFunction
)

__all__ = [
    "BandPassFilter",
    "FFTFilter",
    "BrickwallFrequencyResponse",
    "WDMBrickwallFrequencyResponse",
    "GaussianFrequencyResponse"
]


class FFTFilter(Component):
    """
        This component performs a Fourier transform, multiplies the signal
        by the frequency response defined by a function, and then performs
        the inverse Fourier transform.
    """

    #: Function that defines the filter's frequency response
    #: See :any:`FrequencyResponseFunction` for more info
    frequency_response_function: FrequencyResponseFunction

    def __init__(
        self,
        frequency_response_function: FrequencyResponseFunction,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.frequency_response_function = frequency_response_function

    def call(self, signal: ElectricalSignal | OpticalSignal, /, **kwargs) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        fft = tf.signal.fft(tf.cast(signal.samples, tf.complex64))
        n = tf.shape(fft)[2]
        freqs = tf.abs(fftfreq(n, 1.0 / signal.timing.sampling_rate))

        # We need override these parameters, so we drop them from kwargs
        kwargs.pop("batch_size", None)
        kwargs.pop("parameter_dims", None)

        frequency_response = self.frequency_response_function(
            freqs,
            batch_size=tf.constant(1),
            parameter_dims=tf.shape(fft)[1],
            **kwargs
        )

        filtered_samples = tf.signal.ifft(frequency_response * fft)

        if isinstance(signal, ElectricalSignal):
            filtered_samples = tf.cast(tf.math.real(filtered_samples), tf.float32)

        return Signal(timing=signal.timing, samples=filtered_samples)


class BandPassFilter(Component):
    #: Minimum frequency in Hz
    min_frequency: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: Maximum frequency in Hz
    max_frequency: tf.Variable  # shape=[] or [None]; dtype=tf.float32

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


class BrickwallFrequencyResponse(Component):
    """
        Implements brickwall bandpass filter's frequency response function.
        Passes only the frequencies between `min_frequency` and `max_frequency`.
    """

    #: Minimum frequency in Hz
    min_frequency: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: Maximum frequency in Hz
    max_frequency: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    batch_size: tf.Variable  # shape=[]; dtype=tf.int32
    parameter_dims: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        min_frequency: float | Sequence[float] | tf.Tensor,
        max_frequency: float | Sequence[float] | tf.Tensor,
        batch_size: float | tf.Tensor = 1,
        parameter_dims: float | tf.Tensor = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.min_frequency = tf.Variable(tf.constant(min_frequency), name=f"{self.name}__min_frequency")
        self.max_frequency = tf.Variable(tf.constant(max_frequency), name=f"{self.name}__max_frequency")
        self.batch_size = tf.Variable(tf.constant(batch_size), name=f"{self.name}__batch_size")
        self.parameter_dims = tf.Variable(tf.constant(parameter_dims), name=f"{self.name}__parameter_dims")

        self.local_parameters = [
            ("min_frequency", 0),
            ("max_frequency", 0),
            "batch_size",
            "parameter_dims"
        ]

        self.global_parameters = ["batch_size", "parameter_dims"]

    def implementation(
        self,
        frequencies: tf.Tensor,
        /,
        batch_size: tf.Tensor,
        parameter_dims: tf.Tensor,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> tf.Tensor:
        min_frequency = tf.expand_dims(min_frequency, -1)
        max_frequency = tf.expand_dims(max_frequency, -1)

        return tf.cast(
            tf.tile(
                tf.expand_dims(
                    tf.math.logical_and(
                        tf.math.greater_equal(frequencies, min_frequency),
                        tf.math.less_equal(frequencies, max_frequency)
                    ),
                    0
                ),
                [batch_size, parameter_dims, 1]
            ),
            tf.complex64
        )


class WDMBrickwallFrequencyResponse(Component):
    """
        Similar to :any:`BrickwallFrequencyResponse`, but one passband per WDM
        channel. This can be used for an optical demultiplexer.
    """

    #: WDM channel frequencies in Hz
    channel_frequencies: tf.Variable  # shape=[None]; dtype=tf.float32

    #: WDM channel width in Hz
    channel_width: tf.Variable  # shape=[]; dtype=tf.float32

    batch_size: tf.Variable  # shape=[]; dtype=tf.int32
    parameter_dims: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        channel_frequencies: Sequence[float] | tf.Tensor,
        channel_width: float | tf.Tensor,
        batch_size: float | tf.Tensor = 1,
        parameter_dims: float | tf.Tensor = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.channel_frequencies = tf.Variable(
            tf.constant(channel_frequencies),
            name=f"{self.name}__channel_frequencies"
        )

        self.channel_width = tf.Variable(
            tf.constant(channel_width),
            name=f"{self.name}__channel_width"
        )

        self.batch_size = tf.Variable(tf.constant(batch_size), name=f"{self.name}__batch_size")
        self.parameter_dims = tf.Variable(tf.constant(parameter_dims), name=f"{self.name}__parameter_dims")

        self.local_parameters = [
            "channel_frequencies",
            "channel_width",
            "batch_size",
            "parameter_dims"
        ]
        self.global_parameters = ["batch_size", "parameter_dims"]

    def implementation(
        self,
        frequencies: tf.Tensor,
        /,
        channel_frequencies: tf.Tensor,
        channel_width: tf.Tensor,
        batch_size: tf.Tensor,
        parameter_dims: tf.Tensor
    ) -> tf.Tensor:
        return tf.cast(
            tf.tile(
                tf.expand_dims(
                    tf.math.logical_and(
                        tf.math.greater_equal(
                            frequencies,
                            tf.expand_dims(channel_frequencies - channel_width * 0.5, -1)
                        ),
                        tf.math.less_equal(
                            frequencies,
                            tf.expand_dims(channel_frequencies + channel_width * 0.5, -1)
                        )
                    ),
                    1
                ),
                [1, parameter_dims, 1]
            ),
            tf.complex64
        )


class GaussianFrequencyResponse(Component):
    """Frequency response function of a Gaussian filter."""

    #: Filter's central frequency in Hz
    central_frequency: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: Filter's 3 dB bandwidth in Hz
    bandwidth: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: Gaussian filter's order
    order: tf.Variable  # shape=[] or [None]; dtype=tf.int32

    batch_size: tf.Variable  # shape=[]; dtype=tf.int32
    parameter_dims: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        central_frequency: float | Sequence[float] | tf.Tensor,
        bandwidth: float | Sequence[float] | tf.Tensor,
        order: float | Sequence[float] | tf.Tensor,
        batch_size: float | tf.Tensor = 1,
        parameter_dims: float | tf.Tensor = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.central_frequency = tf.Variable(
            tf.constant(central_frequency),
            name=f"{self.name}__central_frequency"
        )
        self.bandwidth = tf.Variable(tf.constant(bandwidth), name=f"{self.name}__bandwidth")
        self.order = tf.Variable(tf.constant(order), name=f"{self.name}__order")
        self.batch_size = tf.Variable(tf.constant(batch_size), name=f"{self.name}__batch_size")
        self.parameter_dims = tf.Variable(tf.constant(parameter_dims), name=f"{self.name}__parameter_dims")

        self.local_parameters = [
            ("central_frequency", 0),
            ("bandwidth", 0),
            ("order", 0),
            "batch_size",
            "parameter_dims"
        ]

        self.global_parameters = ["batch_size", "parameter_dims"]

    def implementation(
        self,
        frequencies: tf.Tensor,
        /,
        central_frequency: tf.Tensor,
        bandwidth: tf.Tensor,
        order: tf.Tensor,
        batch_size: tf.Tensor,
        parameter_dims: tf.Tensor
    ) -> tf.Tensor:
        central_frequency = tf.expand_dims(central_frequency, -1)
        bandwidth = tf.expand_dims(bandwidth, -1)
        order = tf.cast(tf.expand_dims(order, -1), tf.float32)

        return tf.tile(
            tf.expand_dims(
                tf.cast(
                    tf.math.exp(
                        tf.math.divide(
                            -0.5 * tf.math.log(2.0) * tf.constant(2.0) ** (2.0 * order),
                            bandwidth ** (2.0 * order)
                        ) * (frequencies - central_frequency) ** (2.0 * order)
                    ),
                    tf.complex64
                ),
                0
            ),
            [batch_size, 1, 1]
        )

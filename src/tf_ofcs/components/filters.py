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

from typing import Optional
import tensorflow as tf

from ..utils import fftfreq, to_tensor
from .component import Component

from ..datatypes import Signal, FrequencyResponseFunction, TensorCompatible, Timing

__all__ = [
    "BandPassFilter",
    "FFTFilter",
    "FIRFilter",
    "BrickwallFrequencyResponse",
    "WDMBrickwallFrequencyResponse",
    "GaussianFrequencyResponse",
    "BesselFrequencyResponse"
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

    def call(self, signal: Signal, /, **kwargs) -> Signal:
        assert signal.samples.dtype in (tf.float32, tf.complex64)

        if signal.samples.dtype is not tf.complex64:
            fft = tf.signal.fft(tf.complex(signal.samples, 0.0))
        else:
            fft = tf.signal.fft(signal.samples)

        n = tf.shape(fft)[-1]
        freqs = tf.abs(fftfreq(n, 1.0 / signal.timing.sampling_rate))

        # We need override these parameters, so we drop them from kwargs
        kwargs.pop("batch_size", None)

        frequency_response = self.frequency_response_function(
            freqs,
            batch_size=tf.constant(tf.shape(fft)[0]),
            **kwargs
        )

        filtered_samples = tf.signal.ifft(frequency_response * fft)

        if signal.samples.dtype is not filtered_samples.dtype:
            filtered_samples = tf.cast(tf.math.real(filtered_samples), signal.samples.dtype)

        return Signal(timing=signal.timing, samples=filtered_samples)


class FIRFilter(Component):
    """
        Applies FIR filter to the input signal. Accepts `ElectricalSignal` as input.
        Returns a slightly longer (delayed) signal with
        `{sample count} + {number of taps} - 1` samples.

        .. note::

           `tf.nn.conv1d()` actually performs cross-correlation, not
           convolution, so the kernel taps should be in reverse order.

        :ivar taps: `shape=[None], dtype=tf.float32`, filter coefficients.
    """

    #: Filter coefficients
    taps: tf.Variable  # shape=[None]; dtype=tf.float32 or tf.complex64

    def __init__(
        self,
        taps: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.taps = tf.Variable(to_tensor(taps), name=f"{self.name}__taps")
        self.local_parameters = ["taps"]

    def implementation(self, signal: Signal, /, taps: tf.Tensor) -> Signal:
        assert signal.samples.dtype is tf.float32

        n_taps = tf.shape(taps)[-1]

        # Pad and transpose samples for tf.nn.conv1d()
        samples = tf.expand_dims(
            tf.pad(signal.samples, [[0, 0], [n_taps - 1, n_taps - 1]]),
            -1
        )  # shape=[1, sample_count + 2 * (n_taps - 1), batch_size]

        # Reshape filter taps
        taps = tf.reshape(taps, [n_taps, 1, 1])  # shape=[n_taps, 1, 1]

        filtered_samples = tf.nn.conv1d(
            samples,
            filters=taps,
            stride=1,
            padding="VALID"
        )[:, :, 0]  # shape=[batch_size, sample_count + n_taps, 1]

        return Signal(
            samples=filtered_samples,
            timing=Timing(
                sample_count=tf.shape(filtered_samples)[-1],
                sampling_rate=signal.timing.sampling_rate
            )
        )


class BandPassFilter(Component):
    """
        Applies band-pass filter to the input signal.
        Accepts both `ElectricalSignal` and `OpticalSignal` as input.

        :ivar min_frequency: `shape=[None], dtype=tf.float32`, minimum frequency to pass in Hz.
        :ivar max_frequency: `shape=[None], dtype=tf.float32`, maximum frequency to pass in Hz.
    """

    #: Minimum frequency in Hz
    min_frequency: tf.Variable  # shape=[None]; dtype=tf.float32

    #: Maximum frequency in Hz
    max_frequency: tf.Variable  # shape=[None]; dtype=tf.float32

    def __init__(
        self,
        min_frequency: TensorCompatible,
        max_frequency: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.min_frequency = tf.Variable(to_tensor(min_frequency), name=f"{self.name}__min_frequency")
        self.max_frequency = tf.Variable(to_tensor(max_frequency), name=f"{self.name}__max_frequency")
        self.local_parameters = ["min_frequency", "max_frequency"]

    def implementation(
        self,
        signal: Signal,
        /,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> Signal:
        assert signal.samples.dtype in (tf.float32, tf.complex64)

        if signal.samples.dtype is not tf.complex64:
            fft = tf.signal.fft(tf.complex(signal.samples, 0.0))
        else:
            fft = tf.signal.fft(signal.samples)

        min_frequency = tf.expand_dims(min_frequency, -1)
        max_frequency = tf.expand_dims(max_frequency, -1)

        freqs = tf.abs(fftfreq(tf.shape(fft)[-1], 1.0 / signal.timing.sampling_rate))
        pass_mask = tf.logical_and(tf.math.greater_equal(freqs, min_frequency), tf.less_equal(freqs, max_frequency))
        filtered_fft = tf.where(pass_mask, fft, tf.complex(0.0, 0.0))

        filtered_samples = tf.signal.ifft(filtered_fft)

        if signal.samples.dtype is not filtered_samples.dtype:
            filtered_samples = tf.math.real(filtered_samples)

        return Signal(timing=signal.timing, samples=filtered_samples)


class BrickwallFrequencyResponse(Component):
    """
        Implements brickwall bandpass filter's frequency response function.
        Passes only the frequencies between `min_frequency` and `max_frequency`.

        :ivar min_frequency: ``shape=[None], dtype=tf.float32`, minimum frequency to pass in Hz.
        :ivar max_frequency: `shape=[None], dtype=tf.float32`, maximum frequency to pass in Hz.
    """

    #: Minimum frequency in Hz
    min_frequency: tf.Variable  # shape=[None]; dtype=tf.float32

    #: Maximum frequency in Hz
    max_frequency: tf.Variable  # shape=[None]; dtype=tf.float32

    batch_size: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        min_frequency: TensorCompatible,
        max_frequency: TensorCompatible,
        batch_size: TensorCompatible = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.min_frequency = tf.Variable(to_tensor(min_frequency), name=f"{self.name}__min_frequency")
        self.max_frequency = tf.Variable(to_tensor(max_frequency), name=f"{self.name}__max_frequency")
        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")

        self.local_parameters = [
            "min_frequency",
            "max_frequency",
            "batch_size"
        ]

        self.global_parameters = ["batch_size"]

    def implementation(
        self,
        frequencies: tf.Tensor,
        /,
        batch_size: tf.Tensor,
        min_frequency: tf.Tensor,
        max_frequency: tf.Tensor
    ) -> tf.Tensor:
        min_frequency = tf.expand_dims(min_frequency, -1)
        max_frequency = tf.expand_dims(max_frequency, -1)

        return tf.cast(
            tf.math.logical_and(
                tf.math.greater_equal(frequencies, min_frequency),
                tf.math.less_equal(frequencies, max_frequency)
            ),
            tf.complex64
        )


class WDMBrickwallFrequencyResponse(Component):
    """
        Similar to :any:`BrickwallFrequencyResponse`, but one passband per WDM
        channel. This can be used for an optical demultiplexer.

        :ivar channel_frequencies: `shape=[None], dtype=tf.float32`, WDM channel frequencies in Hz
        :ivar channel_width: `shape=[], dtype=tf.float32`, WDM channel width in Hz
        :ivar batch_size: `shape=[], dtype=tf.int32`, batch size of the output
    """

    #: WDM channel frequencies in Hz
    channel_frequencies: tf.Variable  # shape=[None]; dtype=tf.float32

    #: WDM channel width in Hz
    channel_width: tf.Variable  # shape=[]; dtype=tf.float32

    batch_size: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        channel_frequencies: TensorCompatible,
        channel_width: TensorCompatible,
        batch_size: TensorCompatible = 1,
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

        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")

        self.local_parameters = [
            "channel_frequencies",
            "channel_width",
            "batch_size"
        ]
        self.global_parameters = ["batch_size"]

    def implementation(
        self,
        frequencies: tf.Tensor,
        /,
        channel_frequencies: tf.Tensor,
        channel_width: tf.Tensor,
        batch_size: tf.Tensor
    ) -> tf.Tensor:
        return tf.cast(
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
            tf.complex64
        )


# TODO: fix this later
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
        central_frequency: TensorCompatible,
        bandwidth: TensorCompatible,
        order: TensorCompatible,
        batch_size: TensorCompatible = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.central_frequency = tf.Variable(
            tf.constant(central_frequency),
            name=f"{self.name}__central_frequency"
        )
        self.bandwidth = tf.Variable(to_tensor(bandwidth), name=f"{self.name}__bandwidth")
        self.order = tf.Variable(to_tensor(order), name=f"{self.name}__order")
        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")

        self.local_parameters = [
            "central_frequency",
            "bandwidth",
            "order",
            "batch_size"
        ]

        self.global_parameters = ["batch_size"]

    def implementation(
        self,
        frequencies: tf.Tensor,
        /,
        central_frequency: tf.Tensor,
        bandwidth: tf.Tensor,
        order: tf.Tensor,
        batch_size: tf.Tensor
    ) -> tf.Tensor:
        central_frequency = tf.expand_dims(central_frequency, -1)
        bandwidth = tf.expand_dims(bandwidth, -1)
        order = tf.cast(tf.expand_dims(order, -1), tf.float32)

        return tf.cast(
            tf.math.exp(
                tf.math.divide(
                    -0.5 * tf.math.log(2.0) * tf.constant(2.0) ** (2.0 * order),
                    bandwidth ** (2.0 * order)
                ) * (frequencies - central_frequency) ** (2.0 * order)
            ),
            tf.complex64
        )


# TODO: fix this later
class BesselFrequencyResponse(Component):
    """Frequency response function of Bessel bandpass filter."""

    #: Filter's central frequency in Hz
    central_frequency: tf.Variable  # shape=[None]; dtype=tf.float32

    #: Bessel filter's bandwidth
    bandwidth: tf.Variable  # shape=[None]; dtype=tf.float32

    #: Bessel filter's order
    order: tf.Variable  # shape=[]; dtype=tf.int32

    batch_size: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        central_frequency: TensorCompatible,
        bandwidth: TensorCompatible,
        order: TensorCompatible,
        batch_size: TensorCompatible = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.central_frequency = tf.Variable(
            tf.constant(central_frequency),
            name=f"{self.name}__central_frequency"
        )
        self.bandwidth = tf.Variable(to_tensor(bandwidth), name=f"{self.name}__bandwidth")
        self.order = tf.Variable(to_tensor(order), name=f"{self.name}__order")
        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")

        self.local_parameters = [
            "central_frequency",
            "bandwidth",
            "order",
            "batch_size"
        ]

        self.global_parameters = ["batch_size"]

    def _factorial(self, n: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_greater_equal(n, 0)

        # assume n is int32, so we can't compute big factorials
        tf.debugging.assert_less(n, 13)

        shape = tf.shape(n)
        n = tf.reshape(n, [tf.size(n)])

        n = tf.math.maximum(n, 1)
        f = tf.map_fn(lambda i: tf.reduce_prod(tf.range(1, i + 1)), n)

        return tf.reshape(f, shape)

    def _a(self, k: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
        return tf.divide(
            self._factorial(2 * n - k),
            tf.math.multiply(
                tf.math.pow(2, n - k) * self._factorial(k),
                self._factorial(n - k)
            )
        )

    def _theta(self, s: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
        k = tf.range(n)  # shape=[n]

        # shape=[n, f]
        s_k = s ** tf.expand_dims(tf.expand_dims(tf.cast(k, tf.complex64), -1), -1)

        # Avoid NaN due to 0.0 ** 0.0
        s_k = tf.concat(
            [
                tf.fill(tf.concat([[1], tf.shape(s_k)[1:]], 0), tf.cast(1.0, tf.complex64)),
                s_k[1:]
            ],
            0
        )

        return tf.reduce_sum(
            tf.multiply(
                # shape=[n, 1]
                tf.expand_dims(tf.expand_dims(tf.cast(self._a(k, n), tf.complex64), -1), -1),
                s_k
            ),
            axis=0
        )

    def _theta_0(self, n: tf.Tensor) -> tf.Tensor:
        return tf.multiply(
            tf.cast(self._a(tf.constant(0), n), tf.complex64),
            tf.cast(1.0, tf.complex64)
        )

    def implementation(
        self,
        frequencies: tf.Tensor,
        /,
        central_frequency: tf.Tensor,
        bandwidth: tf.Tensor,
        order: tf.Tensor,
        batch_size: tf.Tensor
    ) -> tf.Tensor:
        central_frequency = tf.expand_dims(central_frequency, -1)
        bandwidth = tf.expand_dims(bandwidth, -1)

        return tf.divide(
            self._theta_0(order),
            self._theta(tf.complex(0.0, (frequencies - central_frequency) / bandwidth), order)
        )

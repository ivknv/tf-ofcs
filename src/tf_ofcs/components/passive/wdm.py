# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from ...utils import fftfreq, from_db
from ...datatypes import OpticalSignal
from ..component import Component

__all__ = ["OpticalMultiplexer", "OpticalDemultiplexer"]


class OpticalMultiplexer(Component):
    """
        Multiplexes multiple optical signals into one.

        Works similar to `ParallelToSerial` in a sense that it produces
        `batch_size // n` signals at the output.
    """

    #: Number of signals to multiplex (number of input ports)
    n: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(self, n: int | tf.Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = tf.Variable(tf.constant(n))
        self.local_parameters = ["n"]

    def implementation(self, input_signal: OpticalSignal, /, n: tf.Tensor) -> OpticalSignal:
        shape = tf.shape(input_signal.samples)
        batch_size = shape[0]

        tf.debugging.assert_equal(tf.math.floormod(batch_size, n), 0)

        output_samples = tf.math.reduce_sum(
            tf.reshape(
                input_signal.samples,
                tf.concat([[batch_size // n, n], shape[1:]], 0)
            ),
            axis=1
        ) / tf.math.sqrt(tf.cast(n, tf.complex64))

        return OpticalSignal(timing=input_signal.timing, samples=output_samples)


class OpticalDemultiplexer(Component):
    """
        Demultiplexes one optical signal into multiple based on frequencies.

        Works similar to `SerialToParallel` in a sense that it produces
        `batch_size * n` signals at the output.

        Every output signal corresponds to some specific frequency.
    """

    #: Frequencies of the output ports, this parameter also determines the
    #: number of output ports
    frequencies: tf.Variable  # shape=[None]; dtype=tf.float32

    #: Channel width in Hz
    channel_width: tf.Variable  # shape=[]; dtype=tf.float32

    #: Crosstalk attenuation in dB. This value applies to all adjacent channels
    crosstalk_attenuation: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        frequencies: Sequence[float] | tf.Tensor,
        channel_width: float | tf.Tensor,
        crosstalk_attenuation: float | tf.Tensor = float("inf"),
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.frequencies = tf.Variable(tf.constant(frequencies))
        self.channel_width = tf.Variable(tf.constant(channel_width))
        self.crosstalk_attenuation = tf.Variable(tf.constant(crosstalk_attenuation))
        self.local_parameters = ["frequencies", "channel_width", "crosstalk_attenuation"]

    def implementation(
        self,
        input_signal: OpticalSignal,
        /,
        frequencies: tf.Tensor,
        channel_width: tf.Tensor,
        crosstalk_attenuation: tf.Tensor
    ) -> OpticalSignal:
        fft = tf.signal.fft(input_signal.samples)
        fft_freqs = fftfreq(tf.shape(fft)[-1], 1.0 / input_signal.timing.sampling_rate)

        batch_size = tf.shape(fft)[0]
        n = tf.size(frequencies)

        # shape=[batch_size * n, 1, sample_count]
        pass_mask = tf.tile(
            tf.expand_dims(
                tf.math.logical_and(
                    tf.math.greater_equal(
                        fft_freqs,
                        tf.expand_dims(frequencies - channel_width * 0.5, -1)
                    ),
                    tf.math.less(
                        fft_freqs,
                        tf.expand_dims(frequencies + channel_width * 0.5, -1)
                    )
                ),
                1
            ),
            [batch_size, 1, 1]
        )

        fft = tf.tile(fft, [n, 1, 1])

        # Coefficient needed for the total power to be conserved
        c = tf.math.sqrt(tf.cast(n, tf.float32))

        attenuation = from_db(-crosstalk_attenuation * 0.5)
        passband_coeff = c * tf.math.sqrt(1.0 - (c * attenuation) ** 2)

        samples = tf.signal.ifft(
            fft * tf.where(
                pass_mask,
                tf.complex(passband_coeff, 0.0),
                tf.complex(attenuation, 0.0)
            )
        )

        return OpticalSignal(timing=input_signal.timing, samples=samples)

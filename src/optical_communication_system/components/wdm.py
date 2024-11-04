# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from ..utils import fftfreq
from ..datatypes import OpticalSignal
from .component import Component

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

    #: Multiplexing step in Hz
    step: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        frequencies: Sequence[float] | tf.Tensor,
        step: float | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.frequencies = tf.Variable(tf.constant(frequencies))
        self.step = tf.Variable(tf.constant(step))
        self.local_parameters = ["frequencies", "step"]

    def implementation(
        self,
        input_signal: OpticalSignal,
        /,
        frequencies: tf.Tensor,
        step: tf.Tensor
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
                        tf.expand_dims(frequencies - step * 0.5, -1)
                    ),
                    tf.math.less(
                        fft_freqs,
                        tf.expand_dims(frequencies + step * 0.5, -1)
                    )
                ),
                1
            ),
            [batch_size, 1, 1]
        )

        fft = tf.tile(fft, [n, 1, 1])

        samples = tf.signal.ifft(
            tf.where(pass_mask, fft, tf.complex(0.0, 0.0))
        ) * tf.math.sqrt(tf.cast(n, tf.complex64))

        return OpticalSignal(timing=input_signal.timing, samples=samples)

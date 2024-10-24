# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from ..component import Component
from ..datatypes import ElectricalSignal, OpticalSignal, Timing

__all__ = ["SerialToParallel"]


class SerialToParallel(Component):
    """
        Splits `m` serial signals into `m * n` parallel signals.
        A single parallel signal is considered to span `batch_size` // `n` batches
        of the signal.

        If the input signal has the shape `(batch_size, parameter_dims, sample_count)`,
        then the output shape is `(batch_size * n, parameter_dims, sample_count // n)`
    """

    #: Number of parallel signals at the output
    n: int

    def __init__(self, n: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = n

    def call(
        self,
        serial_signal: OpticalSignal | ElectricalSignal,
        /,
        **kwargs
    ) -> ElectricalSignal | OpticalSignal:
        timing = serial_signal.timing
        sampling_rate = timing.sampling_rate

        tf.debugging.assert_equal(tf.math.floormod(timing.sample_count, self.n), 0)

        SignalClass = serial_signal.__class__

        new_sample_count = timing.sample_count // self.n
        batch_size = tf.shape(serial_signal.samples)[0]

        sample_start_indices = tf.range(0, timing.sample_count, new_sample_count)
        sample_end_indices = sample_start_indices + new_sample_count

        new_signal_samples = tf.vectorized_map(
            lambda i: serial_signal.samples[
                i // self.n,
                :,
                sample_start_indices[i % self.n]:sample_end_indices[i % self.n]
            ],
            tf.range(batch_size * self.n)
        )

        new_timing = Timing(
            sample_count=new_sample_count,
            sampling_rate=sampling_rate,
        )

        return SignalClass(timing=new_timing, samples=new_signal_samples)

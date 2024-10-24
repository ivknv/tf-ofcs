# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from ..component import Component
from ..datatypes import ElectricalSignal, OpticalSignal, Timing

__all__ = ["Downsample"]


class Downsample(Component):
    #: Desired sampling rate in samples/second
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Shift signal in time before taking samples
    time_offset: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        sampling_rate: float | tf.Tensor,
        time_offset: float | tf.Tensor = 0.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.time_offset = tf.Variable(tf.constant(time_offset))
        self.local_parameters = ["sampling_rate", "time_offset"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        sampling_rate: tf.Tensor,
        time_offset: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        n = tf.cast(tf.round(signal.timing.sampling_rate / sampling_rate), tf.int32)
        shift = tf.cast(tf.round(signal.timing.sampling_rate * time_offset), tf.int32)

        indices = tf.range(shift, tf.shape(signal.samples)[-1] - shift, n)

        signal_samples = tf.gather(signal.samples, indices, axis=-1)

        return Signal(
            timing=Timing(
                sample_count=n,
                sampling_rate=sampling_rate
            ),
            samples=signal_samples
        )

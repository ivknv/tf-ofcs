# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from ..component import Component
from ...datatypes import ElectricalSignal, OpticalSignal, Timing

__all__ = ["Downsample"]


class Downsample(Component):
    #: Downsampling factor
    factor: tf.Variable  # shape=[]; dtype=int32

    #: Shift signal in time before taking samples
    time_offset: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        factor: int | tf.Tensor,
        time_offset: float | tf.Tensor = 0.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.factor = tf.Variable(tf.constant(factor))
        self.time_offset = tf.Variable(tf.constant(time_offset))
        self.local_parameters = ["factor", "time_offset"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        factor: tf.Tensor,
        time_offset: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        shift = tf.cast(tf.round(signal.timing.sampling_rate * time_offset), tf.int32)

        signal_samples = signal.samples[:, :, shift::factor]

        return Signal(
            timing=Timing(
                sample_count=tf.shape(signal_samples)[-1],
                sampling_rate=signal.timing.sampling_rate / tf.cast(factor, tf.float32)
            ),
            samples=signal_samples
        )

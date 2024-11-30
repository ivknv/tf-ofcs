# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from ..component import Component
from ...datatypes import ElectricalSignal, OpticalSignal, Timing

__all__ = ["Upsample"]


class Upsample(Component):
    #: Upsampling factor
    factor: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        factor: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.factor = tf.Variable(tf.constant(factor))
        self.local_parameters = ["factor"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        factor: tf.Tensor,
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        signal_samples = tf.repeat(signal.samples, factor, axis=-1)

        return Signal(
            timing=Timing(
                sample_count=tf.shape(signal_samples)[-1],
                sampling_rate=signal.timing.sampling_rate * tf.cast(factor, tf.float32)
            ),
            samples=signal_samples
        )

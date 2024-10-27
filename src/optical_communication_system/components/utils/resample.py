# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from ..component import Component
from ...datatypes import ElectricalSignal, OpticalSignal, Timing

__all__ = ["Resample"]


@tf.function
def _downsample(x: tf.Tensor, factor: tf.Tensor, axis: tf.Tensor = tf.constant(0)) -> tf.Tensor:
    return tf.gather(x, tf.range(0, tf.shape(x)[axis], factor), axis=axis)


class Resample(Component):
    #: Desired sampling rate of the signal
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    def __init__(self, sampling_rate: int | tf.Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.local_parameters = ["sampling_rate"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        sampling_rate: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        assert isinstance(signal, (ElectricalSignal, OpticalSignal))

        timing = signal.timing

        if sampling_rate > timing.sampling_rate:
            factor = tf.cast(tf.round(sampling_rate / timing.sampling_rate), tf.int32)

            new_sample_count = timing.sample_count * factor
            new_samples = tf.repeat(signal.samples, factor, axis=-1)
        else:
            factor = tf.cast(tf.round(timing.sampling_rate / sampling_rate), tf.int32)

            tf.debugging.assert_equal(tf.math.floormod(timing.sample_count, factor), 0)

            new_sample_count = timing.sample_count // factor

            new_samples = _downsample(signal.samples, factor, axis=-1)


        new_timing = Timing(
            sampling_rate=timing.sampling_rate * tf.cast(factor, tf.float32),
            sample_count=new_sample_count
        )

        SignalClass = signal.__class__

        return SignalClass(timing=new_timing, samples=new_samples)

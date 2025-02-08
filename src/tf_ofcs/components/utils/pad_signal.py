# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from ...datatypes import ElectricalSignal, OpticalSignal, Timing

from ..component import Component

__all__ = ["PadSignal"]


class PadSignal(Component):
    """Pads signal with zeros from left and right sides."""

    #: Left padding factor (ratio to signal duration)
    left_padding: tf.Variable  # shape=[]; dtype=tf.float32

    #: Right padding factor (ratio to signal duration)
    right_padding: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        left_padding: float | tf.Tensor,
        right_padding: Optional[float | tf.Tensor] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.left_padding = tf.Variable(tf.constant(left_padding), name=f"{self.name}__left_padding")

        if right_padding is None:
            right_padding = left_padding

        self.right_padding = tf.Variable(tf.constant(right_padding), name=f"{self.name}__right_padding")

        self.local_parameters = ["left_padding", "right_padding"]

    def implementation(
        self, signal: ElectricalSignal | OpticalSignal,
        /,
        left_padding: tf.Tensor,
        right_padding: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__
        shape = tf.shape(signal.samples)
        sample_count = tf.cast(shape[-1], tf.float32)

        left_sample_count = tf.cast(sample_count * left_padding, tf.int32)
        right_sample_count = tf.cast(sample_count * right_padding, tf.int32)

        left_samples = tf.zeros(
            tf.concat([shape[:-1], [left_sample_count]], 0),
            dtype=signal.samples.dtype
        )
        right_samples = tf.zeros(
            tf.concat([shape[:-1], [right_sample_count]], 0),
            dtype=signal.samples.dtype
        )

        padded_samples = tf.concat([left_samples, signal.samples, right_samples], -1)
        padded_timing = Timing(
            sample_count=signal.timing.sample_count + left_sample_count + right_sample_count,
            sampling_rate=signal.timing.sampling_rate
        )

        return Signal(samples=padded_samples, timing=padded_timing)

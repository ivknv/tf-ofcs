# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from ...datatypes import ElectricalSignal, OpticalSignal, Timing

from ..component import Component

__all__ = ["PadSignal"]


class PadSignal(Component):
    """
        Pads signal with zeros from left and right sides.
        If `unpad` is set to True, the padding will be removed from the signal
        instead.
    """

    #: Left padding factor (ratio to signal duration)
    left_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    #: Right padding factor (ratio to signal duration)
    right_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    #: If set to `True`, padding will be removed from the signal instead
    unpad: tf.Variable  # shape=[]; dtype=tf.bool

    def __init__(
        self,
        left_padding: float | tf.Tensor,
        right_padding: Optional[float | tf.Tensor] = None,
        unpad: bool | tf.Tensor = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if right_padding is None:
            right_padding = left_padding

        self.left_padding = tf.Variable(tf.constant(left_padding), name=f"{self.name}__left_padding")
        self.right_padding = tf.Variable(tf.constant(right_padding), name=f"{self.name}__right_padding")
        self.unpad = tf.Variable(tf.constant(unpad), name=f"{self.name}__unpad")

        self.local_parameters = ["left_padding", "right_padding", "unpad"]

    def implementation(
        self, signal: ElectricalSignal | OpticalSignal,
        /,
        left_padding: tf.Tensor,
        right_padding: tf.Tensor,
        unpad: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__
        shape = tf.shape(signal.samples)
        dtype = left_padding.dtype
        sample_count = tf.cast(shape[-1], dtype)

        if unpad:  # type: ignore[operator]
            # Figure out sample count of the signal before it was padded
            sample_count //= tf.cast(1, dtype) + left_padding + right_padding

        left_sample_count = tf.cast(sample_count * left_padding, tf.int32)
        right_sample_count = tf.cast(sample_count * right_padding, tf.int32)

        padded_samples = signal.samples

        if unpad:  # type: ignore[operator]
            if right_sample_count == 0:
                padded_samples = padded_samples[:, :, left_sample_count:]
            else:
                padded_samples = padded_samples[:, :, left_sample_count:-right_sample_count]
        else:
            left_samples = tf.zeros(
                tf.concat([shape[:-1], [left_sample_count]], 0),
                dtype=signal.samples.dtype
            )

            right_samples = tf.zeros(
                tf.concat([shape[:-1], [right_sample_count]], 0),
                dtype=signal.samples.dtype
            )

            padded_samples = tf.concat([left_samples, padded_samples, right_samples], -1)

        padded_timing = Timing(
            sample_count=tf.shape(padded_samples)[-1],
            sampling_rate=signal.timing.sampling_rate
        )

        return Signal(samples=padded_samples, timing=padded_timing)

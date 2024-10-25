# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from ..datatypes import ElectricalSignal, OpticalSignal
from ..component import Component

__all__ = ["SliceSignal"]


class SliceSignal(Component):
    """
        Slices the input signal, returning
        `samples[a * batch_size // factor:b * batch_size // factor]`,
        where `a, b = slice[0], slice[1]`.
    """

    slice: tf.Variable  # shape=[2]; dtype=tf.int32
    factor: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        slice: tuple[int, int] | tf.Tensor,
        factor: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.slice = tf.Variable(tf.constant(slice))
        self.factor = tf.Variable(tf.constant(factor))
        self.local_parameters = ["slice", "factor"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        slice: tf.Tensor,
        factor: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__
        slice = slice * tf.shape(signal.samples)[0] // factor

        return Signal(samples=signal.samples[slice[0]:slice[1]], timing=signal.timing)

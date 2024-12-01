# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf
from ..component import Component
from ...datatypes import ElectricalSignal, OpticalSignal

__all__ = ["RedistributeDimensions"]


class RedistributeDimensions(Component):
    """
        Reshapes the signal to [
            batch_size * factors[0] // factors[1],
            parameter_dims * factors[1] // factors[0],
            sample_count
        ]
    """

    factors: tf.Variable  # shape=[2]; dtype=int32

    def __init__(
        self,
        factors: tuple[int, int] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.factors = tf.Variable(tf.constant(factors), name=f"{self.name}__factors")
        self.local_parameters = ["factors"]

    def implementation(
        self,
        signal: OpticalSignal | ElectricalSignal,
        /,
        factors: tf.Tensor
    ) -> OpticalSignal | ElectricalSignal:
        Signal = signal.__class__

        batch_size = tf.shape(signal.samples)[0]
        parameter_dims = tf.shape(signal.samples)[1]
        sample_count = tf.shape(signal.samples)[2]

        batch_size = batch_size * factors[0] // factors[1]
        parameter_dims = parameter_dims * factors[1] // factors[0]

        return Signal(
            timing=signal.timing,
            samples=tf.reshape(signal.samples, (batch_size, parameter_dims, sample_count))
        )

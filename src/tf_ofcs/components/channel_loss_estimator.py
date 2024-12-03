# -*- coding: utf-8 -*-

import tensorflow as tf
from typing import Optional

from ..datatypes import Timing
from .component import Component
from .utils import Composition, Fork, Lambda, MapComponents

__all__ = ["ChannelLossEstimator"]


class ChannelLossEstimator(Composition):
    #: Input laser
    laser: Component

    #: Channel to test
    channel: Component

    #: Photodiode that will be used to estimate signal power
    photodiode: Component

    #: Sampling rate in samples/second
    sampling_rate: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        laser: Component,
        channel: Component,
        photodiode: Component,
        sampling_rate: float | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__([], name=name)

        self.laser = laser
        self.channel = channel
        self.photodiode = photodiode
        mean_amplitude = Lambda(
            lambda signal: tf.math.reduce_mean(tf.abs(signal.samples), axis=-1),
            name=f"{self.name}__lambda1"
        )

        self.components = [
            Lambda(self._generate_timing, allow_kwargs=True, name=f"{self.name}__lambda2"),
            laser,
            Fork(shape=[2], name=f"{self.name}__fork"),
            MapComponents([
                mean_amplitude,
                Composition(
                    [channel, photodiode, mean_amplitude],
                    name=f"{self.name}__composition"
                )
            ], name=f"{self.name}__map_components"),
            Lambda(self._estimate_attenuation, name=f"{self.name}__lambda3")
        ]

        self.sampling_rate = tf.Variable(tf.constant(float(sampling_rate)), name=f"{self.name}__sampling_rate")

        self.local_parameters = ["sampling_rate"]

    def _generate_timing(self, duration: tf.Tensor, **kwargs) -> tf.Tensor:
        sampling_rate = self.get_component_parameter(kwargs, "sampling_rate")

        return Timing(
            sampling_rate=sampling_rate,
            sample_count=tf.cast(duration * sampling_rate, tf.int32)
        )

    @tf.function(input_signature=((
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32)
    ),))
    def _estimate_attenuation(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        input_amplitude, output_power = inputs

        return input_amplitude ** 2 / output_power

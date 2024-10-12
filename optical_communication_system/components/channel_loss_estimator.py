# -*- coding: utf-8 -*-

import tensorflow as tf
from typing import Optional

from .component import Component
from .utils import Composition, Fork, Lambda, MapComponents, MeanAmplitude, TimingGenerator

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
        sampling_rate: float,
        name: Optional[str] = None
    ) -> None:
        self._timing_generator = TimingGenerator(sampling_rate=sampling_rate, symbol_rate=sampling_rate)

        self.laser = laser
        self.channel = channel
        self.photodiode = photodiode
        mean_amplitude = MeanAmplitude()

        super().__init__(
            components=[
                # We set the name to forward kwargs to the Lambda component
                Lambda(self._generate_timing, allow_kwargs=True, name=name),
                laser,
                Fork(shape=[2]),
                MapComponents(components=[
                    mean_amplitude,
                    Composition(components=[channel, photodiode, mean_amplitude])
                ]),
                Lambda(self._estimate_attenuation)
            ],
            name=name
        )

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))

        self.parameters = ["sampling_rate"]

    @tf.function
    def _generate_timing(self, duration: tf.Tensor, **kwargs) -> tf.Tensor:
        sampling_rate = self.get_component_parameter(kwargs, "sampling_rate")

        return self._timing_generator(
            tf.cast(duration * sampling_rate, tf.int32),
            **kwargs
        )

    @tf.function(input_signature=(
        (tf.TensorSpec(shape=[], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)),
    ))
    def _estimate_attenuation(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        input_amplitude, output_power = inputs

        return input_amplitude ** 2 / output_power

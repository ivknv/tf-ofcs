# -*- coding: utf-8 -*-

from collections.abc import Callable
from typing import Any, Optional

import tensorflow as tf
from . import constants

__all__ = [
    "fftfreq",
    "gaussian_distribution",
    "normalize_signal",
    "System",
    "MapComponent",
    "MapComponents",
    "Identity"
]


@tf.function(input_signature=(
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.float32)
))
def fftfreq(n: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
    if tf.math.floormod(n, 2) == 0:  # type: ignore
        return tf.cast(
            tf.concat([
                tf.range(tf.cast(tf.math.floordiv(n, 2), tf.int32)),
                tf.range(tf.cast(-tf.math.floordiv(n + 1, 2), tf.int32), 0)
            ], 0),
            tf.float32
        ) / (d * tf.cast(n, tf.float32))
    else:
        return tf.cast(
            tf.concat([
                tf.range(tf.cast(tf.math.floordiv(n + 1, 2), tf.int32)),
                tf.range(tf.cast(-tf.math.floordiv(n - 1, 2), tf.int32), 0)
            ], 0),
            tf.float32
        ) / (d * tf.cast(n, tf.float32))


@tf.function
def gaussian_distribution(x: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
    return 1 / (
        variance * tf.sqrt(tf.multiply(2.0, constants.pi))
    ) * tf.math.exp(-0.5 * tf.subtract(x, mean) ** 2 / variance ** 2)


@tf.function
def normalize_signal(samples: tf.Tensor) -> tf.Tensor:
    max_amplitude = tf.reduce_max(tf.abs(samples))

    return samples / max_amplitude


class System(tf.Module):
    #: System components
    components: list[Callable]

    def __init__(self, components: list[Callable], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    @tf.function
    def __call__(self, inp: Any = None) -> Any:
        result: Any = inp

        for component in self.components:
            result = component(result)

        return result


class MapComponent(tf.Module):
    component: Callable

    def __init__(self, *, component: Callable, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.component = component

    @tf.function
    def __call__(self, inputs: tuple) -> tuple:
        return tuple(self.component(input) for input in inputs)


class MapComponents(tf.Module):
    components: list[Callable]

    def __init__(self, *, components: list[Callable], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    @tf.function
    def __call__(self, inputs: tuple) -> tuple:
        return tuple(component(input) for component, input in zip(self.components, inputs, strict=True))


class Identity(tf.Module):
    @tf.function
    def __call__(self, inp: Any) -> Any:
        return inp

# -*- coding: utf-8 -*-

from collections.abc import Callable
from typing import Any, Optional

import tensorflow as tf
from . import constants
from .component import Component

__all__ = [
    "fftfreq",
    "gaussian_distribution",
    "normalize_signal",
    "adjust_symbol_for_mzm",
    "System",
    "MapComponent",
    "MapComponents",
    "Identity",
    "Reindex"
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


@tf.function
def adjust_symbol_for_mzm(symbol: tf.Tensor) -> tf.Tensor:
    return tf.math.acos(symbol) / constants.pi


class System(Component):
    #: System components
    components: list[Component]

    def __init__(self, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    @tf.function
    def __call__(self, inp: Any) -> Any:
        result: Any = inp

        for component in self.components:
            result = component(result)

        return result

    def capture(self, inp: Any) -> Any:
        result: Any = inp

        for component in self.components:
            result = component.capture(result)

        self.output = result

        return result


class MapComponent(Component):
    component: Component

    def __init__(self, *, component: Component, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.component = component

    @tf.function
    def __call__(self, inputs: tuple) -> tuple:
        return tuple(self.component(input) for input in inputs)

    def capture(self, inp: tuple) -> tuple:
        return tuple(self.component.capture(input) for input in inp)


class MapComponents(Component):
    components: list[Component]

    def __init__(self, *, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    @tf.function
    def __call__(self, inputs: tuple) -> tuple:
        return tuple(component(input) for component, input in zip(self.components, inputs, strict=True))

    def capture(self, inp: tuple) -> tuple:
        return super().capture(
            tuple(component.capture(input) for component, input in zip(self.components, inp, strict=True))
        )


class Identity(Component):
    @tf.function
    def __call__(self, inp: Any) -> Any:
        return inp


class Reindex(Component):
    indices: list[list | tuple[int, ...]]

    def __init__(
        self,
        indices: list[list | tuple[int, ...]],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.indices = indices

    def _get_tuple_index(self, inputs: tuple, index: tuple[int, ...]) -> Any:
        current = inputs

        for i in index:
            current = current[i]

        return current

    def _reindex(self, inputs: tuple, index: list | tuple[int, ...]) -> Any:
        if isinstance(index, tuple):
            return self._get_tuple_index(inputs, index)
        else:
            return tuple(self._reindex(inputs, i) for i in index)

    @tf.function
    def __call__(self, inputs: tuple) -> tuple:
        return self._reindex(inputs, self.indices)

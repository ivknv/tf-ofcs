# -*- coding: utf-8 -*-

from collections.abc import Sequence
import math
from typing import Any, Optional

import tensorflow as tf
from . import constants
from .component import Component

from .datatypes import (
    ElectricalSignal, OpticalSignal, Timing, bit_dtype, bit_sequence_spec
)

__all__ = [
    "fftfreq",
    "gaussian_distribution",
    "normalize_signal",
    "adjust_symbol_for_mzm",
    "bits_to_index",
    "index_to_bits",
    "symbol_mapping_to_gray",
    "psk_constellation",
    "pam_constellation",
    "Composition",
    "MapComponent",
    "MapComponents",
    "Identity",
    "Reindex",
    "Fork",
    "ExtractTiming",
    "TimingGenerator",
    "Resample"
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


@tf.function(input_signature=(bit_sequence_spec,))
def bits_to_index(bits: tf.Tensor) -> tf.Tensor:
    n = tf.size(bits)
    return tf.reduce_sum(bits * 2 ** (n - tf.range(n) - 1))


@tf.function(input_signature=2 * (tf.TensorSpec(shape=[], dtype=tf.int32),))
def index_to_bits(index: tf.Tensor, bit_count: tf.Tensor) -> tf.Tensor:
    return tf.vectorized_map(
        fn=lambda i: tf.cast(
            tf.math.floormod(index / 2 ** (bit_count - i - 1), 2),
            bit_dtype
        ),
        elems=tf.range(bit_count)
    )


@tf.function
def index_to_gray(index: tf.Tensor) -> tf.Tensor:
    return index ^ tf.bitwise.right_shift(index, 1)


def symbol_mapping_to_gray(mapping: list) -> list:
    gray_mapping = list(mapping)

    for i, symbol in enumerate(mapping):
        gray_mapping[index_to_gray(tf.constant(i))] = symbol

    return gray_mapping


def psk_constellation(
    bits_per_symbol: int,
    psk_mapping: Optional[Sequence[float]] = None
) -> list[list[float]]:
    if psk_mapping is None:
        symbol_count = 2 ** bits_per_symbol
        psk_mapping = [
            2 * math.pi * (-i) / symbol_count + math.pi * 1.25
            for i in range(symbol_count)
        ]

    iq_mapping = [[math.cos(phase), math.sin(phase)] for phase in psk_mapping]

    return iq_mapping


def pam_constellation(bits_per_symbol: int) -> list[float]:
    return [i / (2 ** bits_per_symbol - 1) for i in range(2 ** bits_per_symbol)]


class Composition(Component):
    #: System components
    components: list[Component]

    def __init__(self, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    @tf.function
    def __call__(self, inp: Any, **kwargs) -> Any:
        result: Any = inp

        for component in self.components:
            result = component(result, **kwargs)

        return result

    def trace(self, inp: Any, level: int = 0, **kwargs) -> Any:
        self.trace_call(inp, level)

        result: Any = inp

        for component in self.components:
            result = component.trace(result, level + 1, **kwargs)

        self.output = result
        self.trace_return_value(self.output, level + 1)

        return result


class MapComponent(Component):
    component: Component

    def __init__(self, *, component: Component, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.component = component

    @tf.function
    def __call__(self, inputs: tuple, **kwargs) -> tuple:
        return tuple(self.component(input, **kwargs) for input in inputs)

    def trace(self, inp: tuple, level: int = 0, **kwargs) -> tuple:
        self.trace_call(inp, level)

        self.output = tuple(
            self.component.trace(input, level + 1, **kwargs)
            for input in inp
        )

        self.trace_return_value(self.output, level + 1)

        return self.output


class MapComponents(Component):
    components: list[Component]

    def __init__(self, *, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    @tf.function
    def __call__(self, inputs: tuple, **kwargs) -> tuple:
        return tuple(
            component(input, **kwargs)
            for component, input in zip(self.components, inputs, strict=True)
        )

    def trace(self, inp: tuple, level: int = 0, **kwargs) -> tuple:
        self.trace_call(inp, level)

        self.output = tuple(
            component.trace(input, level + 1, **kwargs)
            for component, input in zip(self.components, inp, strict=True)
        )

        self.trace_return_value(self.output, level + 1)

        return self.output


class Identity(Component):
    @tf.function
    def implementation(self, inp: Any) -> Any:
        return inp


class Reindex(Component):
    indices: list | tuple[int, ...]

    def __init__(
        self,
        indices: list | tuple[int, ...],
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
    def implementation(self, inputs: tuple) -> Any:
        return self._reindex(inputs, self.indices)


class Fork(Component):
    shape: Sequence[int]

    def __init__(self, shape: Sequence[int], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.shape = shape

    @tf.function
    def implementation(self, inp: Any) -> tuple:
        result = inp

        for i in self.shape:
            result = (result,) * i

        return result


class ExtractTiming(Component):
    @tf.function
    def implementation(self, signal: ElectricalSignal | OpticalSignal) -> Timing:
        return signal.timing


class TimingGenerator(Component):
    sampling_rate: tf.Variable
    symbol_rate: tf.Variable
    duration: tf.Variable

    def __init__(
        self,
        sampling_rate: float,
        symbol_rate: float,
        duration: float = 0.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.symbol_rate = tf.Variable(tf.constant(symbol_rate))
        self.duration = tf.Variable(tf.constant(duration))
        self.parameters = ["sampling_rate", "symbol_rate", "duration"]

    @tf.function
    def implementation(
        self,
        duration: tf.Tensor,
        sampling_rate: tf.Tensor,
        symbol_rate: tf.Tensor
    ) -> Timing:
        time = tf.linspace(
            0.0,
            duration,
            tf.cast(tf.round(sampling_rate * duration), tf.int32)
        )

        return Timing(sampling_rate=sampling_rate, symbol_rate=symbol_rate, time=time)


@tf.function
def _upsample_with_linear_interpolation(x: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
    n = factor + 1

    return tf.concat(
        [
            x[:1],
            tf.reshape(
                tf.vectorized_map(
                    fn=lambda i: tf.linspace(x[i], x[i + 1], n)[1:],
                    elems=tf.range(tf.shape(x)[0] - 1)
                ),
                tf.concat([[(tf.shape(x)[0] - 1) * (n - 1)], tf.shape(x)[1:]], 0)
            )
        ],
        0
    )


@tf.function
def _downsample(x: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
    return tf.gather(x, tf.range(0, tf.shape(x)[0], factor))


class Resample(Component):
    #: Desired sampling rate of the signal
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    def __init__(self, sampling_rate: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.parameters = ["sampling_rate"]

    @tf.function
    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        sampling_rate: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        assert isinstance(signal, (ElectricalSignal, OpticalSignal))

        timing = signal.timing

        if sampling_rate > timing.sampling_rate:
            factor = tf.cast(tf.round(sampling_rate / timing.sampling_rate), tf.int32)

            new_time = _upsample_with_linear_interpolation(timing.time, factor - 1)
            new_samples = tf.repeat(signal.samples, [factor])
        else:
            factor = tf.cast(tf.round(timing.sampling_rate / sampling_rate), tf.int32)

            new_time = _downsample(timing.time, factor)
            new_samples = _downsample(signal.samples, factor)

        new_timing = Timing(
            sampling_rate=timing.sampling_rate * tf.cast(factor, tf.float32),
            symbol_rate=timing.symbol_rate,
            time=new_time
        )

        SignalClass = signal.__class__

        return SignalClass(timing=new_timing, samples=new_samples)

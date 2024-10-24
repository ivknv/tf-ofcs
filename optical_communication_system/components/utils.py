# -*- coding: utf-8 -*-

from collections.abc import Callable, Sequence
import math
from typing import Any, Optional
import typing

import tensorflow as tf
from .. import constants
from .component import Component

from .datatypes import (
    ElectricalSignal, electrical_signal_spec, OpticalSignal, Timing, timing_spec,
    bit_dtype, bit_sequence_spec
)

__all__ = [
    "fftfreq",
    "gaussian_distribution",
    "normalize_signal",
    "adjust_symbol_for_mzm",
    "bits_to_indices",
    "indices_to_bits",
    "symbol_mapping_to_gray",
    "psk_constellation",
    "pam_constellation",
    "generate_timing",
    "timing_to_samples",
    "combine_shape",
    "Composition",
    "MapComponent",
    "MapComponents",
    "Identity",
    "Reindex",
    "Fork",
    "ExtractTiming",
    "TimingGenerator",
    "Resample",
    "Downsample",
    "MeanAmplitude",
    "ParallelToSerial",
    "SerialToParallel",
    "ExtractSamples",
    "Lambda",
    "RedistributeDimensions"
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


@tf.function(input_signature=(bit_sequence_spec, tf.TensorSpec(shape=[], dtype=tf.int32)))
def bits_to_indices(bits: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(bits)
    bits = tf.reshape(bits, (tf.size(bits) // n, n))

    return tf.reshape(
        tf.reduce_sum(bits * 2 ** (n - tf.range(n) - 1), axis=1),
        tf.concat([shape[:-1], [shape[-1] // n]], 0)
    )


@tf.function(input_signature=(
    tf.TensorSpec(shape=typing.cast(tf.TensorShape, None), dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.int32))
)
def indices_to_bits(indices: tf.Tensor, bit_count: tf.Tensor) -> tf.Tensor:
    denominators = 2 ** (bit_count - tf.range(bit_count) - 1)

    return tf.cast(
        tf.math.floormod(tf.expand_dims(indices, -1) // denominators, 2),
        bit_dtype
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


@tf.function
def pam_constellation(bits_per_symbol: tf.Tensor, size: Optional[tf.Tensor] = None) -> tf.Tensor:
    symbol_count = tf.math.pow(2, bits_per_symbol)

    if size is None:
        size = symbol_count

    return tf.concat([
        tf.cast(tf.range(symbol_count) / (symbol_count - 1), tf.float32),
        tf.fill([size - symbol_count], 0.0)
    ], 0)


@tf.function(input_signature=(
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32)
))
def generate_timing(
    sampling_rate: tf.Tensor,
    symbol_rate: tf.Tensor,
    symbol_count: tf.Tensor
) -> Timing:
    sample_count = symbol_count * tf.cast(
        tf.round(sampling_rate / symbol_rate),
        tf.int32
    )

    return Timing(sample_count=sample_count, sampling_rate=sampling_rate)


@tf.function(input_signature=(timing_spec,))
def timing_to_samples(timing: Timing) -> tf.Tensor:
    return tf.cast(tf.range(timing.sample_count), tf.float32) / timing.sampling_rate


@tf.function
def combine_shape(shapes: tf.Tensor) -> tf.Tensor:
    combined = tf.concat(shapes, 0)
    return tf.boolean_mask(combined, combined > 0)


class Composition(Component):
    #: System components
    components: list[Component]

    def __init__(self, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    def call(self, inp: Any, /, tracing_level: int = 0, **kwargs) -> Any:
        result: Any = inp

        for component in self.components:
            result = component(result, tracing_level=tracing_level + 1, **kwargs)

        return result


class MapComponent(Component):
    component: Component

    def __init__(self, component: Component, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.component = component

    def call(self, inputs: tuple, /, tracing_level: int = 0, **kwargs) -> tuple:
        return tuple(
            self.component(
                input,
                tracing_level=tracing_level + 1,
                **kwargs
            )
            for input in inputs
        )


class MapComponents(Component):
    components: list[Component]

    def __init__(self, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    def call(
        self,
        inputs: tuple,
        /,
        tracing_level: int = 0,
        **kwargs
    ) -> tuple:
        return tuple(
            component(
                input,
                tracing_level=tracing_level + 1,
                **kwargs
            )
            for component, input in zip(self.components, inputs, strict=True)
        )


class Identity(Component):
    def call(self, inp: Any, /, **kwargs) -> Any:
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

    def call(self, inputs: tuple, /, **kwargs) -> Any:
        return self._reindex(inputs, self.indices)


class Fork(Component):
    shape: Sequence[int]

    def __init__(self, shape: Sequence[int], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.shape = shape

    def call(self, inp: Any, /, **kwargs) -> tuple:
        result = inp

        for i in self.shape:
            result = (result,) * i

        return result


class ExtractTiming(Component):
    def call(self, signal: ElectricalSignal | OpticalSignal, /, **kwargs) -> Timing:
        return signal.timing


class TimingGenerator(Component):
    sampling_rate: tf.Variable  # shape=[]; dtype=tf.float32
    symbol_rate: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        sampling_rate: float | tf.Tensor,
        symbol_rate: float | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.symbol_rate = tf.Variable(tf.constant(symbol_rate))
        self.local_parameters = ["sampling_rate", "symbol_rate"]

    def implementation(
        self,
        symbol_count: tf.Tensor,
        /,
        sampling_rate: tf.Tensor,
        symbol_rate: tf.Tensor,
        **kwargs
    ) -> Timing:
        return generate_timing(sampling_rate, symbol_rate, symbol_count)


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


class Downsample(Component):
    #: Desired sampling rate in samples/second
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Shift signal in time before taking samples
    time_offset: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        sampling_rate: float | tf.Tensor,
        time_offset: float | tf.Tensor = 0.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.time_offset = tf.Variable(tf.constant(time_offset))
        self.local_parameters = ["sampling_rate", "time_offset"]

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ))
    def implementation(
        self,
        signal: ElectricalSignal,
        /,
        sampling_rate: tf.Tensor,
        time_offset: tf.Tensor
    ) -> ElectricalSignal:
        n = tf.cast(tf.round(signal.timing.sampling_rate / sampling_rate), tf.int32)
        shift = tf.cast(tf.round(signal.timing.sampling_rate * time_offset), tf.int32)

        indices = tf.range(shift, tf.shape(signal.samples)[-1] - shift, n)

        signal_samples = tf.gather(signal.samples, indices, axis=-1)

        return ElectricalSignal(
            timing=Timing(
                sample_count=n,
                sampling_rate=sampling_rate
            ),
            samples=signal_samples
        )


class MeanAmplitude(Component):
    def call(self, input_signal: OpticalSignal | ElectricalSignal, /, **kwargs) -> tf.Tensor:
        assert isinstance(input_signal, (ElectricalSignal, OpticalSignal))
        return tf.math.reduce_mean(tf.abs(input_signal.samples), axis=-1)


class ParallelToSerial(Component):
    """
        Merges `n` parallel signals into one or more serial signals.
        A single parallel signal is considered to span `batch_size` // `n` batches
        of the signal.

        If the input signal has the shape `(batch_size, parameter_dims, sample_count)`,
        then the output shape is `(batch_size // n, parameter_dims, sample_count * n)`
    """

    #: Number of parallel signals to merge into one
    n: int

    def __init__(self, n: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = n

    def call(
        self,
        input_signal: OpticalSignal | ElectricalSignal,
        /,
        **kwargs
    ) -> ElectricalSignal | OpticalSignal:
        timing = input_signal.timing
        sampling_rate = timing.sampling_rate

        SignalClass = input_signal.__class__

        new_sample_count = timing.sample_count * self.n

        batch_size = tf.shape(input_signal.samples)[0]

        tf.debugging.assert_equal(tf.math.floormod(batch_size, self.n), 0)

        new_signal_samples = tf.vectorized_map(
            lambda i: tf.concat(
                [input_signal.samples[i * self.n + j] for j in range(self.n)],
                1
            ),
            tf.range(batch_size // self.n))

        return SignalClass(
            timing=Timing(
                sample_count=new_sample_count,
                sampling_rate=sampling_rate
            ),
            samples=new_signal_samples
        )


class SerialToParallel(Component):
    """
        Splits `m` serial signals into `m * n` parallel signals.
        A single parallel signal is considered to span `batch_size` // `n` batches
        of the signal.

        If the input signal has the shape `(batch_size, parameter_dims, sample_count)`,
        then the output shape is `(batch_size * n, parameter_dims, sample_count // n)`
    """

    #: Number of parallel signals at the output
    n: int

    def __init__(self, n: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = n

    def call(
        self,
        serial_signal: OpticalSignal | ElectricalSignal,
        /,
        **kwargs
    ) -> ElectricalSignal | OpticalSignal:
        timing = serial_signal.timing
        sampling_rate = timing.sampling_rate

        tf.debugging.assert_equal(tf.math.floormod(timing.sample_count, self.n), 0)

        SignalClass = serial_signal.__class__

        new_sample_count = timing.sample_count // self.n
        batch_size = tf.shape(serial_signal.samples)[0]

        sample_start_indices = tf.range(0, timing.sample_count, new_sample_count)
        sample_end_indices = sample_start_indices + new_sample_count

        new_signal_samples = tf.vectorized_map(
            lambda i: serial_signal.samples[
                i // self.n,
                :,
                sample_start_indices[i % self.n]:sample_end_indices[i % self.n]
            ],
            tf.range(batch_size * self.n)
        )

        new_timing = Timing(
            sample_count=new_sample_count,
            sampling_rate=sampling_rate,
        )

        return SignalClass(timing=new_timing, samples=new_signal_samples)


class ExtractSamples(Component):
    def call(self, signal: ElectricalSignal | OpticalSignal, /, **kwargs) -> tf.Tensor:
        assert isinstance(signal, (ElectricalSignal, OpticalSignal))
        return signal.samples


class Lambda(Component):
    #: `Callable` that will be invoked
    function: Callable

    #: If `True`, keyword arguments will be passed to `function`
    allow_kwargs: bool

    def __init__(
        self,
        function: Callable,
        allow_kwargs: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.function = function
        self.allow_kwargs = allow_kwargs

    def call(self, inp: Any, /, **kwargs) -> Any:
        if self.allow_kwargs:
            return self.function(inp, **kwargs)

        return self.function(inp)


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

        self.factors = tf.Variable(tf.constant(factors))
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

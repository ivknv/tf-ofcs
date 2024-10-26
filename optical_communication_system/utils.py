# -*- coding: utf-8 -*-

from collections.abc import Sequence
import math
import typing
import tensorflow as tf

from .datatypes import Timing, timing_spec, bit_sequence_spec, bit_dtype
from . import constants

__all__ = [
    "log10",
    "dbm_to_w",
    "dbm_to_mw",
    "w_to_dbm",
    "mw_to_dbm",
    "from_db",
    "to_db",
    "wavelength_m_to_frequency",
    "wavelength_nm_to_frequency",
    "frequency_to_wavelength_m",
    "frequency_to_wavelength_nm",
    "linewidth_m_to_frequency",
    "linewidth_nm_to_frequency",
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
    "lorentz_distribution"
]

_log_of_10 = tf.math.log(10.0)


def log10(x: float | tf.Tensor) -> tf.Tensor:
    return tf.math.log(x) / _log_of_10


def dbm_to_w(power: float | tf.Tensor) -> tf.Tensor:
    return tf.math.pow(10.0, power * 0.1) * 1e-3


def dbm_to_mw(power: float | tf.Tensor) -> tf.Tensor:
    return tf.math.pow(10.0, power * 0.1)


def w_to_dbm(power: float | tf.Tensor) -> tf.Tensor:
    return 10.0 * log10(power * 1e3)


def mw_to_dbm(power: float | tf.Tensor) -> tf.Tensor:
    return 10.0 * log10(power)


def from_db(ratio: float | tf.Tensor) -> tf.Tensor:
    return tf.math.pow(10.0, ratio * 0.1)


def to_db(ratio: float | tf.Tensor) -> tf.Tensor:
    return 10.0 * log10(ratio)


def wavelength_m_to_frequency(wavelength: float | tf.Tensor) -> tf.Tensor:
    return constants.c / wavelength


def wavelength_nm_to_frequency(wavelength: float | tf.Tensor) -> tf.Tensor:
    return constants.c * 1e9 / wavelength


def frequency_to_wavelength_m(frequency: float | tf.Tensor) -> tf.Tensor:
    return constants.c / frequency


def frequency_to_wavelength_nm(frequency: float | tf.Tensor) -> tf.Tensor:
    return constants.c * 1e9 / frequency


def linewidth_m_to_frequency(
    linewidth: float | tf.Tensor,
    central_wavelength: float | tf.Tensor
) -> tf.Tensor:
    half = linewidth * 0.5

    return tf.subtract(
        wavelength_m_to_frequency(central_wavelength - half),
        wavelength_m_to_frequency(central_wavelength + half)
    )


def linewidth_nm_to_frequency(
    linewidth: float | tf.Tensor,
    central_wavelength: float | tf.Tensor
) -> tf.Tensor:
    half = linewidth * 0.5

    return tf.subtract(
        wavelength_nm_to_frequency(central_wavelength - half),
        wavelength_nm_to_frequency(central_wavelength + half)
    )


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
    psk_mapping: typing.Optional[Sequence[float]] = None
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
def pam_constellation(bits_per_symbol: tf.Tensor, size: typing.Optional[tf.Tensor] = None) -> tf.Tensor:
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


@tf.function
def lorentz_distribution(x: tf.Tensor, x_0: tf.Tensor, gamma: tf.Tensor) -> tf.Tensor:
    return gamma / (math.pi * ((x - x_0) ** 2 + gamma ** 2))

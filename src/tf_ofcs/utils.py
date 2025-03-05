# -*- coding: utf-8 -*-
# Copyright (C) 2024 Ivan Konovalov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Sequence
import math
import typing
import tensorflow as tf

from .datatypes import TensorCompatible, Timing, bit_dtype
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
    "lorentz_distribution",
    "brickwall_frequency_response",
    "to_tensor"
]

_log_of_10 = tf.math.log(10.0)


def log10(x: TensorCompatible) -> tf.Tensor:
    return tf.math.log(x) / _log_of_10


def dbm_to_w(power: TensorCompatible) -> tf.Tensor:
    return tf.math.pow(10.0, to_tensor(power) * 0.1) * 1e-3


def dbm_to_mw(power: TensorCompatible) -> tf.Tensor:
    return tf.math.pow(10.0, to_tensor(power) * 0.1)


def w_to_dbm(power: TensorCompatible) -> tf.Tensor:
    return 10.0 * log10(to_tensor(power) * 1e3)


def mw_to_dbm(power: TensorCompatible) -> tf.Tensor:
    return 10.0 * log10(power)


def from_db(ratio: TensorCompatible) -> tf.Tensor:
    return tf.math.pow(10.0, to_tensor(ratio) * 0.1)


def to_db(ratio: TensorCompatible) -> tf.Tensor:
    return 10.0 * log10(ratio)


def wavelength_m_to_frequency(wavelength: TensorCompatible) -> tf.Tensor:
    return constants.c / wavelength


def wavelength_nm_to_frequency(wavelength: TensorCompatible) -> tf.Tensor:
    return constants.c * 1e9 / wavelength


def frequency_to_wavelength_m(frequency: TensorCompatible) -> tf.Tensor:
    return constants.c / frequency


def frequency_to_wavelength_nm(frequency: TensorCompatible) -> tf.Tensor:
    return constants.c * 1e9 / frequency


def linewidth_m_to_frequency(
    linewidth: TensorCompatible,
    central_wavelength: TensorCompatible
) -> tf.Tensor:
    half = tf.stack(linewidth) * 0.5

    return tf.subtract(
        wavelength_m_to_frequency(central_wavelength - half),
        wavelength_m_to_frequency(central_wavelength + half)
    )


def linewidth_nm_to_frequency(
    linewidth: TensorCompatible,
    central_wavelength: TensorCompatible
) -> tf.Tensor:
    half = to_tensor(linewidth) * 0.5

    return tf.subtract(
        wavelength_nm_to_frequency(central_wavelength - half),
        wavelength_nm_to_frequency(central_wavelength + half)
    )


def fftfreq(n: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
    """
        Return the Discrete Fourier Transform sample frequencies.

        :param n: `shape=[], dtype=tf.int32`, sample window length (number of points).
        :param d: `shape=[], dtype=tf.float32`, sample spacing (inverse of the sampling rate).

        :return: `shape=[n], dtype=tf.float32`, the values of the frequency
                 bins in cycles/unit of the sample spacing.
    """

    def even_case() -> tf.Tensor:
        return tf.cast(
            tf.concat([
                tf.range(tf.cast(tf.math.floordiv(n, 2), tf.int32)),
                tf.range(tf.cast(-tf.math.floordiv(n + 1, 2), tf.int32), 0)
            ], 0),
            tf.float32
        ) / (d * tf.cast(n, tf.float32))

    def odd_case() -> tf.Tensor:
        return tf.cast(
            tf.concat([
                tf.range(tf.cast(tf.math.floordiv(n + 1, 2), tf.int32)),
                tf.range(tf.cast(-tf.math.floordiv(n - 1, 2), tf.int32), 0)
            ], 0),
            tf.float32
        ) / (d * tf.cast(n, tf.float32))

    return tf.cond(tf.math.floormod(n, 2) == 0, even_case, odd_case)

def gaussian_distribution(x: tf.Tensor, mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
    return 1 / (
        variance * tf.sqrt(tf.multiply(2.0, constants.pi))
    ) * tf.math.exp(-0.5 * tf.subtract(x, mean) ** 2 / variance ** 2)


def normalize_signal(samples: tf.Tensor) -> tf.Tensor:
    max_amplitude = tf.reduce_max(tf.abs(samples))

    return samples / max_amplitude


def adjust_symbol_for_mzm(symbol: tf.Tensor) -> tf.Tensor:
    return tf.math.acos(symbol) / constants.pi


def bits_to_indices(bits: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
    """
        Converts a tensor of bits into a tensor of indices.

        :param bits: `shape=[None, bit_count], dtype=tf.int32`, tensor of bits.
        :param n: `shape=[], dtype=tf.int32`, number of bits per symbol.

        :return: `shape=[None, bit_count // n], dtype=tf.int32`, tensor of indices.
    """

    shape = tf.shape(bits)
    bits = tf.reshape(bits, (tf.size(bits) // n, n))

    return tf.reshape(
        tf.reduce_sum(bits * 2 ** (n - tf.range(n) - 1), axis=1),
        tf.concat([shape[:-1], [shape[-1] // n]], 0)
    )


def indices_to_bits(indices: tf.Tensor, bit_count: tf.Tensor) -> tf.Tensor:
    """
        Converts a tensor of indices into a tensor of bits.

        :param indices: any shape, `dtype=tf.int32`, tensor of `n` indices.
        :param bit_count: `shape=[], dtype=tf.int32`, number of bits per symbol.

        :return: same shape but with `n * bit_count` bits
    """

    denominators = 2 ** (bit_count - tf.range(bit_count) - 1)

    return tf.cast(
        tf.math.floormod(tf.expand_dims(indices, -1) // denominators, 2),
        bit_dtype
    )


def index_to_gray(index: tf.Tensor) -> tf.Tensor:
    """
        Converts an index to its Gray code representation.

        :param index: any shape, `dtype=tf.int32`, tensor of indices.

        :return: same shape but with Gray code representation
    """

    return index ^ tf.bitwise.right_shift(index, 1)


def symbol_mapping_to_gray(mapping: list) -> list:
    """
        Applies Gray code to a symbol mapping.

        :param mapping: `list`, list of symbols.

        :return: `list`, list of symbols with Gray code applied.
    """

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


def pam_constellation(bits_per_symbol: tf.Tensor, size: typing.Optional[tf.Tensor] = None) -> tf.Tensor:
    symbol_count = tf.math.pow(2, bits_per_symbol)

    if size is None:
        size = symbol_count

    return tf.concat([
        tf.cast(tf.range(symbol_count) / (symbol_count - 1), tf.float32),
        tf.fill([size - symbol_count], 0.0)
    ], 0)


def generate_timing(
    sampling_rate: tf.Tensor,
    symbol_rate: tf.Tensor,
    symbol_count: tf.Tensor
) -> Timing:
    """
        Produces a `Timing` object.

        :param sampling_rate: `shape=[], dtype=tf.float32`, sampling rate in Hz.
        :param symbol_rate: `shape=[], dtype=tf.float32`, symbol rate in symbols/second.
        :param symbol_count: `shape=[], dtype=tf.int32`, number of symbols.

        :return: `Timing` object.
    """

    sample_count = symbol_count * tf.cast(
        tf.round(sampling_rate / symbol_rate),
        tf.int32
    )

    return Timing(sample_count=sample_count, sampling_rate=sampling_rate)


def timing_to_samples(timing: Timing) -> tf.Tensor:
    """
        Converts a `Timing` object into a tensor of sample times.

        :param timing: `Timing` object.

        :return: `shape=[timing.sample_count], dtype=tf.float32`, tensor of sample times.
    """

    return tf.cast(tf.range(timing.sample_count), tf.float32) / timing.sampling_rate


def combine_shape(shapes: tf.Tensor) -> tf.Tensor:
    combined = tf.concat(shapes, 0)
    return tf.boolean_mask(combined, combined > 0)


def lorentz_distribution(x: tf.Tensor, x_0: tf.Tensor, gamma: tf.Tensor) -> tf.Tensor:
    return gamma / (math.pi * ((x - x_0) ** 2 + gamma ** 2))


def brickwall_frequency_response(
    frequencies: tf.Tensor,
    shape: tf.Tensor,
    *,
    min_frequency: tf.Tensor,
    max_frequency: tf.Tensor
) -> tf.Tensor:
    return tf.cast(
        tf.tile(
            tf.expand_dims(
                tf.math.logical_and(
                    tf.math.greater_equal(frequencies, min_frequency),
                    tf.math.less(frequencies, max_frequency)
                ),
                1
            ),
            [shape[0], shape[1], 1]
        ),
        tf.complex64
    )


def to_tensor(x: typing.Any) -> tf.Tensor:
    """
        Converts a value to a tensor.

        :param x: `typing.Any`, value to convert.
        :return: `tf.Tensor`, tensor representation of the value.
    """

    return tf.stack(x)

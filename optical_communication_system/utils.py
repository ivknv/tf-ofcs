# -*- coding: utf-8 -*-

import tensorflow as tf
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
    "linewidth_nm_to_frequency"
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

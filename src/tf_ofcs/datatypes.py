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

import tensorflow as tf
from typing import Protocol

__all__ = [
    "Timing",
    "timing_spec",
    "ElectricalSignal",
    "electrical_signal_spec",
    "OpticalSignal",
    "optical_signal_spec",
    "bit_dtype",
    "bit_sequence_spec",
    "IQSignal",
    "iq_signal_spec",
    "FrequencyResponseFunction"
]


class Timing(tf.experimental.ExtensionType):
    #: Number of samples in the signal
    sample_count: tf.Tensor  # shape=[]; dtype=tf.int32

    #: Sampling rate in Hz
    sampling_rate: tf.Tensor  # shape=[]; dtype=tf.float32


timing_spec = Timing.Spec(
    sample_count=tf.TensorSpec(shape=[], dtype=tf.int32),
    sampling_rate=tf.TensorSpec(shape=[], dtype=tf.float32)
)


class ElectricalSignal(tf.experimental.ExtensionType):
    #: Signal's timing information
    timing: Timing

    #: Signal's samples
    samples: tf.Tensor  # shape=[batch_size, parameter_dims, timing.sample_count]; dtype=tf.float32;


electrical_signal_spec = ElectricalSignal.Spec(
    timing=timing_spec,
    samples=tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
)


class OpticalSignal(tf.experimental.ExtensionType):
    #: Signal's timing information
    timing: Timing

    #: Signal's samples
    samples: tf.Tensor  # shape=[batch_size, parameter_dims, timing.sample_count]; dtype=tf.complex64


optical_signal_spec = OpticalSignal.Spec(
    timing=timing_spec,
    samples=tf.TensorSpec(shape=[None, None, None], dtype=tf.complex64)
)

bit_dtype = tf.int32
bit_sequence_spec = tf.TensorSpec(shape=[None, None], dtype=bit_dtype)

IQSignal = tuple[ElectricalSignal, ElectricalSignal]
iq_signal_spec = (electrical_signal_spec, electrical_signal_spec)


class FrequencyResponseFunction(Protocol):
    """
        Protocol for a function, describing filter's frequency response.

        :param frequencies: `tf.Tensor`, shape=[n], dtype=tf.float32, tensor of
                            input frequencies in Hz, where n is the number of
                            frequencies
        :param batch_size: `tf.Tensor`, expected batch_size dimension of the output
        :param parameter_dims: `tf.Tensor`, expected parameter_dims dimension of the output
        :param kwargs: other arguments passed from components

        .. note::

           batch_size and parameter_dims might not correspond to the actual
           signal's shape, they only indicate the desired shape of frequency response.

        :returns: a `tf.Tensor` with shape=[batch_size, parameter_dims, n], dtype=tf.complex64
    """

    def __call__(
        self,
        frequencies: tf.Tensor,
        /,
        **kwargs
    ) -> tf.Tensor:
        ...

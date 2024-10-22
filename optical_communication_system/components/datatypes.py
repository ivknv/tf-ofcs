# -*- coding: utf-8 -*-

import tensorflow as tf

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
    "iq_signal_spec"
]


class Timing(tf.experimental.ExtensionType):
    #: Number of samples in the signal
    sample_count: tf.Tensor  # shape=[]; dtype=tf.int32

    #: Sampling rate in Hz
    sampling_rate: tf.Tensor  # shape=[]; dtype=tf.float32

    #: Data rate in symbols/second
    symbol_rate: tf.Tensor  # shape=[]; dtype=tf.float32


timing_spec = Timing.Spec(
    sample_count=tf.TensorSpec(shape=[], dtype=tf.int32),
    sampling_rate=tf.TensorSpec(shape=[], dtype=tf.float32),
    symbol_rate=tf.TensorSpec(shape=[], dtype=tf.float32)
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

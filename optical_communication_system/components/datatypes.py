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
    #: Time axis of the signal (in seconds)
    time: tf.Tensor  # shape=[None]; dtype=tf.float32

    #: Sampling rate in Hz
    sampling_rate: tf.Tensor  # shape=[]; dtype=tf.float32

    #: Data rate in symbols/second
    symbol_rate: tf.Tensor  # shape=[]; dtype=tf.float32


timing_spec = Timing.Spec(
    time=tf.TensorSpec(shape=[None], dtype=tf.float32),
    sampling_rate=tf.TensorSpec(shape=[], dtype=tf.float32),
    symbol_rate=tf.TensorSpec(shape=[], dtype=tf.float32)
)


class ElectricalSignal(tf.experimental.ExtensionType):
    #: Signal's timing information
    timing: Timing

    #: Signal's samples
    samples: tf.Tensor  # shape=[None]; dtype=tf.float32


electrical_signal_spec = ElectricalSignal.Spec(
    timing=timing_spec,
    samples=tf.TensorSpec(shape=[None], dtype=tf.float32)
)


class OpticalSignal(tf.experimental.ExtensionType):
    #: Signal's timing information
    timing: Timing

    #: Signal's frequencies (in Hz), for every frequency there's an amplitude
    frequencies: tf.Tensor  # shape=[None]; dtype=tf.float32

    #: Signal's samples
    samples: tf.Tensor  # shape=[None]; dtype=tf.complex64

    #: Power (maximum) of the optical signal in watts
    power: tf.Tensor  # shape=[]; dtype=tf.float32

    #: Central frequency of the of the emitted light in Hz
    central_frequency: tf.Tensor  # shape=[]; dtype=tf.float32


optical_signal_spec = OpticalSignal.Spec(
    timing=timing_spec,
    frequencies=tf.TensorSpec(shape=[None], dtype=tf.float32),
    samples=tf.TensorSpec(shape=[None, None], dtype=tf.complex64),
    power=tf.TensorSpec(shape=[], dtype=tf.float32),
    central_frequency=tf.TensorSpec(shape=[], dtype=tf.float32)
)

bit_dtype = tf.int32
bit_sequence_spec = tf.TensorSpec(shape=[None], dtype=bit_dtype)

IQSignal = tuple[ElectricalSignal, ElectricalSignal]
iq_signal_spec = (electrical_signal_spec, electrical_signal_spec)

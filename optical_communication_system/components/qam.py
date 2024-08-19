# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Any, Optional

import tensorflow as tf

from .datatypes import (
    bit_dtype, bit_sequence_spec, ElectricalSignal, IQSignal, Timing
)

__all__ = ["QAMConstellation", "QAMEncoder"]


class QAMConstellation(tf.Module):
    order: tf.Variable  # shape=(1,); dtype=int32
    mapping: tf.Variable  # shape=(2 ** order,2); dtype=complex64

    def __init__(
        self,
        *,
        order: int,
        mapping: Sequence[Sequence[float]],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.order = tf.Variable(tf.constant(order))
        self.mapping = tf.Variable(tf.constant(mapping))

    @tf.function(input_signature=(bit_sequence_spec,))
    def encode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        symbol_index: Any = tf.reduce_sum(
            symbol * 2 ** (self.order - tf.range(self.order) - 1)
        )

        return self.mapping[symbol_index]

    @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
    def decode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        best_index = tf.cast(
            tf.argmin(
                tf.reduce_sum(
                    tf.abs(self.mapping - symbol),
                    axis=1
                )
            ),
            tf.int32
        )

        return tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.math.floormod(best_index / 2 ** (self.order - i - 1), 2),
                bit_dtype
            ),
            elems=tf.range(self.order)
        )


class QAMEncoder(tf.Module):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: QAMConstellation

    #: Modulation order (number of bits per symbol)
    order: tf.Variable  # shape=(1,); dtype=int32

    def __init__(
        self,
        *,
        sampling_rate: float,
        data_rate: float,
        constellation: QAMConstellation,
        order: int = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.order = tf.Variable(tf.constant(order))
        self.constellation = constellation

    @tf.function(input_signature=(bit_sequence_spec,))
    def __call__(self, bitseq: tf.Tensor) -> IQSignal:
        tf.debugging.assert_equal(tf.math.floormod(tf.size(bitseq), self.order), 0)

        signal_duration = tf.divide(
            tf.cast(tf.size(bitseq), tf.float32),
            self.data_rate
        )

        time = tf.linspace(
            0.0,
            signal_duration,
            tf.cast(tf.round(self.sampling_rate * signal_duration), tf.int32)
        )

        iq_sequence = tf.vectorized_map(
            fn=lambda i: self.constellation.encode_symbol(bitseq[i * self.order:(i + 1) * self.order]),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), self.order))
        )

        timing = Timing(
            time=time,
            sampling_rate=self.sampling_rate,
            symbol_rate=tf.divide(self.data_rate, tf.cast(self.order, tf.float32))
        )

        repeat_count = tf.cast(
            tf.math.ceil(tf.divide(tf.size(time), tf.shape(iq_sequence)[0])),
            tf.int32
        )

        i_samples = tf.repeat(iq_sequence[:,0], repeat_count)[:tf.size(time)]
        q_samples = tf.repeat(iq_sequence[:,1], repeat_count)[:tf.size(time)]

        return IQSignal(
            i=ElectricalSignal(
                timing=timing,
                samples=i_samples
            ),
            q=ElectricalSignal(
                timing=timing,
                samples=q_samples
            ),
        )

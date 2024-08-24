# -*- coding: utf-8 -*-

from collections.abc import Sequence
import typing
from typing import Any, Optional
import tensorflow as tf

from .utils import adjust_symbol_for_mzm

from .datatypes import (
    bit_sequence_spec, bit_dtype,
    Timing,
    ElectricalSignal, electrical_signal_spec,
)

__all__ = [
    "PAMConstellation",
    "PAMEncoder",
    "PAMDecoder"
]


class PAMConstellation(tf.Module):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=(1,); dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,); dtype=float32

    #: If true, the constellation will be adjusted for MZM
    adjust_for_mzm: tf.Variable  # shape=(1,); dtype=bool

    def __init__(
        self,
        *,
        bits_per_symbol: int,
        mapping: Sequence[Sequence[float]],
        adjust_for_mzm: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol))

        if mapping is None:
            self.mapping = tf.Variable(
                typing.cast(tf.Tensor, self._default_pam_constellation())
            )
        else:
            self.mapping = tf.Variable(tf.constant(mapping))

        self.adjust_for_mzm = tf.Variable(tf.constant(adjust_for_mzm))

    @tf.function
    def _default_pam_constellation(self) -> tf.Tensor:
        return tf.vectorized_map(
            fn=lambda i: tf.cast(i / (tf.math.pow(2, self.bits_per_symbol) - 1), tf.float32),
            elems=tf.range(tf.math.pow(2, self.bits_per_symbol))
        )

    @tf.function(input_signature=(bit_sequence_spec,))
    def encode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        symbol_index: Any = tf.reduce_sum(
            symbol * 2 ** (self.bits_per_symbol - tf.range(self.bits_per_symbol) - 1)
        )

        if self.adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(self.mapping[symbol_index])
        else:
            return self.mapping[symbol_index]

    @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.complex64),))
    def decode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        best_index = tf.cast(tf.argmin(tf.abs(self.mapping - symbol)), tf.int32)

        return tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.math.floormod(best_index / 2 ** (self.bits_per_symbol - i - 1), 2),
                bit_dtype
            ),
            elems=tf.range(self.bits_per_symbol)
        )


class PAMEncoder(tf.Module):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: PAMConstellation

    def __init__(
        self,
        sampling_rate: float,
        data_rate: float,
        constellation: PAMConstellation,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.constellation = constellation

    @tf.function(input_signature=(bit_sequence_spec,))
    def __call__(self, bitseq: tf.Tensor) -> ElectricalSignal:
        bits_per_symbol = self.constellation.bits_per_symbol

        tf.debugging.assert_equal(tf.math.floormod(tf.size(bitseq), bits_per_symbol), 0)

        signal_duration = tf.divide(
            tf.cast(tf.size(bitseq), tf.float32),
            self.data_rate
        )

        time = tf.linspace(
            0.0,
            signal_duration,
            tf.cast(tf.round(self.sampling_rate * signal_duration), tf.int32)
        )

        pam_sequence = tf.vectorized_map(
            fn=lambda i: self.constellation.encode_symbol(
                bitseq[i * bits_per_symbol:(i + 1) * bits_per_symbol]
            ),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), bits_per_symbol))
        )

        return ElectricalSignal(
            timing=Timing(
                time=time,
                sampling_rate=self.sampling_rate,
                symbol_rate=tf.divide(self.data_rate, tf.cast(bits_per_symbol, tf.float32))
            ),
            samples=tf.repeat(
                pam_sequence,
                tf.cast(
                    tf.math.ceil(tf.divide(tf.size(time), tf.size(pam_sequence))),
                    tf.int32
                )
            )[:tf.size(time)]
        )


class PAMDecoder(tf.Module):
    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: PAMConstellation

    #: Minimum current (or power in watts) recognized as non-zero
    threshold: tf.Variable  # shape=(1,); dtype=float32

    #: Currents above this value will be clipped
    max_current: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        data_rate: float,
        constellation: PAMConstellation,
        threshold: float = 1e-5,
        max_current: float = 2e-3,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.constellation = constellation
        self.threshold = tf.Variable(tf.constant(threshold))
        self.max_current = tf.Variable(tf.constant(max_current))

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, signal: ElectricalSignal) -> tf.Tensor:
        bits_per_symbol = self.constellation.bits_per_symbol

        n = tf.cast(
            tf.round(signal.timing.sampling_rate / self.data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        current = tf.clip_by_value(tf.abs(signal.samples), 0, self.max_current)
        max_current = tf.math.maximum(tf.reduce_max(current), self.threshold)

        symbols = tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.reduce_mean(current[i * n:(i + 1) * n]) / max_current,
                tf.complex64
            ),
            elems=tf.range(tf.cast(tf.size(current) / n, tf.int32))
        )

        return tf.reshape(
            tf.vectorized_map(
                fn=self.constellation.decode_symbol,
                elems=symbols
            ),
            (tf.size(symbols) * bits_per_symbol,)
        )

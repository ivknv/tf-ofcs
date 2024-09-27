# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Any, Optional
import tensorflow as tf

from .utils import adjust_symbol_for_mzm, index_to_bits, bits_to_index, pam_constellation
from .component import Component

from .datatypes import (
    bit_sequence_spec,
    Timing,
    ElectricalSignal, electrical_signal_spec,
)

__all__ = [
    "PAMSymbolEncoder",
    "PAMSymbolDecoder",
    "PAMEncoder",
    "PAMDecoder"
]


class PAMSymbolEncoder(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[None]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,); dtype=float32

    #: If true, the constellation will be adjusted for MZM
    adjust_for_mzm: tf.Variable  # shape=[]; dtype=bool

    def __init__(
        self,
        *,
        bits_per_symbol: int,
        mapping: Optional[Sequence[float]] = None,
        adjust_for_mzm: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol))

        if mapping is None:
            mapping = pam_constellation(bits_per_symbol)

        self.mapping = tf.Variable(tf.constant(mapping))
        self.adjust_for_mzm = tf.Variable(tf.constant(adjust_for_mzm))

        self.parameters = ["mapping", "adjust_for_mzm"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    ))
    def implementation(
        self,
        bitseq: tf.Tensor,
        mapping: tf.Tensor,
        adjust_for_mzm: tf.Tensor
    ) -> tf.Tensor:
        symbol_index: Any = bits_to_index(bitseq)

        if adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(mapping[symbol_index])
        else:
            return mapping[symbol_index]


class PAMSymbolDecoder(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[None]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,); dtype=float32

    def __init__(
        self,
        *,
        bits_per_symbol: int,
        mapping: Optional[Sequence[float]] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol))

        if mapping is None:
            mapping = pam_constellation(bits_per_symbol)

        self.mapping = tf.Variable(tf.constant(mapping))

        self.parameters = ["bits_per_symbol", "mapping"]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
    ))
    def implementation(
        self,
        symbol: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor,
    ) -> tf.Tensor:
        best_index = tf.cast(tf.argmin(tf.abs(mapping - symbol)), tf.int32)

        return index_to_bits(best_index, bits_per_symbol)


class PAMEncoder(Component):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    #: Constellation used for encoding
    symbol_encoder: PAMSymbolEncoder

    def __init__(
        self,
        sampling_rate: float,
        data_rate: float,
        symbol_encoder: PAMSymbolEncoder,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.symbol_encoder = symbol_encoder

        self.parameters = ["sampling_rate", "data_rate"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ))
    def implementation(
        self,
        bitseq: tf.Tensor,
        sampling_rate: tf.Tensor,
        data_rate: tf.Tensor
    ) -> ElectricalSignal:
        bits_per_symbol = self.symbol_encoder.bits_per_symbol

        tf.debugging.assert_equal(tf.math.floormod(tf.size(bitseq), bits_per_symbol), 0)

        signal_duration = tf.divide(
            tf.cast(tf.size(bitseq), tf.float32),
            data_rate
        )

        time = tf.linspace(
            0.0,
            signal_duration,
            tf.cast(tf.round(sampling_rate * signal_duration), tf.int32)
        )

        pam_sequence = tf.vectorized_map(
            fn=lambda i: self.symbol_encoder(
                bitseq[i * bits_per_symbol:(i + 1) * bits_per_symbol]
            ),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), bits_per_symbol))
        )

        return ElectricalSignal(
            timing=Timing(
                time=time,
                sampling_rate=sampling_rate,
                symbol_rate=tf.divide(data_rate, tf.cast(bits_per_symbol, tf.float32))
            ),
            samples=tf.repeat(
                pam_sequence,
                tf.cast(
                    tf.math.ceil(tf.divide(tf.size(time), tf.size(pam_sequence))),
                    tf.int32
                )
            )[:tf.size(time)]
        )


class PAMDecoder(Component):
    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    #: Constellation used for encoding
    symbol_decoder: PAMSymbolDecoder

    #: Minimum current (or power in watts) recognized as non-zero
    threshold: tf.Variable  # shape=[]; dtype=float32

    #: Received symbols will be scaled by this factor (necessary for M-ary modulation formats)
    symbol_scale: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        data_rate: float,
        symbol_decoder: PAMSymbolDecoder,
        threshold: float = 1e-5,
        symbol_scale: float = 1.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.symbol_decoder = symbol_decoder
        self.threshold = tf.Variable(tf.constant(threshold))
        self.symbol_scale = tf.Variable(tf.constant(symbol_scale))

        self.parameters = ["data_rate", "threshold", "symbol_scale"]

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ))
    def _extract_pam_symbols(
        self,
        signal: ElectricalSignal,
        data_rate: tf.Tensor,
        threshold: tf.Tensor,
        symbol_scale: tf.Tensor,
        bits_per_symbol: tf.Tensor
    ) -> tf.Tensor:
        n = tf.cast(
            tf.round(signal.timing.sampling_rate / data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        current = tf.math.maximum(tf.abs(signal.samples), threshold)

        symbols = tf.vectorized_map(
            fn=lambda i: tf.cast(
                current[i * n + n // 2],
                tf.float32
            ),
            elems=tf.range(tf.cast(tf.size(current) // n, tf.int32))
        ) / symbol_scale

        return symbols

    @tf.function
    def __call__(self, signal: ElectricalSignal, **kwargs) -> tf.Tensor:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        threshold = self.get_component_parameter(kwargs, "threshold")
        symbol_scale = self.get_component_parameter(kwargs, "symbol_scale")
        bits_per_symbol = self.get_component_parameter(kwargs, "bits_per_symbol")

        symbols = self._extract_pam_symbols(
            signal, data_rate, threshold, symbol_scale, bits_per_symbol
        )

        return tf.reshape(
            tf.vectorized_map(
                fn=lambda symbol: self.symbol_decoder(symbol, **kwargs),
                elems=symbols
            ),
            (tf.size(symbols) * bits_per_symbol,)
        )

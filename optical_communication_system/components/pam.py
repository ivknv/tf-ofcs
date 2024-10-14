# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from .utils import adjust_symbol_for_mzm, indices_to_bits, bits_to_indices, pam_constellation
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
        bits_per_symbol: int | tf.Tensor,
        mapping: Optional[Sequence[float]] | tf.Tensor = None,
        adjust_for_mzm: bool | tf.Tensor = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol))

        if mapping is None:
            mapping = pam_constellation(int(bits_per_symbol))

        self.mapping = tf.Variable(tf.constant(mapping))
        self.adjust_for_mzm = tf.Variable(tf.constant(adjust_for_mzm))

        self.parameters = ["mapping", "bits_per_symbol", "adjust_for_mzm"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    ))
    def implementation(
        self,
        bitseq: tf.Tensor,
        mapping: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        adjust_for_mzm: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(bitseq)[0]
        bit_count = tf.shape(bitseq)[1]

        symbol_indices = bits_to_indices(bitseq, bits_per_symbol)

        symbols = tf.reshape(
            tf.gather(mapping, symbol_indices),
            (batch_size, bit_count // bits_per_symbol)
        )

        if adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(symbols)
        else:
            return symbols


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
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
    ))
    def implementation(
        self,
        symbols: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor,
    ) -> tf.Tensor:
        batch_size = tf.shape(symbols)[0]
        symbol_count = tf.shape(symbols)[1]

        best_indices = tf.cast(
            tf.argmin(
                tf.abs(mapping - tf.expand_dims(symbols, -1)),
                axis=-1
            ),
            tf.int32
        )

        return tf.reshape(
            indices_to_bits(
                tf.reshape(best_indices, (tf.size(best_indices),)),
                bits_per_symbol
            ),
            (batch_size, symbol_count, bits_per_symbol)
        )


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

    @tf.function
    def __call__(self, bitseq: tf.Tensor, **kwargs) -> ElectricalSignal:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        sampling_rate = self.get_component_parameter(kwargs, "sampling_rate")
        bits_per_symbol = self.symbol_encoder.get_component_parameter(kwargs, "bits_per_symbol")

        bit_count = tf.shape(bitseq)[1]

        tf.debugging.assert_equal(tf.math.floormod(bit_count, bits_per_symbol), 0)

        symbol_rate = data_rate / tf.cast(bits_per_symbol, tf.float32)

        time = tf.cast(
            tf.range(
                tf.multiply(
                    bit_count,
                    tf.cast(tf.round(sampling_rate / data_rate), tf.int32)
                )
            ),
            tf.float32
        ) / sampling_rate

        pam_sequence = self.symbol_encoder(bitseq)

        return ElectricalSignal(
            timing=Timing(
                time=time,
                sampling_rate=sampling_rate,
                symbol_rate=symbol_rate
            ),
            samples=tf.repeat(
                pam_sequence,
                tf.cast(
                    tf.math.ceil(tf.divide(tf.size(time), tf.shape(pam_sequence)[1])),
                    tf.int32
                ),
                axis=1
            )[:,:tf.size(time)]
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
        symbol_scale: float = 1.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.symbol_decoder = symbol_decoder
        self.symbol_scale = tf.Variable(tf.constant(symbol_scale))

        self.parameters = ["data_rate", "symbol_scale"]

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ))
    def _extract_pam_symbols(
        self,
        signal: ElectricalSignal,
        data_rate: tf.Tensor,
        symbol_scale: tf.Tensor,
        bits_per_symbol: tf.Tensor
    ) -> tf.Tensor:
        n = tf.cast(
            tf.round(signal.timing.sampling_rate / data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        indices = n * tf.range(tf.shape(signal.samples)[1] // n) + n // 2
        symbols = tf.abs(tf.gather(signal.samples, indices, axis=1)) / symbol_scale

        return symbols

    @tf.function
    def __call__(self, signal: ElectricalSignal, **kwargs) -> tf.Tensor:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        symbol_scale = self.get_component_parameter(kwargs, "symbol_scale")
        bits_per_symbol = self.symbol_decoder.get_component_parameter(kwargs, "bits_per_symbol")

        symbols = self._extract_pam_symbols(
            signal, data_rate, symbol_scale, bits_per_symbol
        )

        batch_size = tf.shape(symbols)[0]
        symbol_count = tf.shape(symbols)[1]

        return tf.reshape(
            self.symbol_decoder(symbols, **kwargs),
            (batch_size, symbol_count * bits_per_symbol,)
        )

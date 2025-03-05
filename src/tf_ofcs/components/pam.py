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

from typing import Optional
import tensorflow as tf

from ..utils import (
    adjust_symbol_for_mzm, indices_to_bits, bits_to_indices, pam_constellation,
    to_tensor
)
from .component import Component

from ..datatypes import (
    bit_sequence_spec,
    Timing,
    ElectricalSignal,
    electrical_signal_spec,
    TensorCompatible
)

__all__ = [
    "PAMSymbolEncoder",
    "PAMSymbolDecoder",
    "PAMEncoder",
    "PAMDecoder"
]


class PAMSymbolEncoder(Component):
    """
        This component encodes a bit sequence into a sequence of PAM symbols.

        :ivar bits_per_symbol: `shape=[], dtype=tf.int32`, number of bits per symbol
        :ivar mapping: `shape=[2 ** bits_per_symbol], dtype=float32`, mapping of symbol indices to PAM symbols
        :ivar adjust_for_mzm: `shape=[], dtype=tf.bool`, if true, the constellation will be adjusted for MZM
    """

    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=[2 ** bits_per_symbol]; dtype=float32

    #: If true, the constellation will be adjusted for MZM
    adjust_for_mzm: tf.Variable  # shape=[]; dtype=bool

    def __init__(
        self,
        *,
        bits_per_symbol: int | tf.Tensor,
        mapping: Optional[TensorCompatible] = None,
        adjust_for_mzm: bool | tf.Tensor = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(to_tensor(bits_per_symbol), name=f"{self.name}__bits_per_symbol")

        if mapping is None:
            self.mapping = tf.Variable(pam_constellation(self.bits_per_symbol), name=f"{self.name}__mapping")
        else:
            self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")

        self.adjust_for_mzm = tf.Variable(to_tensor(adjust_for_mzm), name=f"{self.name}__adjust_for_mzm")

        self.local_parameters = ["mapping", "bits_per_symbol", "adjust_for_mzm"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="mapping"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="bits_per_symbol"),
        tf.TensorSpec(shape=[], dtype=tf.bool, name="adjust_for_mzm")
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

        return tf.cond(
            adjust_for_mzm,
            lambda: adjust_symbol_for_mzm(symbols),
            lambda: symbols
        )


class PAMSymbolDecoder(Component):
    """
        This component decodes a sequence of PAM symbols into a bit sequence.

        :ivar bits_per_symbol: `shape=[], dtype=tf.int32`, number of bits per symbol
        :ivar mapping: `shape=[2 ** bits_per_symbol], dtype=float32`, mapping symbol indices to PAM symbols
    """

    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,); dtype=float32

    def __init__(
        self,
        *,
        bits_per_symbol: TensorCompatible,
        mapping: Optional[TensorCompatible] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(to_tensor(bits_per_symbol), name=f"{self.name}__bits_per_symbol")

        if mapping is None:
            self.mapping = tf.Variable(pam_constellation(self.bits_per_symbol) ** 2, name=f"{self.name}__mapping")
        else:
            self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")

        self.local_parameters = ["bits_per_symbol", "mapping"]

    def implementation(
        self,
        symbols: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(symbols)[0]
        symbol_count = tf.shape(symbols)[-1]

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
    """
        Encodes a bit sequence into an electrical signal using Pulse Amplitude Modulation (PAM).

        :ivar sampling_rate: `shape=[], dtype=tf.float32`, sampling rate in Hz
        :ivar data_rate: `shape=[], dtype=tf.float32`, data rate in bits/second
        :ivar symbol_encoder: `PAMSymbolEncoder`, constellation used for encoding
    """

    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    #: Constellation used for encoding
    symbol_encoder: PAMSymbolEncoder

    def __init__(
        self,
        sampling_rate: TensorCompatible,
        data_rate: TensorCompatible,
        symbol_encoder: PAMSymbolEncoder,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(to_tensor(sampling_rate), name=f"{self.name}__sampling_rate")
        self.data_rate = tf.Variable(to_tensor(data_rate), name=f"{self.name}__data_rate")
        self.symbol_encoder = symbol_encoder

        self.local_parameters = ["sampling_rate", "data_rate"]

    def call(self, bitseq: tf.Tensor, /, tracing_level: int = 0, **kwargs) -> ElectricalSignal:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        sampling_rate = self.get_component_parameter(kwargs, "sampling_rate")
        bits_per_symbol = self.symbol_encoder.get_component_parameter(kwargs, "bits_per_symbol")

        bit_count = tf.shape(bitseq)[1]

        tf.debugging.assert_equal(tf.math.floormod(bit_count, bits_per_symbol), 0)

        sample_count = tf.multiply(
            bit_count,
            tf.cast(tf.round(sampling_rate / data_rate), tf.int32)
        )

        pam_sequence = self.symbol_encoder(bitseq, tracing_level=tracing_level + 1, **kwargs)

        return ElectricalSignal(
            timing=Timing(
                sample_count=sample_count,
                sampling_rate=sampling_rate
            ),
            samples=tf.repeat(
                pam_sequence,
                tf.cast(
                    tf.math.ceil(tf.divide(sample_count, tf.shape(pam_sequence)[-1])),
                    tf.int32
                ),
                axis=-1
            )[:,:sample_count]
        )


class PAMDecoder(Component):
    """
        Decodes an electrical signal into a bit sequence using Pulse Amplitude Modulation (PAM).

        :ivar data_rate: `shape=[], dtype=tf.float32`, data rate in bits/second
        :ivar symbol_decoder: `PAMSymbolDecoder`, constellation used for decoding
    """

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    #: Constellation used for encoding
    symbol_decoder: PAMSymbolDecoder

    def __init__(
        self,
        data_rate: TensorCompatible,
        symbol_decoder: PAMSymbolDecoder,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(to_tensor(data_rate), name=f"{self.name}__data_rate")
        self.symbol_decoder = symbol_decoder

        self.local_parameters = ["data_rate"]

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[], dtype=tf.float32, name="data_rate"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="bits_per_symbol"),
    ))
    def _extract_pam_symbols(
        self,
        signal: ElectricalSignal,
        data_rate: tf.Tensor,
        bits_per_symbol: tf.Tensor
    ) -> tf.Tensor:
        n = tf.cast(
            tf.round(signal.timing.sampling_rate / data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        indices = n * tf.range(tf.shape(signal.samples)[-1] // n) + n // 2
        symbols = tf.abs(tf.gather(signal.samples, indices, axis=-1))

        return symbols

    def call(self, signal: ElectricalSignal, /, tracing_level: int = 0, **kwargs) -> tf.Tensor:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        bits_per_symbol = self.symbol_decoder.get_component_parameter(kwargs, "bits_per_symbol")

        symbols = self._extract_pam_symbols(
            signal, data_rate, bits_per_symbol
        )

        batch_size = tf.shape(symbols)[0]
        symbol_count = tf.shape(symbols)[-1]

        return tf.reshape(
            self.symbol_decoder(symbols, tracing_level=tracing_level + 1, **kwargs),
            (batch_size, symbol_count * bits_per_symbol,)
        )

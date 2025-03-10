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

from ..utils import indices_to_bits, bits_to_indices, pam_constellation, to_tensor
from .component import Component

from ..datatypes import Signal, Timing, TensorCompatible

__all__ = ["PAMSymbolEncoder", "PAMSymbolDecoder"]


class PAMSymbolEncoder(Component):
    """
        This component encodes a bit sequence into a sequence of PAM symbols (`Signal`).

        :ivar bits_per_symbol: `shape=[], dtype=tf.int32`, number of bits per symbol
        :ivar mapping: `shape=[2 ** bits_per_symbol], dtype=tf.float32`, mapping of symbol indices to PAM symbols
        :ivar symbol_rate: `shape=[], dtype=tf.float32`, symbol rate in symbols/second
    """

    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=[2 ** bits_per_symbol]; dtype=tf.float32
    symbol_rate: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        *,
        bits_per_symbol: TensorCompatible,
        symbol_rate: TensorCompatible,
        mapping: Optional[TensorCompatible] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(to_tensor(bits_per_symbol), name=f"{self.name}__bits_per_symbol")
        self.symbol_rate = tf.Variable(to_tensor(symbol_rate), name=f"{self.name}__symbol_rate")

        if mapping is None:
            self.mapping = tf.Variable(pam_constellation(self.bits_per_symbol), name=f"{self.name}__mapping")
        else:
            self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")

        self.local_parameters = ["mapping", "bits_per_symbol", "symbol_rate"]

    def implementation(
        self,
        bitseq: tf.Tensor,
        /,
        mapping: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        symbol_rate: tf.Tensor
    ) -> Signal:
        batch_size = tf.shape(bitseq)[0]
        bit_count = tf.shape(bitseq)[1]

        symbol_indices = bits_to_indices(bitseq, bits_per_symbol)
        symbol_count_per_batch = bit_count // bits_per_symbol

        return Signal(
            samples=tf.reshape(
                tf.gather(mapping, symbol_indices),
                (batch_size, symbol_count_per_batch)
            ),
            timing=Timing(
                sample_count=symbol_count_per_batch,
                sampling_rate=symbol_rate
            )
        )


class PAMSymbolDecoder(Component):
    """
        This component decodes a sequence of PAM symbols (`Signal`) into a bit sequence.

        :ivar bits_per_symbol: `shape=[], dtype=tf.int32`, number of bits per symbol
        :ivar mapping: `shape=[2 ** bits_per_symbol], dtype=tf.float32`, mapping symbol indices to PAM symbols
    """

    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=tf.int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,); dtype=tf.float32

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
            self.mapping = tf.Variable(pam_constellation(self.bits_per_symbol), name=f"{self.name}__mapping")
        else:
            self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")

        self.local_parameters = ["bits_per_symbol", "mapping"]

    def implementation(
        self,
        digital_signal: Signal,
        /,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(digital_signal.samples)[0]
        symbol_count = tf.shape(digital_signal.samples)[-1]

        best_indices = tf.cast(
            tf.argmin(
                tf.abs(mapping - tf.expand_dims(digital_signal.samples, -1)),
                axis=-1
            ),
            tf.int32
        )

        return tf.reshape(
            indices_to_bits(
                tf.reshape(best_indices, (tf.size(best_indices),)),
                bits_per_symbol
            ),
            (batch_size, symbol_count * bits_per_symbol)
        )

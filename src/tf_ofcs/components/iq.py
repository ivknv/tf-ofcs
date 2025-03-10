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
import math

import tensorflow as tf

from ..datatypes import (
    TensorCompatible, bit_dtype,
    ElectricalSignal,
    IQSignal,
    Timing
)

from .component import Component
from .mzm import MachZehnderModulator
from .passive import OpticalCoupler1x2, OpticalCoupler2x1, OpticalCoupler2x2, OpticalPhaseShifter
from .utils import Composition, Fork, Identity, MapComponents, Reindex
from ..utils import adjust_symbol_for_mzm, indices_to_bits, bits_to_indices, to_tensor
from .electrical import ElectricalSubtractor

__all__ = ["IQSymbolEncoder", "IQSymbolDecoder", "IQEncoder", "IQModulator", "IQReceiver"]


class IQSymbolEncoder(Component):
    """
        Encodes a sequence of bits to a sequence of IQ symbols.

        :ivar bits_per_symbol: `shape=[]; dtype=tf.int32`, number of bits per symbol
        :ivar mapping: `shape=[2 ** bits_per_symbol, 2]; dtype=float32`, mapping of symbol indices to IQ symbols.
        :ivar adjust_for_mzm: `shape=[]; dtype=bool`, if true, the constellation will be adjusted for MZM
    """

    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol, 2); dtype=float32

    #: If true, the constellation will be adjusted for MZM
    adjust_for_mzm: tf.Variable  # shape=[]; dtype=bool

    def __init__(
        self,
        *,
        bits_per_symbol: int | tf.Tensor,
        mapping: TensorCompatible,
        adjust_for_mzm: TensorCompatible = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(to_tensor(bits_per_symbol), name=f"{self.name}__bits_per_symbol")

        self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")
        self.adjust_for_mzm = tf.Variable(to_tensor(adjust_for_mzm), name=f"{self.name}__adjust_for_mzm")

        self.local_parameters = [
            "mapping", "bits_per_symbol", "adjust_for_mzm"
        ]

    def implementation(
        self,
        bitseq: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor,
        adjust_for_mzm: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(bitseq)[0]
        bit_count = tf.shape(bitseq)[1]

        symbol_indices = bits_to_indices(bitseq, bits_per_symbol)

        symbols = tf.reshape(
            tf.gather(mapping, symbol_indices),
            (batch_size, bit_count // bits_per_symbol, 2)
        )

        return tf.cond(
            adjust_for_mzm,
            lambda: adjust_symbol_for_mzm(symbols),
            lambda: symbols
        )


class IQSymbolDecoder(Component):
    """
        Decodes a sequence of IQ symbols to a sequence of bits.

        :ivar bits_per_symbol: `shape=[]; dtype=tf.int32`, number of bits per symbol
        :ivar mapping: `shape=[2 ** bits_per_symbol, 2]; dtype=float32`, mapping of symbol indices to IQ symbols.

        Input symbols should have `shape=[batch_size, symbol_count, 2]; dtype=float32`.
    """

    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol, 2); dtype=float32

    def __init__(
        self,
        *,
        bits_per_symbol: TensorCompatible,
        mapping: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(to_tensor(bits_per_symbol), name=f"{self.name}__bits_per_symbol")
        self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")

        self.local_parameters = ["bits_per_symbol", "mapping"]

    def implementation(
        self,
        symbols: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(symbols)[0]
        symbol_count = tf.shape(symbols)[-2]

        best_indices = tf.cast(
            tf.argmin(
                tf.math.reduce_sum(tf.abs(mapping - tf.expand_dims(symbols, -2)), axis=-1),
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


class IQEncoder(Component):
    """
        Encodes a sequence of bits to a sequence of IQ symbols and then converts
        it to a tuple of 2 electrical signals.

        :ivar sampling_rate: `shape=[]; dtype=float32`, sampling rate in Hz
        :ivar data_rate: `shape=[]; dtype=float32`, data rate in bits/second
        :ivar symbol_encoder: :class:`IQSymbolEncoder`, symbol encoder
    """

    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    symbol_encoder: IQSymbolEncoder

    def __init__(
        self,
        *,
        sampling_rate: TensorCompatible,
        data_rate: TensorCompatible,
        symbol_encoder: IQSymbolEncoder,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(to_tensor(sampling_rate), name=f"{self.name}__sampling_rate")
        self.data_rate = tf.Variable(to_tensor(data_rate), name=f"{self.name}__data_rate")
        self.symbol_encoder = symbol_encoder

        self.local_parameters = ["sampling_rate", "data_rate"]

    def call(self, bitseq: tf.Tensor, /, tracing_level: int = 0, **kwargs) -> IQSignal:
        assert isinstance(bitseq, tf.Tensor)
        assert bitseq.dtype is bit_dtype

        data_rate = self.get_component_parameter(kwargs, "data_rate")
        sampling_rate = self.get_component_parameter(kwargs, "sampling_rate")
        bits_per_symbol = self.symbol_encoder.get_component_parameter(kwargs, "bits_per_symbol")

        bit_count = tf.shape(bitseq)[-1]

        tf.debugging.assert_equal(tf.math.floormod(bit_count, bits_per_symbol), 0)

        sample_count = tf.multiply(
            bit_count,
            tf.cast(tf.round(sampling_rate / data_rate), tf.int32)
        )

        iq_sequences = self.symbol_encoder(bitseq, tracing_level=tracing_level + 1, **kwargs)

        timing = Timing(
            sample_count=sample_count,
            sampling_rate=sampling_rate
        )

        repeat_count = tf.cast(
            tf.math.ceil(tf.divide(sample_count, tf.shape(iq_sequences)[-2])),
            tf.int32
        )

        i_samples = tf.repeat(iq_sequences[:,:,0], repeat_count, axis=-1)[:,:sample_count]
        q_samples = tf.repeat(iq_sequences[:,:,1], repeat_count, axis=-1)[:,:sample_count]

        return (
            ElectricalSignal(timing=timing, samples=i_samples),
            ElectricalSignal(timing=timing, samples=q_samples)
        )


class IQModulator(Composition):
    """
        IQ modulator. Expects a `tuple[OpticalSignal, (ElectricalSignal, ElectricalSignal)]`
        as input. The first tuple element is the carrier signal, the second is
        the modulation signal.
    """

    def __init__(self, *, mzm: Optional[Component] = None, name: Optional[str] = None) -> None:
        super().__init__([], name=name)

        if mzm is None:
            mzm = MachZehnderModulator(name=f"{self.name}__mzm")

        identity = Identity(name=f"{self.name}__identity")

        self.components = [
            # input: (OpticalSignal, (ElectricalSignal, ElectricalSignal))
            MapComponents([
                Composition([
                    OpticalCoupler1x2(name=f"{self.name}__coupler1x2"),  # (OpticalSignal, OpticalSignal)
                    MapComponents([
                        identity,
                        OpticalPhaseShifter(phase_shift=[0.5 * math.pi], name=f"{self.name}__phase_shifter")
                    ])
                ]),  # (OpticalSignal, OpticalSignal)
                identity  # (ElectricalSignal, ElectricalSignal)
            ]),  # ((OpticalSignal 0, OpticalSignal 1), (ElectricalSignal 0, ElectricalSignal 1))

            Reindex(indices=[[(0, 0), (1, 0)], [(0, 1), (1, 1)]], name=f"{self.name}__reindex"),
            # reindex output: ((OpticalSignal 0, ElectricalSignal 0), (OpticalSignal 1, ElectricalSignal 1))

            MapComponents(mzm),  # (OpticalSignal, OpticalSignal)
            OpticalCoupler2x1(name=f"{self.name}__coupler_2x1")  # OpticalSignal
        ]


class IQReceiver(Composition):
    """
        Coherent IQ receiver. Expects an optical signal as input and outputs
        a tuple of 2 electrical signals.

        :ivar photodiode: :class:`Component`, photodiode for the coherent receiver
        :ivar local_oscillator: :class:`Component`, local oscillator (laser) for the coherent receiver
    """

    #: Photodiode for the coherent receiver
    photodiode: Component

    #: Local oscillator (laser) for the coherent receiver
    local_oscillator: Component

    def __init__(
        self,
        photodiode: Component,
        local_oscillator: Component,
        name: Optional[str] = None
    ) -> None:
        super().__init__([], name=name)

        self.photodiode = photodiode
        self.local_oscillator = local_oscillator

        coupler1x2 = OpticalCoupler1x2(name=f"{self.name}__optical_coupler1x2")
        identity = Identity(f"{self.name}__identity")

        # This is a coherent IQ receiver
        self.components = [
            coupler1x2,  # (OpticalSignal, OpticalSignal)
            Fork(shape=[2], name=f"{self.name}__fork"),  # 2 * ((OpticalSignal, OpticalSignal))
            MapComponents(
                [
                    identity,  # (OpticalSignal, OpticalSignal)
                    Composition(
                        components=[
                            Reindex(indices=(0,), name=f"{self.name}__reindex1"),  # OpticalSignal
                            local_oscillator,  # OpticalSignal
                            coupler1x2,  # (OpticalSignal, OpticalSignal)
                            MapComponents(
                                [
                                    OpticalPhaseShifter(
                                        phase_shift=[math.pi / 2],
                                        name=f"{self.name}__optical_phase_shifter"
                                    ),  # OpticalSignal
                                    identity  # OpticalSignal
                                ],
                                name=f"{self.name}__map_components2"
                            )  # (OpticalSignal, OpticalSignal)
                        ],
                        name=f"{self.name}__composition1"
                    )
                ],
                name=f"{self.name}__map_components1"
            ),  # 2 * ((OpticalSignal, OpticalSignal))
            Reindex(
                indices=[[(1, 1), (0, 0)], [(0, 1), (1, 0)]],
                name=f"{self.name}__reindex2"
            ),  # 2 * ((OpticalSignal, OpticalSignal),)
            MapComponents(
                Composition(
                    [
                        OpticalCoupler2x2(
                            flip=True,
                            name=f"{self.name}__optical_coupler2x2"
                        ),  # (OpticalSignal, OpticalSignal)
                        MapComponents(
                            photodiode,
                            name=f"{self.name}__map_components4"
                        ),  # (ElectricalSignal, ElectricalSignal)
                        ElectricalSubtractor(name=f"{self.name}__electrical_subtractor")  # ElectricalSignal
                    ],
                    name=f"{self.name}__composition2"
                ),
                name=f"{self.name}__map_components3"
            ),  # (ElectricalSignal, ElectricalSignal)
        ]


class IQDecoder(Component):
    """
        Decodes an IQ signal in the form of a tuple of 2 electrical signals to
        a sequence of bits.

        :ivar data_rate: `shape=[]; dtype=float32`, data rate in bits/second
        :ivar symbol_decoder: :class:`IQSymbolDecoder`, symbol decoder
    """

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    symbol_decoder: IQSymbolDecoder

    def __init__(
        self,
        *,
        data_rate: TensorCompatible,
        symbol_decoder: IQSymbolDecoder,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(to_tensor(data_rate), name=f"{self.name}__data_rate")
        self.symbol_decoder = symbol_decoder

        self.local_parameters = ["data_rate"]

    def _extract_iq_symbols(
        self,
        iq_signal: IQSignal,
        bits_per_symbol: tf.Tensor,
        data_rate: tf.Tensor
    ) -> tf.Tensor:
        i_samples = iq_signal[0].samples  # shape=[batch_size, sample_count]
        q_samples = iq_signal[1].samples  # shape=[batch_size, sample_count]

        batch_size = tf.shape(i_samples)[0]
        total_sample_count = tf.size(i_samples) + tf.size(q_samples)

        iq_samples = tf.reshape(
            tf.stack([
                tf.reshape(i_samples, [tf.size(i_samples)]),
                tf.reshape(q_samples, [tf.size(q_samples)])
            ], axis=-1),
            (batch_size, total_sample_count // batch_size // 2, 2)
        )

        sampling_rate = iq_signal[0].timing.sampling_rate

        n = tf.cast(
            tf.round(sampling_rate / data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        indices = n * tf.range(tf.shape(iq_samples)[-2] // n) + n // 2
        iq_symbols = tf.math.real(tf.gather(iq_samples, indices, axis=-2))

        return iq_symbols

    def call(self, iq_signal: IQSignal, /, tracing_level: int = 0, **kwargs) -> tf.Tensor:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        bits_per_symbol = self.symbol_decoder.get_component_parameter(kwargs, "bits_per_symbol")

        batch_size = tf.shape(iq_signal[0].samples)[0]

        # shape=(batch_size, symbol_count, 2)
        iq_symbols = self._extract_iq_symbols(iq_signal, bits_per_symbol, data_rate)
        symbol_count = tf.shape(iq_symbols)[-2]

        return tf.reshape(
            self.symbol_decoder(iq_symbols, tracing_level=tracing_level + 1, **kwargs),
            (batch_size, symbol_count * bits_per_symbol)
        )

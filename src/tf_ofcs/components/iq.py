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

from collections.abc import Sequence
from typing import Optional
import math

import tensorflow as tf

from ..datatypes import (
    bit_dtype, bit_sequence_spec,
    ElectricalSignal,
    IQSignal, iq_signal_spec,
    Timing
)

from .component import Component
from .mzm import MachZehnderModulator
from .passive import OpticalCoupler1x2, OpticalCoupler2x1, OpticalCoupler2x2, OpticalPhaseShifter
from .utils import Composition, Fork, Identity, MapComponents, Reindex
from ..utils import adjust_symbol_for_mzm, indices_to_bits, bits_to_indices
from .electrical import ElectricalSubtractor

__all__ = ["IQSymbolEncoder", "IQSymbolDecoder", "IQEncoder", "IQModulator", "IQReceiver"]


class IQSymbolEncoder(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol, 2); dtype=float32

    #: If true, the constellation will be adjusted for MZM
    adjust_for_mzm: tf.Variable  # shape=[]; dtype=bool

    parameter_dims: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        *,
        bits_per_symbol: int | tf.Tensor,
        mapping: Sequence[Sequence[float]] | tf.Tensor,
        adjust_for_mzm: bool | tf.Tensor = False,
        parameter_dims: int | tf.Tensor = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol), name=f"{self.name}__bits_per_symbol")

        self.mapping = tf.Variable(tf.constant(mapping), name=f"{self.name}__mapping")
        self.adjust_for_mzm = tf.Variable(tf.constant(adjust_for_mzm), name=f"{self.name}__adjust_for_mzm")
        self.parameter_dims = tf.Variable(tf.constant(parameter_dims), name=f"{self.name}__parameter_dims")

        self.local_parameters = [
            "mapping", "bits_per_symbol", "adjust_for_mzm", "parameter_dims"
        ]
        self.global_parameters = ["parameter_dims"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        tf.TensorSpec(shape=[], dtype=tf.int32, name="bits_per_symbol"),
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="mapping"),
        tf.TensorSpec(shape=[], dtype=tf.bool, name="adjust_for_mzm"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="parameter_dims")
    ))
    def implementation(
        self,
        bitseq: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor,
        adjust_for_mzm: tf.Tensor,
        parameter_dims: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(bitseq)[0]
        bit_count = tf.shape(bitseq)[1]

        symbol_indices = bits_to_indices(bitseq, bits_per_symbol)

        symbols = tf.tile(
            tf.reshape(
                tf.gather(mapping, symbol_indices),
                (batch_size, 1, bit_count // bits_per_symbol, 2)
            ),
            [1, parameter_dims, 1, 1]
        )

        if adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(symbols)
        else:
            return symbols


class IQSymbolDecoder(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol, 2); dtype=float32

    def __init__(
        self,
        *,
        bits_per_symbol: int | tf.Tensor,
        mapping: Sequence[Sequence[float]] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol), name=f"{self.name}__bits_per_symbol")
        self.mapping = tf.Variable(tf.constant(mapping), name=f"{self.name}__mapping")

        self.local_parameters = ["bits_per_symbol", "mapping"]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, None, 2], dtype=tf.float32, name="symbols"),
        tf.TensorSpec(shape=[], dtype=tf.int32, name="bits_per_symbol"),
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="mapping")
    ))
    def implementation(
        self,
        symbols: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(symbols)[0]
        parameter_dims = tf.shape(symbols)[1]
        symbol_count = tf.shape(symbols)[2]

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
            (batch_size, parameter_dims, symbol_count, bits_per_symbol)
        )


class IQEncoder(Component):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    symbol_encoder: IQSymbolEncoder

    def __init__(
        self,
        *,
        sampling_rate: float | tf.Tensor,
        data_rate: float | tf.Tensor,
        symbol_encoder: IQSymbolEncoder,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate), name=f"{self.name}__sampling_rate")
        self.data_rate = tf.Variable(tf.constant(data_rate), name=f"{self.name}__data_rate")
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

        i_samples = tf.repeat(iq_sequences[:,:,:,0], repeat_count, axis=-1)[:,:,:sample_count]
        q_samples = tf.repeat(iq_sequences[:,:,:,1], repeat_count, axis=-1)[:,:,:sample_count]

        return (
            ElectricalSignal(timing=timing, samples=i_samples),
            ElectricalSignal(timing=timing, samples=q_samples)
        )


class IQModulator(Composition):
    def __init__(self, carrier_source: Component, name=None) -> None:
        super().__init__([
            MapComponents([
                MachZehnderModulator(carrier_source=carrier_source),
                MachZehnderModulator(
                    carrier_source=Composition(components=[
                        carrier_source,
                        OpticalPhaseShifter(phase_shift=0.5 * math.pi)
                    ])
                )
            ]),
            OpticalCoupler2x1()
        ], name=name)


class IQReceiver(Composition):
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
                                        phase_shift=math.pi / 2,
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
    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    symbol_decoder: IQSymbolDecoder

    def __init__(
        self,
        *,
        data_rate: float | tf.Tensor,
        symbol_decoder: IQSymbolDecoder,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(tf.constant(data_rate), name=f"{self.name}__data_rate")
        self.symbol_decoder = symbol_decoder

        self.local_parameters = ["data_rate"]

    @tf.function(input_signature=(
        iq_signal_spec,
        tf.TensorSpec(shape=[], dtype=tf.int32, name="bits_per_symbol"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="data_rate")
    ))
    def _extract_iq_symbols(
        self,
        iq_signal: IQSignal,
        bits_per_symbol: tf.Tensor,
        data_rate: tf.Tensor
    ) -> tf.Tensor:
        i_samples = iq_signal[0].samples  # shape=[batch_size, parameter_dims, sample_count]
        q_samples = iq_signal[1].samples  # shape=[batch_size, parameter_dims, sample_count]

        batch_size = tf.shape(i_samples)[0]
        parameter_dims = tf.shape(i_samples)[1]
        total_sample_count = tf.size(i_samples) + tf.size(q_samples)

        iq_samples = tf.reshape(
            tf.stack([
                tf.reshape(i_samples, [tf.size(i_samples)]),
                tf.reshape(q_samples, [tf.size(q_samples)])
            ], axis=-1),
            (batch_size, parameter_dims, total_sample_count // batch_size // parameter_dims // 2, 2)
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
        parameter_dims = tf.shape(iq_signal[0].samples)[1]

        # shape=(batch_size, parameter_dims, symbol_count, 2)
        iq_symbols = self._extract_iq_symbols(iq_signal, bits_per_symbol, data_rate)
        symbol_count = tf.shape(iq_symbols)[-2]

        return tf.reshape(
            self.symbol_decoder(iq_symbols, tracing_level=tracing_level + 1, **kwargs),
            (batch_size, parameter_dims, symbol_count * bits_per_symbol)
        )

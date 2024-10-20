# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import math
import typing

import tensorflow as tf

from .datatypes import (
    bit_dtype, bit_sequence_spec,
    ElectricalSignal,
    IQSignal, iq_signal_spec,
    Timing
)

from .component import Component
from .mzm import MachZehnderModulator
from .passive import OpticalCoupler1x2, OpticalCoupler2x1, OpticalCoupler2x2, OpticalPhaseShifter
from .utils import (
    Composition, ExtractTiming, Fork, Identity, MapComponent, MapComponents,
    Reindex, adjust_symbol_for_mzm, indices_to_bits, bits_to_indices
)
from .electrical import ElectricalSubtractor

__all__ = ["IQSymbolEncoder", "IQSymbolDecoder", "IQEncoder", "IQModulator", "IQReceiver"]


class IQSymbolEncoder(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol, 2) or (None, 2 ** bits_per_symbol, 2); dtype=float32

    #: If true, the constellation will be adjusted for MZM
    adjust_for_mzm: tf.Variable  # shape=[]; dtype=bool

    def __init__(
        self,
        *,
        bits_per_symbol: int | tf.Tensor,
        mapping: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | tf.Tensor,
        adjust_for_mzm: bool | tf.Tensor = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol))

        self.mapping = tf.Variable(tf.constant(mapping))
        self.adjust_for_mzm = tf.Variable(tf.constant(adjust_for_mzm))

        self.parameters = ["bits_per_symbol", "mapping", "adjust_for_mzm"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=typing.cast(tf.TensorShape, None), dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    ))
    def implementation(
        self,
        bitseq: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor,
        adjust_for_mzm: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(bitseq)[0]
        bit_count = tf.shape(bitseq)[1]

        if tf.rank(mapping) < 3:
            mapping = tf.expand_dims(mapping, 0)

        parameter_dims = tf.shape(mapping)[0]

        symbol_indices = bits_to_indices(bitseq, bits_per_symbol)

        symbols = tf.reshape(
            tf.vectorized_map(lambda m: tf.gather(m, symbol_indices), mapping),
            (batch_size, parameter_dims, bit_count // bits_per_symbol, 2)
        )

        if adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(symbols)
        else:
            return symbols


class IQSymbolDecoder(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol, 2) or (None, 2 ** bits_per_symbol,2); dtype=float32

    def __init__(
        self,
        *,
        bits_per_symbol: int | tf.Tensor,
        mapping: Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol))

        self.mapping = tf.Variable(tf.constant(mapping))

        self.parameters = ["bits_per_symbol", "mapping"]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, None, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=typing.cast(tf.TensorShape, None), dtype=tf.float32)
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

        mapping = tf.broadcast_to(mapping, (parameter_dims, tf.math.pow(2, bits_per_symbol), 2))

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

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.symbol_encoder = symbol_encoder

        self.parameters = ["sampling_rate", "data_rate"]

    @tf.function
    def __call__(self, bitseq: tf.Tensor, **kwargs) -> IQSignal:
        assert isinstance(bitseq, tf.Tensor)
        assert bitseq.dtype is bit_dtype

        data_rate = self.get_component_parameter(kwargs, "data_rate")
        sampling_rate = self.get_component_parameter(kwargs, "sampling_rate")
        bits_per_symbol = self.symbol_encoder.get_component_parameter(kwargs, "bits_per_symbol")

        bit_count = tf.shape(bitseq)[-1]

        tf.debugging.assert_equal(tf.math.floormod(bit_count, bits_per_symbol), 0)

        symbol_rate = tf.divide(data_rate, tf.cast(bits_per_symbol, tf.float32))

        sample_count = tf.multiply(
            bit_count,
            tf.cast(tf.round(sampling_rate / data_rate), tf.int32)
        )

        signal_duration = tf.cast(sample_count, tf.float32) / sampling_rate

        iq_sequences = self.symbol_encoder(bitseq, **kwargs)

        timing = Timing(
            duration=signal_duration,
            sample_count=sample_count,
            sampling_rate=sampling_rate,
            symbol_rate=symbol_rate
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
        super().__init__(components=[
            MapComponents(components=[
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
        self.photodiode = photodiode
        self.local_oscillator = local_oscillator

        coupler1x2 = OpticalCoupler1x2()
        identity = Identity()

        # This is a coherent IQ receiver
        super().__init__(components=[
            coupler1x2,  # (OpticalSignal, OpticalSignal)
            Fork(shape=[2]),  # 2 * ((OpticalSignal, OpticalSignal))
            MapComponents(components=[
                identity,  # (OpticalSignal, OpticalSignal)
                Composition(components=[
                    Reindex(indices=(0,)),  # OpticalSignal
                    ExtractTiming(),  # Timing
                    local_oscillator,  # OpticalSignal
                    coupler1x2,  # (OpticalSignal, OpticalSignal)
                    MapComponents(components=[
                        OpticalPhaseShifter(phase_shift=math.pi / 2),  # OpticalSignal
                        identity  # OpticalSignal
                    ])  # (OpticalSignal, OpticalSignal)
                ])
            ]),  # 2 * ((OpticalSignal, OpticalSignal))
            Reindex(indices=[[(1, 1), (0, 0)], [(0, 1), (1, 0)]]),  # 2 * ((OpticalSignal, OpticalSignal),)
            MapComponent(
                component=Composition(components=[
                    OpticalCoupler2x2(flip=True),  # (OpticalSignal, OpticalSignal)
                    MapComponent(component=photodiode),  # (ElectricalSignal, ElectricalSignal)
                    ElectricalSubtractor()  # ElectricalSignal
                ])
            ),  # (ElectricalSignal, ElectricalSignal)
        ], name=name)


class IQDecoder(Component):
    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    symbol_decoder: IQSymbolDecoder

    #: Received symbols will be scaled by this factor (necessary for M-ary modulation formats)
    symbol_scale: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        *,
        data_rate: float | tf.Tensor,
        symbol_decoder: IQSymbolDecoder,
        symbol_scale: float | tf.Tensor = 1.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.symbol_decoder = symbol_decoder
        self.symbol_scale = tf.Variable(tf.constant(symbol_scale))

        self.parameters = ["data_rate", "symbol_scale"]

    @tf.function(input_signature=(
        iq_signal_spec,
        tf.TensorSpec(shape=[], dtype=tf.int32, name="bits_per_symbol"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="data_rate"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="symbol_scale")
    ))
    def _extract_iq_symbols(
        self,
        iq_signal: IQSignal,
        bits_per_symbol: tf.Tensor,
        data_rate: tf.Tensor,
        symbol_scale: tf.Tensor
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
            (batch_size, parameter_dims, total_sample_count // batch_size // 2, 2)
        )

        sampling_rate = iq_signal[0].timing.sampling_rate

        n = tf.cast(
            tf.round(sampling_rate / data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        indices = n * tf.range(tf.shape(iq_samples)[-2] // n) + n // 2
        iq_symbols = tf.math.real(tf.gather(iq_samples, indices, axis=-2)) / symbol_scale

        return iq_symbols

    @tf.function
    def __call__(self, iq_signal: IQSignal, **kwargs) -> tf.Tensor:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        symbol_scale = self.get_component_parameter(kwargs, "symbol_scale")
        bits_per_symbol = self.symbol_decoder.get_component_parameter(kwargs, "bits_per_symbol")

        batch_size = tf.shape(iq_signal[0].samples)[0]
        parameter_dims = tf.shape(iq_signal[0].samples)[1]

        # shape=(batch_size, parameter_dims, symbol_count, 2)
        iq_symbols = self._extract_iq_symbols(iq_signal, bits_per_symbol, data_rate, symbol_scale)
        symbol_count = tf.shape(iq_symbols)[-2]

        return tf.reshape(
            self.symbol_decoder(iq_symbols, **kwargs),
            (batch_size, parameter_dims, symbol_count * bits_per_symbol)
        )

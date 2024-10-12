# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import math

import tensorflow as tf

from .datatypes import (
    bit_dtype,
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
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,2); dtype=float32

    #: If true, the constellation will be adjusted for MZM
    adjust_for_mzm: tf.Variable  # shape=[]; dtype=bool

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

        self.mapping = tf.Variable(tf.constant(mapping))
        self.adjust_for_mzm = tf.Variable(tf.constant(adjust_for_mzm))

        self.parameters = ["bits_per_symbol", "mapping", "adjust_for_mzm"]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None], dtype=bit_dtype),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
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

        symbol_indices = bits_to_indices(
            tf.reshape(bitseq, (tf.size(bitseq) // bits_per_symbol, bits_per_symbol))
        )

        symbols = tf.reshape(
            tf.gather(mapping, symbol_indices),
            (batch_size, bit_count // bits_per_symbol, 2)
        )

        if adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(symbols)
        else:
            return symbols


class IQSymbolDecoder(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,2); dtype=float32

    def __init__(
        self,
        *,
        bits_per_symbol: int,
        mapping: Sequence[Sequence[float]],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(tf.constant(bits_per_symbol))
        self.mapping = tf.Variable(tf.constant(mapping))

        self.parameters = ["bits_per_symbol", "mapping"]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32)
    ))
    def implementation(
        self,
        symbols: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(symbols)[0]
        symbol_count = tf.shape(symbols)[1]

        best_indices = tf.cast(
            tf.argmin(
                tf.reduce_sum(tf.abs(mapping - tf.expand_dims(symbols, 2)), axis=-1),
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
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=[]; dtype=float32

    symbol_encoder: IQSymbolEncoder

    def __init__(
        self,
        *,
        sampling_rate: float,
        data_rate: float,
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

        bit_count = tf.shape(bitseq)[1]

        tf.debugging.assert_equal(tf.math.floormod(bit_count, bits_per_symbol), 0)

        symbol_rate = tf.divide(data_rate, tf.cast(bits_per_symbol, tf.float32))

        time = tf.cast(
            tf.range(
                tf.multiply(
                    bit_count,
                    tf.cast(tf.round(sampling_rate / data_rate), tf.int32)
                )
            ),
            tf.float32
        ) / sampling_rate

        iq_sequences = self.symbol_encoder(bitseq, **kwargs)

        timing = Timing(
            time=time,
            sampling_rate=sampling_rate,
            symbol_rate=symbol_rate
        )

        repeat_count = tf.cast(
            tf.math.ceil(tf.divide(tf.size(time), tf.shape(iq_sequences)[1])),
            tf.int32
        )

        i_samples = tf.repeat(iq_sequences[:,:,0], repeat_count, axis=1)[:,:tf.size(time)]
        q_samples = tf.repeat(iq_sequences[:,:,1], repeat_count, axis=1)[:,:tf.size(time)]

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
        data_rate: float,
        symbol_decoder: IQSymbolDecoder,
        threshold: float = 1e-5,
        symbol_scale: float = 1.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.symbol_decoder = symbol_decoder
        self.symbol_scale = tf.Variable(tf.constant(symbol_scale))

        self.parameters = ["data_rate", "symbol_scale"]

    @tf.function(input_signature=(
        iq_signal_spec,
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ))
    def _extract_iq_symbols(
        self,
        iq_signal: IQSignal,
        bits_per_symbol: tf.Tensor,
        data_rate: tf.Tensor,
        symbol_scale: tf.Tensor
    ) -> tf.Tensor:
        i_samples = iq_signal[0].samples
        q_samples = iq_signal[1].samples

        batch_size = tf.shape(i_samples)[0]
        sample_count = tf.size(i_samples) + tf.size(q_samples)

        iq_samples = tf.reshape(
            tf.stack([
                tf.reshape(i_samples, [tf.size(i_samples)]),
                tf.reshape(q_samples, [tf.size(q_samples)])
            ], axis=1),
            (batch_size, sample_count // batch_size // 2, 2)
        )

        sampling_rate = iq_signal[0].timing.sampling_rate

        n = tf.cast(
            tf.round(sampling_rate / data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        indices = n * tf.range(tf.shape(iq_samples)[1] // n) + n // 2
        iq_symbols = tf.math.real(tf.gather(iq_samples, indices, axis=1)) / symbol_scale

        return iq_symbols

    @tf.function
    def __call__(self, iq_signal: IQSignal, **kwargs) -> tf.Tensor:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        symbol_scale = self.get_component_parameter(kwargs, "symbol_scale")
        bits_per_symbol = self.symbol_decoder.get_component_parameter(kwargs, "bits_per_symbol")

        batch_size = tf.shape(iq_signal[0].samples)[0]

        # shape=(batch_size, symbol_count, 2)
        iq_symbols = self._extract_iq_symbols(iq_signal, bits_per_symbol, data_rate, symbol_scale)
        symbol_count = tf.shape(iq_symbols)[1]

        return tf.reshape(
            self.symbol_decoder(iq_symbols, **kwargs),
            (batch_size, symbol_count * bits_per_symbol)
        )

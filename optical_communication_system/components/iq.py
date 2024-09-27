# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Any, Optional
import math

import tensorflow as tf

from .datatypes import (
    bit_sequence_spec,
    ElectricalSignal,
    IQSignal, iq_signal_spec,
    Timing
)

from .component import Component
from .mzm import MachZehnderModulator
from .passive import OpticalCoupler1x2, OpticalCoupler2x1, OpticalCoupler2x2, OpticalPhaseShifter
from .utils import (
    Composition, ExtractTiming, Fork, Identity, MapComponent, MapComponents,
    Reindex, adjust_symbol_for_mzm, index_to_bits, bits_to_index
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
        bit_sequence_spec,
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
        symbol_index: Any = bits_to_index(bitseq)

        if adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(mapping[symbol_index])
        else:
            return mapping[symbol_index]


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
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32)
    ))
    def implementation(
        self,
        symbol: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor
    ) -> tf.Tensor:
        best_index = tf.cast(
            tf.argmin(
                tf.reduce_sum(
                    tf.abs(mapping - symbol),
                    axis=1
                )
            ),
            tf.int32
        )

        return index_to_bits(best_index, bits_per_symbol)


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
        assert bitseq.dtype is tf.int32

        data_rate = self.get_component_parameter(kwargs, "data_rate")
        sampling_rate = self.get_component_parameter(kwargs, "sampling_rate")
        bits_per_symbol = self.symbol_encoder.get_component_parameter(kwargs, "bits_per_symbol")

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

        iq_sequence = tf.vectorized_map(
            fn=lambda i: self.symbol_encoder(
                bitseq[i * bits_per_symbol:(i + 1) * bits_per_symbol],
                **kwargs
            ),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), bits_per_symbol))
        )

        timing = Timing(
            time=time,
            sampling_rate=sampling_rate,
            symbol_rate=tf.divide(data_rate, tf.cast(bits_per_symbol, tf.float32))
        )

        repeat_count = tf.cast(
            tf.math.ceil(tf.divide(tf.size(time), tf.shape(iq_sequence)[0])),
            tf.int32
        )

        i_samples = tf.repeat(iq_sequence[:,0], repeat_count)[:tf.size(time)]
        q_samples = tf.repeat(iq_sequence[:,1], repeat_count)[:tf.size(time)]

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
        i_current = tf.math.real(iq_signal[0].samples)
        q_current = tf.math.real(iq_signal[1].samples)

        sampling_rate = iq_signal[0].timing.sampling_rate

        n = tf.cast(
            tf.round(sampling_rate / data_rate * tf.cast(bits_per_symbol, tf.float32)),
            tf.int32
        )

        iq_symbols = tf.transpose(
            tf.vectorized_map(
                fn=lambda i: [
                    i_current[i * n + n // 2],
                    q_current[i * n + n // 2]
                ],
                elems=tf.range(tf.cast(tf.size(i_current) // n, tf.int32))
            )
        ) / symbol_scale

        return iq_symbols

    @tf.function
    def __call__(self, iq_signal: IQSignal, **kwargs) -> tf.Tensor:
        data_rate = self.get_component_parameter(kwargs, "data_rate")
        symbol_scale = self.get_component_parameter(kwargs, "symbol_scale")
        bits_per_symbol = self.symbol_decoder.get_component_parameter(kwargs, "bits_per_symbol")

        iq_symbols = self._extract_iq_symbols(iq_signal, bits_per_symbol, data_rate, symbol_scale)

        return tf.reshape(
            tf.vectorized_map(
                fn=lambda symbol: self.symbol_decoder(symbol, **kwargs),
                elems=iq_symbols
            ),
            (tf.shape(iq_symbols)[0] * bits_per_symbol,)
        )

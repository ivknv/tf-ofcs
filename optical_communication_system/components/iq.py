# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Any, Optional
import math

import tensorflow as tf

from .datatypes import (
    bit_dtype, bit_sequence_spec,
    ElectricalSignal,
    OpticalSignal,
    IQSignal, iq_signal_spec,
    Timing
)

from .component import Component
from .mzm import MachZehnderModulator
from .passive import OpticalCoupler1x2, OpticalCoupler2x2, OpticalPhaseShifter
from .utils import (
    Composition, ExtractTiming, Fork, Identity, MapComponent, MapComponents,
    Reindex, adjust_symbol_for_mzm, normalize_signal, index_to_bits, bits_to_index
)
from .electrical import ElectricalSubtractor

__all__ = ["IQConstellation", "IQEncoder", "IQModulator", "IQReceiver"]


class IQConstellation(Component):
    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=(1,); dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol,2); dtype=float32

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
        self.mapping = tf.Variable(tf.constant(mapping))
        self.adjust_for_mzm = tf.Variable(tf.constant(adjust_for_mzm))

    @tf.function(input_signature=(bit_sequence_spec,))
    def encode_symbol(self, bitseq: tf.Tensor) -> tf.Tensor:
        symbol_index: Any = bits_to_index(bitseq)

        if self.adjust_for_mzm:  # type: ignore
            return adjust_symbol_for_mzm(self.mapping[symbol_index])
        else:
            return self.mapping[symbol_index]

    @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
    def decode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        best_index = tf.cast(
            tf.argmin(
                tf.reduce_sum(
                    tf.abs(self.mapping - symbol),
                    axis=1
                )
            ),
            tf.int32
        )

        return index_to_bits(best_index, self.bits_per_symbol)


class IQEncoder(Component):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: IQConstellation

    def __init__(
        self,
        *,
        sampling_rate: float,
        data_rate: float,
        constellation: IQConstellation,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.constellation = constellation

    @tf.function(input_signature=(bit_sequence_spec,))
    def __call__(self, bitseq: tf.Tensor) -> IQSignal:
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

        iq_sequence = tf.vectorized_map(
            fn=lambda i: self.constellation.encode_symbol(
                bitseq[i * bits_per_symbol:(i + 1) * bits_per_symbol]
            ),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), bits_per_symbol))
        )

        timing = Timing(
            time=time,
            sampling_rate=self.sampling_rate,
            symbol_rate=tf.divide(self.data_rate, tf.cast(bits_per_symbol, tf.float32))
        )

        repeat_count = tf.cast(
            tf.math.ceil(tf.divide(tf.size(time), tf.shape(iq_sequence)[0])),
            tf.int32
        )

        i_samples = tf.repeat(iq_sequence[:,0], repeat_count)[:tf.size(time)]
        q_samples = tf.repeat(iq_sequence[:,1], repeat_count)[:tf.size(time)]

        return (
            ElectricalSignal(
                timing=timing,
                samples=i_samples
            ),
            ElectricalSignal(
                timing=timing,
                samples=q_samples
            )
        )


class IQModulator(Component):
    def __init__(self, carrier_source: Component, name=None) -> None:
        super().__init__(name=name)

        self.i_modulator = MachZehnderModulator(carrier_source=carrier_source)
        self.q_modulator = MachZehnderModulator(
            carrier_source=Composition(components=[
                carrier_source,
                OpticalPhaseShifter(phase_shift=0.5 * math.pi)
            ])
        )

    @tf.function(input_signature=(iq_signal_spec,))
    def __call__(self, iq_signal: IQSignal) -> OpticalSignal:
        i_signal = self.i_modulator(iq_signal[0])
        q_signal = self.q_modulator(iq_signal[1])

        combined_samples = (i_signal.samples + q_signal.samples) * 0.5 ** 0.5

        return OpticalSignal(
            timing=i_signal.timing,
            frequencies=i_signal.frequencies,
            samples=combined_samples
        )


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
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: IQConstellation

    #: Minimum current (or power in watts) recognized as non-zero
    threshold: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        *,
        data_rate: float,
        constellation: IQConstellation,
        threshold: float = 1e-5,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.constellation = constellation
        self.threshold = tf.Variable(tf.constant(threshold))

    @tf.function(input_signature=(iq_signal_spec,))
    def __call__(self, iq_signal: IQSignal) -> tf.Tensor:
        i_current = normalize_signal(tf.math.real(iq_signal[0].samples))
        q_current = normalize_signal(tf.math.real(iq_signal[1].samples))

        sampling_rate = iq_signal[0].timing.sampling_rate

        bits_per_symbol = self.constellation.bits_per_symbol

        n = tf.cast(
            tf.round(sampling_rate / self.data_rate * tf.cast(bits_per_symbol, tf.float32)),
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
        )

        return tf.reshape(
            tf.vectorized_map(
                fn=self.constellation.decode_symbol,
                elems=iq_symbols
            ),
            (tf.shape(iq_symbols)[0] * bits_per_symbol,)
        )

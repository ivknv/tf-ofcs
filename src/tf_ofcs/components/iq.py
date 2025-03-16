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

from ..datatypes import TensorCompatible, Signal, Timing

from .component import Component
from .mzm import MachZehnderModulator
from .passive import OpticalCoupler1x2, OpticalCoupler2x1, OpticalCoupler2x2, OpticalPhaseShifter
from .utils import Composition, Fork, Identity, MapComponents, Reindex
from ..utils import indices_to_bits, bits_to_indices, to_tensor
from .electrical_and_digital import Subtractor

__all__ = [
    "IQSymbolEncoder", "IQSymbolDecoder", "IQModulator", "IQReceiver",
    "IQToComplex", "ComplexToIQ"
]


class IQSymbolEncoder(Component):
    """
        Encodes a sequence of bits to a sequence of IQ symbols (complex-valued `Signal`).

        :ivar bits_per_symbol: `shape=[]; dtype=tf.int32`, number of bits per symbol
        :ivar mapping: `shape=[2 ** bits_per_symbol, 2]; dtype=float32`, mapping of symbol indices to IQ symbols.
        :ivar symbol_rate: `shape=[], dtype=tf.float32`, symbol rate in symbols/second
    """

    #: Number of bits per symbol
    bits_per_symbol: tf.Variable  # shape=[]; dtype=int32
    mapping: tf.Variable  # shape=(2 ** bits_per_symbol, 2); dtype=float32
    symbol_rate: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        *,
        bits_per_symbol: int | tf.Tensor,
        symbol_rate: TensorCompatible,
        mapping: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = tf.Variable(to_tensor(bits_per_symbol), name=f"{self.name}__bits_per_symbol")
        self.symbol_rate = tf.Variable(to_tensor(symbol_rate), name=f"{self.name}__symbol_rate")
        self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")

        self.local_parameters = ["bits_per_symbol", "symbol_rate", "mapping"]

    def implementation(
        self,
        bitseq: tf.Tensor,
        bits_per_symbol: tf.Tensor,
        symbol_rate: tf.Tensor,
        mapping: tf.Tensor
    ) -> Signal:
        batch_size = tf.shape(bitseq)[0]
        bit_count = tf.shape(bitseq)[1]

        symbol_indices = bits_to_indices(bitseq, bits_per_symbol)
        symbols = tf.reshape(
            tf.gather(mapping, symbol_indices),
            (batch_size, bit_count // bits_per_symbol, 2)
        )

        timing = Timing(sample_count=bit_count // bits_per_symbol, sampling_rate=symbol_rate)

        return Signal(samples=tf.complex(symbols[:, :, 0], symbols[:, :, 1]), timing=timing)


class IQSymbolDecoder(Component):
    """
        Decodes a sequence of IQ symbols (complex-valued `Signal`) to a sequence of bits.

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
        digital_signals: Signal,
        bits_per_symbol: tf.Tensor,
        mapping: tf.Tensor
    ) -> tf.Tensor:
        i_symbols = tf.math.real(digital_signals.samples)
        q_symbols = tf.math.imag(digital_signals.samples)

        symbols = tf.stack([i_symbols, q_symbols], axis=-1)

        batch_size = tf.shape(i_symbols)[0]
        symbol_count = tf.shape(i_symbols)[-1]

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
            (batch_size, symbol_count * bits_per_symbol)
        )


class IQModulator(Composition):
    """
        IQ modulator. Expects a `tuple[Signal, tuple[Signal, Signal]]`
        as input. The first tuple element is the carrier signal (optical signal),
        the second is the modulation signal (electrical I and Q signals).
    """

    def __init__(self, *, mzm: Optional[Component] = None, name: Optional[str] = None) -> None:
        super().__init__([], name=name)

        if mzm is None:
            mzm = MachZehnderModulator(name=f"{self.name}__mzm")

        identity = Identity(name=f"{self.name}__identity")

        self.components = [
            # input: (optical Signal, (electrical Signal, electrical Signal))
            MapComponents([
                Composition([
                    OpticalCoupler1x2(name=f"{self.name}__coupler1x2"),  # (optical Signal, optical Signal)
                    MapComponents([
                        identity,
                        OpticalPhaseShifter(phase_shift=[0.5 * math.pi], name=f"{self.name}__phase_shifter")
                    ])
                ]),  # (optical Signal, optical Signal)
                identity  # (electrical Signal, electrical Signal)
            ]),  # ((optical Signal 0, optical Signal 1), (electrical Signal 0, electrical Signal 1))

            Reindex(indices=[[(0, 0), (1, 0)], [(0, 1), (1, 1)]], name=f"{self.name}__reindex"),
            # reindex output: ((optical Signal 0, electrical Signal 0), (optical Signal 1, electrical Signal 1))

            MapComponents(mzm),  # (optical Signal, optical Signal)
            OpticalCoupler2x1(name=f"{self.name}__coupler_2x1")  # optical Signal
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
            coupler1x2,  # (optical signal, optical signal)
            Fork(shape=[2], name=f"{self.name}__fork"),  # 2 * ((optical signal, optical signal))
            MapComponents(
                [
                    identity,  # (optical signal, optical signal)
                    Composition(
                        components=[
                            Reindex(indices=(0,), name=f"{self.name}__reindex1"),  # optical signal
                            local_oscillator,  # optical signal
                            coupler1x2,  # (optical signal, optical signal)
                            MapComponents(
                                [
                                    OpticalPhaseShifter(
                                        phase_shift=[math.pi / 2],
                                        name=f"{self.name}__optical_phase_shifter"
                                    ),  # optical signal
                                    identity  # optical signal
                                ],
                                name=f"{self.name}__map_components2"
                            )  # (optical signal, optical signal)
                        ],
                        name=f"{self.name}__composition1"
                    )
                ],
                name=f"{self.name}__map_components1"
            ),  # 2 * ((optical signal, optical signal))
            Reindex(
                indices=[[(1, 1), (0, 0)], [(0, 1), (1, 0)]],
                name=f"{self.name}__reindex2"
            ),  # 2 * ((optical signal, optical signal),)
            MapComponents(
                Composition(
                    [
                        OpticalCoupler2x2(
                            flip=True,
                            name=f"{self.name}__optical_coupler2x2"
                        ),  # (optical signal, optical signal)
                        MapComponents(
                            photodiode,
                            name=f"{self.name}__map_components4"
                        ),  # (electrical signal, electrical signal)
                        Subtractor(name=f"{self.name}__electrical_subtractor")  # electrical signal
                    ],
                    name=f"{self.name}__composition2"
                ),
                name=f"{self.name}__map_components3"
            ),  # (electrical signal, electrical signal)
        ]


class IQToComplex(Component):
    """Converts a tuple of I and Q signals to a single complex-valued signal."""

    def implementation(self, iq_signal: tuple[Signal, Signal], /) -> Signal:
        i, q = iq_signal

        return Signal(samples=tf.complex(i.samples, q.samples), timing=i.timing)


class ComplexToIQ(Component):
    """Converts a complex-valued signal to a tuple of I (real) and Q (imaginary) signals."""

    def implementation(self, complex_signal: Signal, /) -> tuple[Signal, Signal]:
        i, q = tf.math.real(complex_signal.samples), tf.math.imag(complex_signal.samples)

        return (
            Signal(samples=i, timing=complex_signal.timing),
            Signal(samples=q, timing=complex_signal.timing)
        )

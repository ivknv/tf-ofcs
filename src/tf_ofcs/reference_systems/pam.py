# -*- coding: utf-8 -*-
# Copyright (C) 2025 Ivan Konovalov

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

import math
import tensorflow as tf

from .. import components
from ..datatypes import ElectricalSignal, TensorCompatible
from ..utils import dbm_to_w, pam_constellation, symbol_mapping_to_gray
from ..communication_system import CommunicationSystem

from typing import Any, Optional

__all__ = ["PAMSystem"]


class PAMSystem(components.Composition, CommunicationSystem):
    data_rate: TensorCompatible

    def __init__(
        self,
        bits_per_symbol: TensorCompatible = 4,
        data_rate: TensorCompatible = 42e9,
        n_taps: TensorCompatible = 13,
        name: Optional[str] = None
    ) -> None:
        components.Composition.__init__(self, [], name=name)

        self.sampling_rate = 336e9
        self.bits_per_symbol = tf.constant(bits_per_symbol)
        self.data_rate = tf.constant(data_rate)

        self.component_parameters = {}
        self.parameter_distributions = {}

        self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        self.lpf = components.BandPassFilter(
            min_frequency=[0.0],
            max_frequency=[self.data_rate],
            name="lpf"
        )
        self.laser = components.Laser(power=dbm_to_w([2.0]), name="laser")
        self.photodiode = components.Photodiode(
            maximum_power=dbm_to_w(tf.constant([3.0])),
            sensitivity=dbm_to_w(tf.constant([-20.0]))
        )

        padding = (n_taps - tf.constant(1)) // 2
        self.rx_filter = components.FIRFilter(
            taps=tf.pad([[1.0]], [[0, 0], [padding, padding]]),
            name="rx_filter"
        )

        if n_taps > 1:  # type: ignore
            self.slice_signal: components.Component = components.SliceSignal(start=padding, end=-padding)
        else:
            self.slice_signal = components.Identity()

        self.bitseq_generator = components.BitSequenceGenerator(length=self.bits_per_symbol * 32, rng=self.rng)
        self.channel = components.Substitute(components.Identity())

        self.downsampler = components.Downsample(4)

        mapping = symbol_mapping_to_gray(list(pam_constellation(self.bits_per_symbol).numpy()))

        self.encoder = components.PAMEncoder(
            sampling_rate=self.sampling_rate,
            data_rate=self.data_rate,
            symbol_encoder=components.PAMSymbolEncoder(
                bits_per_symbol=self.bits_per_symbol,
                mapping=tf.sqrt(mapping),
                adjust_for_mzm=True,
                name="pam_symbol_encoder"
            ),
            name="encoder"
        )

        self.decoder = components.PAMDecoder(
            data_rate=self.data_rate,
            symbol_decoder=components.PAMSymbolDecoder(
                bits_per_symbol=self.bits_per_symbol,
                mapping=mapping,
                name="pam_symbol_decoder"
            ),
            name="decoder"
        )

        self.transmitter = components.Composition([
            self.lpf,
            components.Fork(shape=[2]),
            components.MapComponents([self.laser, components.Identity()]),
            components.MachZehnderModulator(name="modulator")
        ], name="transmitter")

        self.receiver = components.Composition([
            self.photodiode,
            self.lpf,
            self.downsampler,
            components.Identity(name="before_filter"),
            self.rx_filter,
            self.slice_signal,
        ], name="receiver")

        self.components = [
            self.encoder,
            self.transmitter,
            self.channel,
            self.receiver,
            self.decoder
        ]

        self.test_sequence_generator = components.Composition([
            self.bitseq_generator,
            self.encoder,
            components.Fork([2]),
            components.MapComponents([
                components.Composition([
                    self.transmitter,
                    self.channel,
                    self.photodiode,
                    self.lpf,
                    self.downsampler,
                ]),
                components.Composition([
                    self.lpf,
                    components.Lambda(
                        lambda signal: ElectricalSignal(
                            samples=tf.square(tf.math.cos(math.pi * signal.samples)),
                            timing=signal.timing
                        )
                    ),
                    self.downsampler
                ])
            ])
        ])

        self.optimizer = tf.optimizers.Adam(learning_rate=1.0)

    def compute_filter_loss(
        self,
        received_seq: ElectricalSignal,
        expected_seq: ElectricalSignal,
    ) -> tf.Tensor:
        received_seq = self.slice_signal(self.rx_filter(received_seq))
        return tf.math.reduce_mean(tf.square(received_seq.samples - expected_seq.samples))

    @tf.function
    def train_filter(self, steps: TensorCompatible = 3000, **parameters) -> tf.Tensor:
        actual, expected = self.test_sequence_generator(**parameters)

        # Initialize the filter, figure out gain
        n_taps = tf.shape(self.rx_filter.taps)[-1]
        self.rx_filter.taps.assign(
            tf.pad(
                [[tf.math.reduce_sum(expected.samples) / tf.math.reduce_sum(actual.samples)]],
                [[0, 0], [(n_taps - 1) // 2, (n_taps - 1) // 2]]
            )
        )

        def loss_function() -> tf.Tensor:
            return self.compute_filter_loss(actual, expected)

        for _ in tf.range(steps):
            self.optimizer.minimize(loss_function, var_list=[self.rx_filter.taps])

        return loss_function()

    def setup(self, channel: components.Component, component_parameters: dict[str, Any]) -> None:
        self.channel.substitute = channel  # type: ignore[attr-defined]
        self.component_parameters.clear()
        self.component_parameters.update(component_parameters)

        parameters = dict(self.component_parameters)
        parameters["batch_size"] = tf.constant(1)
        self.train_filter(**parameters)

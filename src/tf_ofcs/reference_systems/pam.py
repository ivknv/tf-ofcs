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

import tensorflow as tf

from .. import components
from ..datatypes import Signal, TensorCompatible
from ..communication_system import CommunicationSystem
from ..utils import dbm_to_w, symbol_mapping_to_gray, pam_constellation

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

        maximum_power = dbm_to_w(tf.constant([3.0]))
        adc_step = maximum_power / 256.0
        adc_sampling_rate = 84e9
        dac_step = 1.0 / 256.0

        symbol_rate = self.data_rate / tf.cast(self.bits_per_symbol, tf.float32)
        oversampling_factor = int(adc_sampling_rate / symbol_rate)
        sim_upsampling_factor = 4

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
            maximum_power=maximum_power,
            dark_current=dbm_to_w(tf.constant([-20.0]))
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

        self.bitseq_generator = components.BitSequenceGenerator(length=self.bits_per_symbol * 512, rng=self.rng)
        self.channel = components.Substitute(components.Identity())

        self.sim_upsampler = components.Upsample(sim_upsampling_factor)

        mapping = symbol_mapping_to_gray(list(pam_constellation(self.bits_per_symbol).numpy()))

        self._pam_encoder = components.PAMSymbolEncoder(
            bits_per_symbol=self.bits_per_symbol,
            mapping=mapping,
            symbol_rate=symbol_rate,
            name="pam_symbol_encoder"
        )
        self.encoder = components.Composition([
            self._pam_encoder,
            components.PhotodiodePreDistortion(),
            components.MZMPreDistortion()
        ], name="encoder")

        self.decoder = components.PAMSymbolDecoder(
            bits_per_symbol=self.bits_per_symbol,
            mapping=mapping,
            name="decoder"
        )

        self.dac = components.DigitalToAnalogConverter(
            step=dac_step,
            sampling_rate=adc_sampling_rate,
            min_level=0.0,
            max_level=1.0
        )

        self.adc = components.AnalogToDigitalConverter(
            step=adc_step,
            min_level=0.0,
            max_level=maximum_power,
            sampling_rate=adc_sampling_rate
        )

        self.transmitter = components.Composition([
            components.Upsample(oversampling_factor),
            self.lpf,
            self.dac,
            self.sim_upsampler,
            components.Fork(shape=[2]),
            components.MapComponents([self.laser, components.Identity()]),
            components.MachZehnderModulator(name="modulator")
        ], name="transmitter")

        self.receiver = components.Composition([
            self.photodiode,
            self.adc,
            self.lpf,
            self.rx_filter,
            self.slice_signal,
            components.Downsample(factor=oversampling_factor, time_offset=0.5 / symbol_rate)
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
            components.Fork([2]),
            components.MapComponents([
                components.Composition([
                    self.encoder,
                    self.transmitter,
                    self.channel,
                    self.photodiode,
                    self.adc,
                    self.lpf
                ]),
                components.Composition([
                    self._pam_encoder,
                    components.Upsample(factor=oversampling_factor),
                    self.lpf
                ])
            ])
        ])

        self.optimizer = tf.optimizers.Adam(learning_rate=1.0)

    def compute_filter_loss(
        self,
        received_seq: Signal,
        expected_seq: Signal,
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
                [[
                    tf.math.divide_no_nan(
                        tf.math.reduce_sum(expected.samples),
                        tf.math.reduce_sum(actual.samples)
                    )
                ]],
                [[0, 0], [(n_taps - 1) // 2, (n_taps - 1) // 2]]
            )
        )

        def loss_function() -> tf.Tensor:
            return self.compute_filter_loss(actual, expected)

        for _ in tf.range(steps):
            self.optimizer.minimize(loss_function, var_list=[self.rx_filter.taps])

        return loss_function()

    def set_channel(self, channel: components.Component) -> None:
        self.channel.substitute = channel  # type: ignore[attr-defined]

    def setup(self, channel: components.Component, component_parameters: dict[str, Any]) -> None:
        super().setup(channel, component_parameters)

        parameters = dict(self.component_parameters)
        parameters["batch_size"] = tf.constant(1)
        self.train_filter(**parameters)

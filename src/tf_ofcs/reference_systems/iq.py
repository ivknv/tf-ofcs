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
from ..utils import dbm_to_w, to_tensor

from typing import Any, Literal, Optional

__all__ = ["IQSystem"]


class IQSystem(CommunicationSystem):
    rng: tf.random.Generator

    def __init__(
        self,
        mapping: TensorCompatible,
        bits_per_symbol: int = 2,
        data_rate: float = 42e9,
        n_taps: int = 13,
        enable_phase_noise: bool = False,
        carrier_phase_recovery: Literal["decision_directed"] | Literal["viterbi_viterbi"] | None = None,
        name: Optional[str] = None
    ) -> None:
        maximum_power = dbm_to_w(tf.constant([3.0]))
        adc_step = maximum_power / 128.0
        adc_sampling_rate = 84e9
        dac_step = 1.0 / 256.0

        symbol_rate = data_rate / tf.cast(bits_per_symbol, tf.float32)
        oversampling_factor = int(adc_sampling_rate / symbol_rate)
        sim_upsampling_factor = 4

        self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        lpf = components.FFTFilter(
            components.RootRaisedCosineFrequencyResponse(
                rolloff_factor=0.5,
                symbol_rate=symbol_rate,
                name="lpf"
            )
        )
        laser = components.Laser(
            power=dbm_to_w([2.0]),
            enable_phase_noise=enable_phase_noise,
            name="laser"
        )
        local_oscillator = components.Laser(
            power=dbm_to_w([-8.0]),
            enable_phase_noise=enable_phase_noise,
            name="local_oscillator"
        )
        photodiode = components.Photodiode(
            maximum_power=maximum_power,
            dark_current=dbm_to_w(tf.constant([-20.0]))
        )

        padding = (n_taps - tf.constant(1)) // 2
        self._rx_filter = components.FIRFilter(
            taps=tf.pad([[tf.complex(1.0, 0.0)]], [[0, 0], [padding, padding]]),
            name="rx_filter"
        )

        self._truncate_signal = components.TruncateSignal(left=padding, right=padding)

        bitseq_generator = components.BitSequenceGenerator(length=bits_per_symbol * 512, rng=self.rng)
        sim_upsampler = components.Upsample(sim_upsampling_factor)

        iq_encoder = components.IQSymbolEncoder(
            bits_per_symbol=bits_per_symbol,
            mapping=mapping,
            symbol_rate=symbol_rate,
            name="qam_symbol_encoder"
        )
        encoder = components.Composition([
            iq_encoder,
            components.Upsample(factor=oversampling_factor),
            lpf,
            components.MZMPreDistortion()
        ], name="encoder")

        decoder = components.IQSymbolDecoder(
            bits_per_symbol=bits_per_symbol,
            mapping=mapping,
            name="decoder"
        )

        dac = components.Composition([
            components.ComplexToIQ(),
            components.MapComponents(
                components.DigitalToAnalogConverter(
                    step=dac_step,
                    sampling_rate=adc_sampling_rate,
                    min_level=0.0,
                    max_level=1.0
                )
            )
        ])

        adc = components.Composition([
            components.MapComponents(
                components.AnalogToDigitalConverter(
                    step=adc_step,
                    min_level=-maximum_power / 2.0,
                    max_level=maximum_power / 2.0,
                    sampling_rate=adc_sampling_rate
                )
            ),
            components.IQToComplex()
        ])

        transmitter = components.Composition([
            dac,
            components.MapComponents(sim_upsampler),
            components.Fork(shape=[2]),
            components.MapComponents([
                components.Composition([
                    components.Reindex(indices=(0,)),
                    laser,
                ]),
                components.Identity()
            ]),
            components.IQModulator(name="modulator")
        ], name="transmitter")

        receiver = components.Composition([
            components.IQReceiver(
                photodiode=photodiode,
                local_oscillator=local_oscillator
            ),
            adc,
            lpf,
            components.Identity("before_filter"),
            self._rx_filter,
            self._truncate_signal,
            components.TruncateSignal(left=0.5 / symbol_rate, mode="time"),
            components.Downsample(factor=oversampling_factor),
        ], name="receiver")

        if carrier_phase_recovery == "decision_directed":
            receiver.components.append(
                components.DecisionDirectedPhaseRecovery(
                    window_length=4,
                    mapping=tf.complex(to_tensor(mapping)[:, 0], to_tensor(mapping)[:, 1])
                )
            )
        elif carrier_phase_recovery == "viterbi_viterbi":
            receiver.components.append(
                components.ViterbiViterbiPhaseRecovery(
                    window_length=4,
                    M=tf.shape(mapping)[0],
                    phase_offset=0.0
                )
            )

        super().__init__(
            bits_per_symbol=bits_per_symbol,
            encoder=encoder,
            decoder=decoder,
            transmitter=transmitter,
            receiver=receiver,
            name=name
        )

        self._test_sequence_generator = components.Composition([
            bitseq_generator,
            components.Fork([2]),
            components.MapComponents([
                components.Composition([
                    encoder,
                    transmitter,
                    self.channel,
                    components.IQReceiver(
                        photodiode=photodiode,
                        local_oscillator=local_oscillator
                    ),
                    adc,
                    lpf
                ]),
                components.Composition([
                    iq_encoder,
                    components.Upsample(factor=oversampling_factor),
                    lpf,
                    lpf
                ])
            ])
        ])

        self._optimizer = tf.optimizers.Adam(learning_rate=1.0)

    def compute_filter_loss(
        self,
        received_seq: Signal,
        expected_seq: Signal,
    ) -> tf.Tensor:
        received_seq = self._truncate_signal(self._rx_filter(received_seq))
        return tf.math.reduce_mean(tf.square(tf.abs(received_seq.samples - expected_seq.samples)))

    @tf.function
    def train_filter(self, steps: TensorCompatible = 3000, **parameters) -> tf.Tensor:
        actual, expected = self._test_sequence_generator(**parameters)

        # Initialize the filter, figure out gain
        n_taps = tf.shape(self._rx_filter.taps)[-1]
        self._rx_filter.taps.assign(
            tf.pad(
                [[
                    tf.complex(
                        tf.abs(
                            tf.math.divide_no_nan(
                                tf.math.reduce_sum(expected.samples),
                                tf.math.reduce_sum(actual.samples)
                            )
                        ),
                        0.0
                    )
                ]],
                [[0, 0], [(n_taps - 1) // 2, (n_taps - 1) // 2]]
            )
        )

        def loss_function() -> tf.Tensor:
            return self.compute_filter_loss(actual, expected)

        for _ in tf.range(steps):
            self._optimizer.minimize(loss_function, var_list=[self._rx_filter.taps])

        return loss_function()

    def setup(self, channel: components.Component, component_parameters: dict[str, Any]) -> None:
        super().setup(channel, component_parameters)

        parameters = dict(self.component_parameters)
        parameters["batch_size"] = tf.constant(1)
        self.train_filter(**parameters)

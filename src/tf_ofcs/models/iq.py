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

from collections.abc import Callable, Sequence
import math
from typing import Any, Optional
import tensorflow as tf
import keras  # type: ignore[import-untyped]

from ..datatypes import Signal, Timing
from ..utils import bits_to_indices, indices_to_bits
from ..utils import dbm_to_w

from ..components import (
    Component, Composition, IQModulator, Laser, IQReceiver, Photodiode,
    BandPassFilter, SignalFromTensor, Lambda, RandomizeParameters,
    MapComponents, Gain, Fork, Identity, Reindex, Substitute,
    AnalogToDigitalConverter, DigitalToAnalogConverter, Downsample, Upsample,
    SelectSamples
)
from ..communication_system import CommunicationSystem
from .common import TransmitterConfig, ReceiverConfig

__all__ = ["IQAutoencoder"]


def _clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0.0, 1.0)


def _apply_symbol_mapping(inputs: tf.Tensor, mapping: tf.Tensor) -> tf.Tensor:
    return tf.gather(mapping, inputs, axis=-1)


def _reverse_mapping(mapping: tf.Tensor) -> tf.Tensor:
    return tf.argsort(mapping)


def _to_iq(samples: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(samples)[0]
    sample_count = tf.shape(samples)[-1]
    shape = [batch_size, sample_count]

    return tf.reshape(samples[:, 0, :], shape), tf.reshape(samples[:, 1, :], shape)


def _from_iq_signal(signals: tuple[Signal, Signal]) -> tf.Tensor:
    i, q = signals  # shape=[batch_size, symbols_per_batch * n]
    shape = tf.shape(i.samples)
    batch_size, sample_count = shape[0], shape[1]

    return tf.reshape(tf.stack([i.samples, q.samples], axis=1), [batch_size, 2, sample_count])


class TransmissionLayer(keras.layers.Layer):
    def __init__(
        self,
        *,
        M: int,
        n: int,
        tx_config: TransmitterConfig,
        rx_config: ReceiverConfig,
        simulation_sampling_rate: float,
        enable_phase_noise: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name, trainable=False)

        assert "N" in tx_config
        assert "N" in rx_config
        assert "dac_sampling_rate" in tx_config
        assert "adc_sampling_rate" in rx_config
        assert "dsp_sampling_rate" in tx_config

        adc_sampling_rate = rx_config["adc_sampling_rate"]
        dac_sampling_rate = tx_config["dac_sampling_rate"]
        tx_dsp_sampling_rate = tx_config["dsp_sampling_rate"]

        self._channel = Substitute(Identity())
        self._component_parameters: dict[str, Any] = {}
        self._parameter_distributions: dict[str, Any] = {}

        self._validation_component_parameters: dict[str, Any] = {}
        self._validation_parameter_distributions: dict[str, Any] = {}

        self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        self._M = M
        self._N_tx = tx_config["N"]
        self._N_rx = rx_config["N"]
        self._n = n
        self._sampling_rate = simulation_sampling_rate
        self._rx_oversampling = int(simulation_sampling_rate / adc_sampling_rate)

        bandwidth = 32e9 # Hz

        lpf = BandPassFilter(min_frequency=[0.0], max_frequency=[bandwidth])

        dac_max_power = 1.0
        dac_step = dac_max_power / 256.0
        adc_max_power = dbm_to_w(2.0)
        adc_step = adc_max_power / 256.0

        adc = AnalogToDigitalConverter(
            sampling_rate=adc_sampling_rate,
            step=adc_step,
            min_level=0.0,
            max_level=adc_max_power
        )

        dac = DigitalToAnalogConverter(
            sampling_rate=dac_sampling_rate,
            step=dac_step,
            min_level=0.0,
            max_level=dac_max_power
        )

        self._laser = Substitute(Laser(power=dbm_to_w([2.0]), enable_phase_noise=enable_phase_noise))
        self._local_oscillator = Substitute(
            Laser(
                power=dbm_to_w([-6.0]),
                enable_phase_noise=enable_phase_noise,
                name="local_oscillator"
            )
        )

        self._transmitter = Composition([
            MapComponents(
                Composition([
                    lpf,
                    Downsample(int(tx_dsp_sampling_rate / dac_sampling_rate)),
                    dac,
                    Upsample(int(tx_dsp_sampling_rate / dac_sampling_rate))
                ])
            ),
            Fork(shape=[2]),
            MapComponents([
                Composition([
                    Reindex(indices=(0,)),
                    self._laser
                ]),
                Identity()
            ]),
            IQModulator(),
        ], name="transmitter")

        self._receiver = Composition([
            IQReceiver(
                Photodiode(
                    maximum_power=dbm_to_w(tf.constant([3.0])),
                    dark_current=dbm_to_w(tf.constant([-20.0]))
                ),
                self._local_oscillator
            ),
            MapComponents(Composition([
                adc,
                Gain(gain=[1e5]),
                lpf
            ])),
        ], name="receiver")

        self._communication_system = Composition([
            Lambda(_to_iq, name="to_iq"),
            MapComponents(SignalFromTensor(sampling_rate=simulation_sampling_rate)),
            self._transmitter,
            self._channel,
            self._receiver,
            MapComponents(SelectSamples(
                count=self._N_rx,
                size=n // 2 // self._rx_oversampling,
                name="select_symbols"
            )),
            Lambda(_from_iq_signal)
        ])

        self._training_randomize = RandomizeParameters(
            self._communication_system,
            shape=[1],
            distributions=self._parameter_distributions,
            rng=self.rng,
            name="training_randomize"
        )
        self._validation_randomize = RandomizeParameters(
            self._communication_system,
            shape=[1],
            distributions=self._parameter_distributions,
            rng=self.rng,
            name="validation_randomize"
        )

    def _set_channel(self, channel: Component) -> None:
        self._channel.substitute = channel

    def _precompute_lasers(self) -> None:
        """
            Replace laser by a function that returns a constant precomputed signal.
            This can speed up training a bit, but requires the assumption that
            laser signal doesn't change inbetween steps.
        """

        def laser_substitute(laser: Substitute) -> Callable:
            timing = Timing(sampling_rate=self._sampling_rate, sample_count=self._n // 2 * self._N_tx)
            params = dict(self._component_parameters)
            params["batch_size"] = tf.constant(1)
            signal = laser.component(timing, **params)

            return lambda *_, **kwargs: Signal(
                samples=tf.tile(signal.samples, [kwargs["batch_size"], 1]),
                timing=signal.timing
            )

        self._laser.substitute = laser_substitute(self._laser)
        self._local_oscillator.substitute = laser_substitute(self._local_oscillator)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        sample_count = tf.shape(inputs)[-1]

        # Number of adjacent symbols that will be kept after the receiver (before and after the main symbol)
        half_adj_rx = (self._N_rx - 1) // 2

        main_symbol_index = (self._N_tx - 1) // 2

        if self._N_tx > 1:
            adjacent_inputs = tf.gather(
                inputs,
                self.rng.uniform(((self._N_tx - 1) * batch_size,), 0, batch_size, dtype=tf.int32)
            )  # shape=[batch_size * (N - 1), 2, n // 2]

            tf.debugging.assert_equal(
                tf.shape(adjacent_inputs),
                [batch_size * (self._N_tx - 1), 2, self._n // 2]
            )

            # shape=[batch_size, 2, (N - 1) * n // 2]
            adjacent_inputs = tf.stack(
                [
                    tf.reshape(adjacent_inputs[:, 0, :], [batch_size, -1]),
                    tf.reshape(adjacent_inputs[:, 1, :], [batch_size, -1])
                ],
                axis=1
            )

            inputs = tf.concat([
                adjacent_inputs[:, :, :main_symbol_index * sample_count],
                inputs,
                adjacent_inputs[:, :, main_symbol_index * sample_count:]
            ], -1)

        tf.debugging.assert_equal(tf.shape(inputs)[-1], self._N_tx * self._n // 2)

        if training:
            parameters = dict(self._component_parameters)
            component = self._training_randomize
        else:
            parameters = dict(self._validation_component_parameters)
            component = self._validation_randomize

        parameters["batch_size"] = batch_size
        parameters["validation_randomize__shape"] = tf.expand_dims(batch_size, 0)
        parameters["training_randomize__shape"] = tf.expand_dims(batch_size, 0)

        return tf.reshape(
            component(
                inputs,
                select_symbols__index=main_symbol_index - half_adj_rx,
                **parameters
            ),
            [batch_size, 2, sample_count * self._N_rx // self._rx_oversampling]
        )

    def get_config(self) -> dict:
        return {}


class TransmitterLayer(keras.layers.Layer):
    def __init__(
        self,
        M: int,
        n: int,
        config: TransmitterConfig,
        name: Optional[str] = None
    ) -> None:
        super().__init__(trainable=config.get("trainable", True), name=name)

        self._layers = []
        n_layers = config.get("n_layers", 3)
        enable_batch_normalization = config.get("enable_batch_normalization", True)
        dropout_rate = config.get("dropout_rate", 0.0)

        l2_regularization = config.get("l2_regularization", 0.0)

        if l2_regularization > 0.0:
            regularizer = keras.regularizers.L2(l2=l2_regularization)
        else:
            regularizer = None

        for i in range(n_layers - 1):
            self._layers.append(
                keras.layers.Dense(
                    2 * M,
                    regularizer=regularizer,
                    name=f"{self.name}_{i + 1}"
                )
            )

            if enable_batch_normalization:
                self._layers.append(keras.layers.BatchNormalization(name=f"{self.name}_batch_norm{i + 1}"))

            self._layers.append(keras.layers.Activation("relu", name=f"{self.name}_{i + 1}_relu"))

            if dropout_rate > 0.0:
                self._layers.append(keras.layers.Dropout(dropout_rate, name=f"{self.name}_dropout{i + 1}"))

        self._layers.append(keras.layers.Dense(n, activation=_clip, name=f"{self.name}_{n_layers}"))
        self._layers.append(keras.layers.Reshape((2, n // 2), name=f"{self.name}_reshape"))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = inputs

        for layer in self._layers:
            output = layer(output)

        return output


class ReceiverLayer(keras.layers.Layer):
    def __init__(
        self,
        M: int,
        n: int,
        simulation_sampling_rate: float,
        config: ReceiverConfig,
        name: Optional[str] = None
    ) -> None:
        super().__init__(trainable=config.get("trainable", True), name=name)

        assert "adc_sampling_rate" in config

        self._layers = []
        n_layers = config.get("n_layers", 3)
        enable_batch_normalization = config.get("enable_batch_normalization", True)
        N_rx = config.get("N", 1)
        oversampling = int(simulation_sampling_rate / config["adc_sampling_rate"])
        dropout_rate = config.get("dropout_rate", 0.0)

        l2_regularization = config.get("l2_regularization", 0.0)

        if l2_regularization > 0.0:
            regularizer = keras.regularizers.L2(l2=l2_regularization)
        else:
            regularizer = None

        self._layers.append(keras.layers.Reshape((n * N_rx // oversampling,), name=f"{self.name}__reshape"))

        for i in range(n_layers - 1):
            self._layers.append(
                keras.layers.Dense(
                    2 * M,
                    kernel_regularizer=regularizer,
                    name=f"{self.name}_{i + 1}"
                )
            )

            if enable_batch_normalization:
                self._layers.append(keras.layers.BatchNormalization(name=f"{self.name}_batch_norm{i + 1}"))

            self._layers.append(keras.layers.Activation("relu", name=f"{self.name}_{i + 1}_relu"))

            if dropout_rate > 0.0:
                self._layers.append(keras.layers.Dropout(dropout_rate, name=f"{self.name}_dropout{i + 1}"))

        self._layers.append(keras.layers.Dense(M, activation="softmax", name=f"{self.name}_{n_layers}"))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = inputs

        for layer in self._layers:
            output = layer(output)

        return output


class IQAutoencoderModel(keras.models.Sequential):
    """
        An autoencoder model with an IQ modulator.

        :ivar M: `int`, number of values a single message (symbol) carries
        :ivar n: `int`, number of samples per symbol produced by the transmitter
        :ivar N_tx: `int`, number of adjacent symbols transmitted during training (must be odd)
        :ivar N_rx: `int`, number of symbols processed by the receiver (must be odd)

        :ivar encoder_network: `Component`, transmitter layers of the neural network
        :ivar decoder_network: `Component`, receiver layers of the neural network
    """

    M: int
    n: int
    bits_per_symbol: int
    simulation_sampling_rate: float

    encoder_network: Component
    decoder_network: Component

    def __init__(
        self,
        M: int,
        n: int,
        enable_phase_noise: bool = False,
        simulation_sampling_rate: float = 336e9,
        tx_config: Optional[TransmitterConfig] = None,
        rx_config: Optional[ReceiverConfig] = None,
        symbol_mapping: Optional[Sequence[int] | tf.Tensor] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.tx_config = TransmitterConfig({
            "dsp_sampling_rate": simulation_sampling_rate,
            "dac_sampling_rate": 84e9,
            "N": 1,
            "n_layers": 3,
            "trainable": True,
            "enable_batch_normalization": True
        })
        self.tx_config.update(tx_config or {})

        self.rx_config = ReceiverConfig({
            "adc_sampling_rate": 84e9,
            "N": 1,
            "n_layers": 3,
            "trainable": True,
            "enable_batch_normalization": True
        })
        self.rx_config.update(rx_config or {})

        self.bits_per_symbol = int(math.log2(M))
        self.simulation_sampling_rate = simulation_sampling_rate

        assert "adc_sampling_rate" in self.rx_config
        self._rx_oversampling = int(simulation_sampling_rate / self.rx_config["adc_sampling_rate"])

        #: Size of the input one-hot vector
        self.M = M

        #: Number of samples produced by the final transmitter layer
        self.n = n

        if symbol_mapping is None:
            self.symbol_mapping = tf.range(M, dtype=tf.int32)
        else:
            self.symbol_mapping = tf.constant(symbol_mapping, dtype=tf.int32)

        self._reverse_symbol_mapping = _reverse_mapping(self.symbol_mapping)

        self._enable_phase_noise = enable_phase_noise

        if self.tx_config.get("N", 1) % 2 != 1:
            raise ValueError("N (for transmitter) must be odd")

        if self.rx_config.get("N", 1) % 2 != 1:
            raise ValueError("N (for receiver) must be odd")

        if n % 2 != 0:
            raise ValueError("n must be even")

        self._tx_layer = TransmitterLayer(M=M, n=n, config=self.tx_config, name="tx")
        self._rx_layer = ReceiverLayer(
            M=M,
            n=n,
            simulation_sampling_rate=simulation_sampling_rate,
            config=self.rx_config,
            name="rx"
        )

        self._transmission_layer = TransmissionLayer(
            M=M,
            n=n,
            tx_config=self.tx_config,
            rx_config=self.rx_config,
            simulation_sampling_rate=simulation_sampling_rate,
            enable_phase_noise=enable_phase_noise,
            name="transmission"
        )

        self.encoder_network = Lambda(self._tx_layer, name="encoder_network")
        self.decoder_network = Lambda(self._rx_layer, name="decoder_network")

        self.encoder = Lambda(self.encode_input)
        self.decoder = Lambda(self.decode_signal)

        self.add(keras.Input(shape=(M,)))
        self.add(self._tx_layer)
        self.add(self._transmission_layer)
        self.add(self._rx_layer)

    def encode_input(self, bitseq: tf.Tensor) -> tuple[Signal, Signal]:
        inputs = bits_to_indices(bitseq, tf.constant(self.bits_per_symbol))
        inputs = _apply_symbol_mapping(inputs, self._reverse_symbol_mapping)

        # shape=[batch_size, bitseq_size // batch_size // self.bits_per_symbol, self.M]
        inputs = tf.one_hot(tf.expand_dims(inputs, -1), self.M)

        N_rx = self.rx_config["N"]  # pyright: ignore[reportTypedDictNotRequiredAccess]

        # Mirror inputs on both sides to add (N_rx - 1) // 2 adjacent symbols
        adjacent_symbols_count = (N_rx - 1) // 2
        inputs = tf.concat(
            [
                inputs[:, :adjacent_symbols_count],
                inputs,
                inputs[:, tf.shape(inputs)[1] - adjacent_symbols_count:]
            ],
            1
        )  # shape=[batch_size, bitseq_size // self.bits_per_symbol + N_rx - 1, self.M]
        inputs = tf.reshape(inputs, [tf.size(inputs) // self.M, self.M])
        x = self.encoder_network(inputs)  # shape=[batch_size * symbols_per_batch, 2, n // 2]

        batch_size = tf.shape(bitseq)[0]

        i_seq = x[:, 0, :]
        q_seq = x[:, 1, :]

        i_seq = tf.reshape(i_seq, [batch_size, tf.size(i_seq) // batch_size])
        q_seq = tf.reshape(q_seq, [batch_size, tf.size(q_seq) // batch_size])

        timing = Timing(sample_count=tf.shape(i_seq)[-1], sampling_rate=self.simulation_sampling_rate)

        return Signal(samples=i_seq, timing=timing), Signal(samples=q_seq, timing=timing)

    def decode_signal(self, input_signal: tuple[Signal, Signal]) -> tf.Tensor:
        samples = _from_iq_signal(input_signal)

        N_rx = self.rx_config["N"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        batch_size = tf.shape(samples)[0]
        samples_per_symbol = self.n // 2 // self._rx_oversampling
        samples_per_block = N_rx * samples_per_symbol

        samples = tf.signal.frame(
            samples,
            samples_per_block,
            samples_per_symbol,
            axis=2
        )

        i_seq = samples[:, 0, :]
        q_seq = samples[:, 1, :]

        tf.debugging.assert_equal(tf.size(i_seq) % samples_per_block, 0)

        i_seq = tf.reshape(i_seq, [tf.size(i_seq) // samples_per_block, samples_per_block])
        q_seq = tf.reshape(q_seq, [tf.size(q_seq) // samples_per_block, samples_per_block])

        samples = tf.stack([i_seq, q_seq], axis=1)

        x = self.decoder_network(samples)  # shape=[None, M]

        indices = tf.cast(tf.argmax(x, axis=1), tf.int32)
        indices = _apply_symbol_mapping(indices, self.symbol_mapping)
        bitseq = indices_to_bits(indices, tf.constant(self.bits_per_symbol))

        return tf.reshape(bitseq, (batch_size, tf.size(bitseq) // batch_size))

    def _setup(
        self,
        channel: Component,
        component_parameters: dict[str, Any],
        validation_component_parameters: dict[str, Any],
        parameter_distributions: dict[str, Any],
        validation_parameter_distributions: dict[str, Any]
    ) -> None:
        self._transmission_layer._set_channel(channel)

        self._transmission_layer._component_parameters.clear()
        self._transmission_layer._component_parameters.update(component_parameters)

        self._transmission_layer._validation_component_parameters.clear()
        self._transmission_layer._validation_component_parameters.update(validation_component_parameters)

        self._transmission_layer._parameter_distributions.clear()
        self._transmission_layer._parameter_distributions.update(parameter_distributions)

        self._transmission_layer._validation_parameter_distributions.clear()
        self._transmission_layer._validation_parameter_distributions.update(validation_parameter_distributions)

        if not self._enable_phase_noise:
            self._transmission_layer._precompute_lasers()


class IQAutoencoder(CommunicationSystem):
    def __init__(
        self,
        M: int,
        n: int,
        enable_phase_noise: bool = False,
        simulation_sampling_rate: float = 336e9,
        tx_config: Optional[TransmitterConfig] = None,
        rx_config: Optional[ReceiverConfig] = None,
        symbol_mapping: Optional[Sequence[int] | tf.Tensor] = None,
        name: Optional[str] = None
    ) -> None:
        if name is None:
            name = "iq_autoencoder_system"

        model = IQAutoencoderModel(
            M=M,
            n=n,
            enable_phase_noise=enable_phase_noise,
            simulation_sampling_rate=simulation_sampling_rate,
            tx_config=tx_config,
            rx_config=rx_config,
            symbol_mapping=symbol_mapping,
            name=f"{name}_model"
        )

        super().__init__(
            bits_per_symbol=model.bits_per_symbol,
            encoder=Lambda(model.encode_input),
            decoder=Lambda(model.decode_signal),
            transmitter=model._transmission_layer._transmitter,
            receiver=model._transmission_layer._receiver,
            model=model,
            name=name
        )

    def setup_model(
        self,
        channel: Component,
        component_parameters: dict[str, Any],
        validation_component_parameters: dict[str, Any],
        parameter_distributions: dict[str, Any],
        validation_parameter_distributions: dict[str, Any]
    ) -> None:
        assert isinstance(self.model, IQAutoencoderModel)
        self.model._setup(
            channel,
            component_parameters,
            validation_component_parameters,
            parameter_distributions,
            validation_parameter_distributions
        )

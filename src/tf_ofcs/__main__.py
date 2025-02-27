# -*- coding: utf-8
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

import argparse
import math
import json
import os
from typing import Any, Optional
import datetime

import h5py  # type: ignore[import-untyped]
import keras  # type: ignore[import-untyped]
import tensorflow as tf

from .models import AutoencoderModel, IMDDAutoencoder, IQAutoencoder
from . import components
from .utils import indices_to_bits


class Command:
    description: str = ""

    def setup(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def __call__(self, ns: argparse.Namespace) -> int:
        raise NotImplementedError


def _load_profile(path: str) -> Any:
    with open(path, "rb") as profile_file:
        profile = json.load(profile_file)

    profile.setdefault("optimizer_parameters", {})
    model_parameters = profile.setdefault("model_parameters", {})
    transmission_layer_parameters = model_parameters.setdefault("transmission_layer_parameters", {})

    for key, value in transmission_layer_parameters.items():
        transmission_layer_parameters[key] = tf.constant(value)

    parameter_distributions = model_parameters.setdefault("parameter_distributions", {})

    dtype_map = {
        "int8": tf.int8,
        "int16": tf.int16,
        "int32": tf.int32,
        "int64": tf.int64,
        "float16": tf.float16,
        "float32": tf.float32,
        "float64": tf.float64
    }

    for value in parameter_distributions.values():
        value["dtype"] = dtype_map[value["dtype"]]

    return profile


def _create_model(profile: dict[str, Any]) -> keras.Model:
    mapping = {
        "IMDDAutoencoder": IMDDAutoencoder,
        "IQAutoencoder": IQAutoencoder
    }

    return mapping[profile["model_type"]](**profile["model_parameters"])


class RevertToBest(keras.callbacks.Callback):
    def __init__(self, model: keras.models.Model, treshold=1.3) -> None:
        self._model = model
        self._best_loss = float("inf")
        self._best_weights = None
        self._treshold = treshold

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        if logs is None:
            return

        loss = logs["loss"]

        if loss < self._best_loss:
            self._best_loss, self._best_weights = loss, self._model.get_weights()
        elif loss / self._best_loss >= self._treshold:
            assert self._best_weights is not None
            print()
            print(f"Reverting weights to previous best: loss={self._best_loss}")
            self._model.set_weights(self._best_weights)


class TrainCommand(Command):
    description = "Train an autoencoder model"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train for")
        parser.add_argument("--batch-size", type=int, default=256, help="batch size for training")
        parser.add_argument("--store", help="store trained model in a .keras file")
        parser.add_argument("--load", help="load trained model from a .keras file")
        parser.add_argument("--device", default="/CPU:0", help="device to run the model on")
        parser.add_argument("--profile", help="simulation profile (JSON file)")
        parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard")
        parser.add_argument("--tensorboard-log-dir", default=".logs", help="TensorBoard log directory")

    def __call__(self, ns: argparse.Namespace) -> int:
        profile = _load_profile(ns.profile)

        with tf.device(ns.device):
            model = _create_model(profile)
            M = profile["model_parameters"]["M"]

            optimizer = keras.optimizers.Adam(**profile["optimizer_parameters"])
            model.compile(optimizer=optimizer, loss="binary_crossentropy")
            model.build((1, M))

            model.summary()

            callbacks: list[keras.callbacks.Callback] = []

            if ns.store:
                callbacks.append(keras.callbacks.ModelCheckpoint(
                    ns.store,
                    monitor="loss",
                    save_best_only=True,
                    save_weights_only=True,
                    save_freq="epoch"
                ))

            if ns.tensorboard:
                now = datetime.datetime.now()
                callbacks.append(keras.callbacks.TensorBoard(
                    log_dir=os.path.join(ns.tensorboard_log_dir, now.strftime("%Y-%m-%d-%H-%M-%S")),
                    histogram_freq=1
                ))

            callbacks.append(RevertToBest(model, treshold=1.3))

            # Create a training dataset of ~512000 input messages
            tile_factor = 512000 // M
            dataset = tf.tile(
                tf.cast(tf.one_hot(tf.range(M), M), tf.float32),
                [tile_factor, 1]
            )

            if ns.load:
                model.load_weights(ns.load)

            model.fit(
                dataset,
                dataset,
                epochs=ns.epochs,
                batch_size=ns.batch_size,
                shuffle=True,
                callbacks=callbacks
            )

        return 0


def _number(x: str) -> float | int:
    if "." in x or "e" in x.lower():
        return float(x)

    return int(x)


class BEROverParameterCommand(Command):
    description = "Evaluate autoencoder's BER as a function of chosen parameter."

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("model", help="path to trained model's .keras file")
        parser.add_argument("--device", default="/CPU:0", help="device to run the model on")
        parser.add_argument("--batch-size", type=int, default=1000, help="batch size for BERTester")
        parser.add_argument("--bitseq-length", default=None, type=int, help="test bit sequence length")
        parser.add_argument("--n-iterations", default=1, type=int, help="number of iterations for BERTester")
        parser.add_argument("--parameter", help="parameter name")
        parser.add_argument("--start", type=_number, help="minimum parameter value")
        parser.add_argument("--end", type=_number, help="maximum parameter value")
        parser.add_argument("--step", type=_number, help="parameter value step")
        parser.add_argument("--output", help="HDF5 output filename")
        parser.add_argument("--profile", help="simulation profile (JSON file)")

    def __call__(self, ns: argparse.Namespace) -> int:
        profile = _load_profile(ns.profile)

        with tf.device(ns.device):
            model: AutoencoderModel = _create_model(profile)
            model.load_weights(ns.model)
            M = profile["model_parameters"]["M"]

            bits_per_symbol = int(math.log2(M))

            # Autoencoder's encoder only takes one symbol per batch
            def reshape_for_encoder(x: tf.Tensor) -> tf.Tensor:
                batch_size, symbol_count, one_hot_size = tf.shape(x)

                return tf.reshape(x, [batch_size * symbol_count, one_hot_size])

            def reshape_for_transmitter(x: tf.Tensor) -> tf.Tensor:
                batch_size, sample_count = tf.shape(x)
                symbol_count = ns.bitseq_length // bits_per_symbol

                return tf.reshape(x, [batch_size // symbol_count, 1, sample_count * symbol_count])

            def reshape_for_decoder(x: tf.Tensor) -> tf.Tensor:
                batch_size, parameter_dims, sample_count = tf.shape(x)
                symbol_count = ns.bitseq_length // bits_per_symbol

                tf.debugging.assert_equal(parameter_dims, 1)

                return tf.reshape(x, [batch_size * symbol_count, sample_count // symbol_count])

            def decode_one_hot(x: tf.Tensor) -> tf.Tensor:
                bitseq = indices_to_bits(tf.cast(tf.argmax(x, axis=1), tf.int32), bits_per_symbol)

                return tf.reshape(bitseq, (ns.batch_size, 1, ns.bitseq_length))

            system = components.Composition([
                model.encoder,
                model.transmitter,
                model.channel,
                model.receiver,
                model.decoder
            ])

            ber_tester = components.BERTester(
                system,
                batch_size=ns.batch_size,
                n_iterations=ns.n_iterations,
                parameter_dims=1
            )

            ber_values = []
            parameter_values = tf.range(ns.start, ns.end, ns.step)

            if ns.bitseq_length is None:
                bitseq_length = tf.constant(bits_per_symbol * 100)
            else:
                bitseq_length = tf.constant(ns.bitseq_length)

            params = dict(profile["model_parameters"]["transmission_layer_parameters"])

            for parameter_value in parameter_values:
                params[ns.parameter] = tf.constant(parameter_value)

                ber = ber_tester(tf.constant(bitseq_length), **params)[0]

                print(f"BER={ber}, {ns.parameter}={parameter_value}")

                ber_values.append(ber)

            if ns.output is not None:
                with h5py.File(ns.output, "w") as hdf5:
                    hdf5.create_dataset("ber", data=ber_values)
                    hdf5.create_dataset(ns.parameter, data=parameter_values)

        return 0


def main(args: list) -> int:
    parser = argparse.ArgumentParser(
        description="Train or evaluate fiber-optic communication system autoencoder models"
    )

    subparsers = parser.add_subparsers(dest="command")

    commands: dict[str, Command] = {
        "train": TrainCommand(),
        "ber-over-parameter": BEROverParameterCommand()
    }

    for command_name, command in commands.items():
        command_parser = subparsers.add_parser(command_name, description=command.description)
        command.setup(command_parser)

    ns = parser.parse_args(args)

    if ns.command not in commands:
        parser.print_help()
        return 1

    return commands[ns.command](ns)


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))

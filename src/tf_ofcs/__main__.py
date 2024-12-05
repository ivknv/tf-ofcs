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
from typing import Any

import h5py
import keras
import tensorflow as tf

from .utils import dbm_to_w

from .models import IMDDAutoencoder
from . import components


class Command:
    description: str = ""

    def setup(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def __call__(self, ns: argparse.Namespace) -> int:
        raise NotImplementedError


def _load_profile(path: str) -> Any:
    with open(path, "rb") as profile_file:
        profile = json.load(profile_file)

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
    mapping = {"IMDDAutoencoder": IMDDAutoencoder}

    return mapping[profile["model_type"]](**profile["model_parameters"])


class TrainCommand(Command):
    description = "Train an IM/DD autoencoder"

    def setup(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train for")
        parser.add_argument("--batch-size", type=int, default=256, help="batch size for training")
        parser.add_argument("--store", help="store trained model in a .keras file")
        parser.add_argument("--load", help="load trained model from a .keras file")
        parser.add_argument("--device", default="/CPU:0", help="device to run the model on")
        parser.add_argument("--profile", help="simulation profile (JSON file)")

    def __call__(self, ns: argparse.Namespace) -> int:
        profile = _load_profile(ns.profile)

        with tf.device(ns.device):
            model = _create_model(profile)
            M = profile["model_parameters"]["M"]

            optimizer = keras.optimizers.Adam()
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

            # Create a training dataset of ~256000 input messages
            tile_factor = 256000 // M
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
            model = _create_model(profile)
            model.load_weights(ns.model)
            M = profile["model_parameters"]["M"]

            encoder = components.Lambda(model.encode_input)
            decoder = components.Lambda(model.decode_signal)

            lpf = components.BandPassFilter(min_frequency=0.0, max_frequency=32e9)
            laser = components.Laser(power=dbm_to_w(2.0))
            photodiode = components.Photodiode(
                maximum_power=dbm_to_w(3.0),
                sensitivity=dbm_to_w(-20.0)
            )

            channel = components.Composition([
                components.OpticalAWGN(relative_db=-30.0),
                components.OpticalFiber(
                    length=40.0,
                    attenuation=0.22,
                    chromatic_dispersion=17.0,
                    eff_non_linear_coeff=0.0,
                    step_count=1,
                    central_frequency=193.414e12
                )
            ])

            transmitter = components.Composition([
                encoder,
                lpf,
                components.MachZehnderModulator(carrier_source=laser)
            ])

            receiver = components.Composition([
                photodiode,
                lpf,
                components.ElectricalGain(gain=1e6),
                decoder
            ])

            system = components.Composition([transmitter, channel, receiver])

            ber_tester = components.BERTester(
                system,
                batch_size=ns.batch_size,
                n_iterations=ns.n_iterations,
                parameter_dims=1
            )

            ber_values = []
            parameter_values = tf.range(ns.start, ns.end, ns.step)

            if ns.bitseq_length is None:
                bitseq_length = tf.constant(int(math.log2(M) * 100))
            else:
                bitseq_length = tf.constant(ns.bitseq_length)

            params = dict(profile["model_parameters"]["transmission_layer_parameters"])

            for parameter_value in parameter_values:
                params[ns.parameter] = tf.constant(parameter_value)

                ber = ber_tester(tf.constant(bitseq_length), **params)[0]

                print(f"BER={ber}, {ns.parameter}={parameter_value}")

                ber_values.append(ber)

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

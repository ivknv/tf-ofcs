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
import json
import os
import time
import typing
from typing import Any, Optional
import datetime

import silence_tensorflow
silence_tensorflow.silence_tensorflow("WARNING")

import h5py  # type: ignore[import-untyped] # noqa: E402
import keras  # type: ignore[import-untyped] # noqa: E402
import tensorflow as tf # noqa: E402

from .models import IMDDAutoencoder, IQAutoencoder  # noqa: E402
from .reference_systems import PAMSystem, IQSystem  # noqa: E402
from . import components  # noqa: E402
from .profile import Profile  # noqa: E402
from .communication_system import CommunicationSystem  # noqa: E402
from .datatypes import TensorCompatible  # noqa: E402
from .message_generator import message_generator  # noqa: E402

class Command:
    description: str = ""

    def setup(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def __call__(self, ns: argparse.Namespace) -> int:
        raise NotImplementedError


def _load_profile(path: str) -> Profile:
    # Load profile from a JSON file.
    # The profile specifies the class that is used as implementation for the
    # communication system, as well as its parameters.
    # It also specifies channel implementation and parameters, training
    # parameters, and default BER test parameters.

    with open(path, "rb") as profile_file:
        js = json.load(profile_file)

    return Profile.from_json(js)


def _create_system(system_profile: Profile.System) -> CommunicationSystem:
    mapping = {
        "IMDDAutoencoder": IMDDAutoencoder,
        "IQAutoencoder": IQAutoencoder,
        "PAMSystem": PAMSystem,
        "QAMSystem": IQSystem
    }

    system = typing.cast(
        CommunicationSystem,
        mapping[system_profile.name](**system_profile.init_args)
    )

    return system


class OpticalFiberAWGN(components.Composition):
    def __init__(
        self,
        attenuation: TensorCompatible = tf.constant([0.2]),
        chromatic_dispersion: TensorCompatible = tf.constant([0.0]),
        eff_non_linear_coeff: TensorCompatible = tf.constant([1.76]),
        length: TensorCompatible = tf.constant([10.0]),
        central_frequency: float | tf.Tensor = 193.414e12,
        step_count: int | tf.Tensor = 3,
        enable_nonlinear_effects: bool = True,
        osnr: TensorCompatible = tf.constant([30.0]),
        padding: float | tf.Tensor = tf.constant(0.0),
        name: Optional[str] = None
    ) -> None:
        super().__init__([], name=name)

        if float(padding) > 1e-6:
            self.components.append(components.PadSignal(padding, name=f"{self.name}__pad_signal"))

        self.components.append(
            components.OpticalFiber(
                attenuation=attenuation,
                chromatic_dispersion=chromatic_dispersion,
                eff_non_linear_coeff=eff_non_linear_coeff,
                length=length,
                central_frequency=central_frequency,
                step_count=step_count,
                enable_nonlinear_effects=enable_nonlinear_effects,
                name=f"{self.name}__optical_fiber"
            )
        )

        if float(padding) > 1e-6:
            self.components.append(
                components.UnpadSignal(padding, name=f"{self.name}__unpad_signal")
            )

        self.components.append(
            components.OpticalAWGN(
                relative_db=-tf.convert_to_tensor(osnr),
                name=f"{self.name}__optical_awgn"
            )
        )


def _create_channel(channel_profile: Profile.Channel) -> components.Component:
    mapping = {
        "OpticalFiber": components.OpticalFiber,
        "OpticalFiberAWGN": OpticalFiberAWGN
    }

    return mapping[channel_profile.name](**channel_profile.init_args)


def _load_weights(model: keras.models.Model, path: str) -> None:
    if not path.endswith(".keras"):
        raise ValueError(f"Invalid file extension: expected a '.keras' file, got {repr(path)} instead")

    # Load only trainable weights, we don't care about the non-trainable ones
    with h5py.File(path, "r") as file:
        trainable_layers = [layer for layer in model.layers if layer.trainable]

        for layer in trainable_layers:
            for variable in layer.weights:
                variable.assign(file[f"{layer.name}/{variable.name}"])


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
        parser.add_argument(
            "--store",
            action="store_true",
            help="store trained model in a .keras file specified in the profile"
        )

        parser.add_argument(
            "--load",
            action="store_true",
            help="load trained model from a .keras file specified in the profile"
        )

        parser.add_argument("--device", default="/CPU:0", help="device to run the model on")
        parser.add_argument("--profile", help="simulation profile (JSON file)")

    def _setup_tensorboard(self, tensorboard_args: dict, base_directory: str) -> keras.callbacks.TensorBoard:
        now = datetime.datetime.now()

        tensorboard_args = dict(tensorboard_args)
        tensorboard_args["log_dir"] = os.path.join(
            base_directory,
            tensorboard_args.get("log_dir", ".logs"),
            now.strftime("%Y-%m-%d-%H-%M-%S")
        )
        tensorboard_args.setdefault("histogram_freq", 1)

        return keras.callbacks.TensorBoard(**tensorboard_args)

    def _setup_checkpoint_callback(self, model_path: Optional[str]) -> keras.callbacks.ModelCheckpoint:
        if model_path is None:
            raise ValueError("Model path was not specified in the profile")

        return keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            save_freq="epoch"
        )

    def _setup_model(
        self,
        system: CommunicationSystem,
        profile: Profile
    ) -> None:
        assert profile.training is not None

        channel_profile = profile.training.channel or profile.default_channel
        channel = _create_channel(channel_profile)

        setup_parameters = dict(profile.system.parameters)
        setup_parameters.update(profile.training.parameters)
        setup_parameters["batch_size"] = tf.constant(profile.training.batch_size)

        system.setup_model(
            channel,
            setup_parameters,
            profile.training.parameter_distributions
        )

        assert system.model is not None

        optimizer = keras.optimizers.Adam(**profile.training.optimizer_parameters)
        system.model.compile(optimizer=optimizer, loss="binary_crossentropy")

    def __call__(self, ns: argparse.Namespace) -> int:
        profile = _load_profile(ns.profile)

        with tf.device(ns.device):
            system = _create_system(profile.system)

            if system.model is None:
                raise ValueError("Specified model is not trainable")

            if profile.training is None:
                raise ValueError("Training config is not provided in the profile")

            self._setup_model(system, profile)
            system.model.summary()

            base_directory = os.path.dirname(ns.profile)

            if profile.system.model_path is not None:
                model_path = os.path.join(base_directory, profile.system.model_path)
            else:
                model_path = None

            callbacks: list[keras.callbacks.Callback] = []

            if ns.store:
                callbacks.append(self._setup_checkpoint_callback(model_path))

            if profile.training.tensorboard is not None:
                callbacks.append(self._setup_tensorboard(profile.training.tensorboard, base_directory))

            if profile.training.reduce_lr_on_plateau is not None:
                callbacks.append(
                    keras.callbacks.ReduceLROnPlateau(**profile.training.reduce_lr_on_plateau)
                )

            callbacks.append(RevertToBest(system.model, treshold=1.3))

            M = 2 ** system.bits_per_symbol

            # Create a training dataset of ~512000 input messages
            dataset = tf.data.Dataset.from_generator(
                message_generator,
                args=(M, profile.training.batch_size),
                output_signature=(
                    tf.TensorSpec(shape=(profile.training.batch_size, M), dtype=tf.float32),
                    tf.TensorSpec(shape=(profile.training.batch_size, M), dtype=tf.float32)
                )
            )

            if ns.load:
                if model_path is None:
                    raise ValueError("Model path was not specified in the profile")

                _load_weights(system.model, model_path)

            system.model.fit(
                dataset,
                epochs=profile.training.epochs,
                batch_size=profile.training.batch_size,
                steps_per_epoch=profile.training.steps_per_epoch,
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
        parser.add_argument("--device", default="/CPU:0", help="device to run the model on")
        parser.add_argument("--output", help="HDF5 output filename")
        parser.add_argument("--profile", help="simulation profile (JSON file)")

    def _load_model_weights(self, system: CommunicationSystem, profile_path: str, profile: Profile) -> None:
        if profile.system.model_path is not None and system.model is not None:
            model_path = os.path.join(os.path.dirname(profile_path), profile.system.model_path)
            _load_weights(system.model, model_path)

    def __call__(self, ns: argparse.Namespace) -> int:
        profile = _load_profile(ns.profile)

        with tf.device(ns.device):
            system: CommunicationSystem = _create_system(profile.system)

            if profile.ber_estimation is None:
                raise ValueError("BER estimation config is not provided in the profile")

            self._load_model_weights(system, ns.profile, profile)

            channel = _create_channel(profile.ber_estimation.channel or profile.default_channel)

            ber_tester = components.BERTester(
                system,
                batch_size=profile.ber_estimation.batch_size,
                n_iterations=profile.ber_estimation.n_iterations,
                block_size=profile.ber_estimation.block_size
            )

            parameter_name = profile.ber_estimation.parameter_name
            parameter_values = tf.range(
                profile.ber_estimation.start,
                profile.ber_estimation.end,
                profile.ber_estimation.step
            )

            bitseq_length = tf.constant(profile.ber_estimation.symbol_count * system.bits_per_symbol)

            system_parameters = dict(profile.system.parameters)
            system_parameters.update(profile.ber_estimation.parameters)
            system_parameters["batch_size"] = tf.constant(profile.ber_estimation.batch_size)

            ber_values = []

            # TODO: use time.monotonic() instead
            last_timestamp = time.time()

            for parameter_value in parameter_values:
                system_parameters[parameter_name] = tf.expand_dims(parameter_value, 0)

                system.setup(channel, system_parameters)
                ber = ber_tester(bitseq_length, **system_parameters)

                current_timestamp = time.time()
                step_duration = current_timestamp - last_timestamp
                last_timestamp = current_timestamp

                print(f"BER={ber}, {parameter_name}={parameter_value}, took {step_duration:.5} s")
                ber_values.append(ber)

            if ns.output is not None:
                with open(ns.profile, "rb") as profile_file:
                    profile_json = profile_file.read()

                with h5py.File(ns.output, "w") as hdf5:
                    hdf5.create_dataset("ber", data=ber_values)
                    hdf5.create_dataset(parameter_name, data=parameter_values)
                    hdf5.create_dataset("profile", data=profile_json)

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

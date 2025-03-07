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

from dataclasses import dataclass
from typing import Any, Optional

import tensorflow as tf

__all__ = ["Profile"]


def _convert_parameters(parameters: dict[str, Any]) -> dict[str, tf.Tensor]:
    return {key: tf.constant(value) for key, value in parameters.items()}


@dataclass
class Profile:
    @dataclass
    class System:
        name: str
        init_args: dict[str, Any]
        setup_args: dict[str, Any]
        parameters: dict[str, Any]
        model_path: Optional[str]

        @staticmethod
        def from_json(json: dict[str, Any]) -> "Profile.System":
            return Profile.System(
                name=json["name"],
                init_args=json["init_args"],
                setup_args=json.get("setup_args", {}),
                parameters=json["parameters"],
                model_path=json.get("model_path")
            )

    @dataclass
    class Channel:
        name: str
        init_args: dict[str, Any]

        @staticmethod
        def from_json(json: dict[str, Any]) -> "Profile.Channel":
            return Profile.Channel(
                name=json["name"],
                init_args=json.get("init_args", {})
            )

    @dataclass
    class Training:
        batch_size: int
        epochs: int
        parameters: dict[str, Any]
        parameter_distributions: dict[str, tf.Tensor]
        optimizer_parameters: dict[str, Any]
        channel: Optional["Profile.Channel"]
        tensorboard: Optional[dict[str, Any]]

        @staticmethod
        def from_json(json: dict[str, Any]) -> "Profile.Training":
            match json.get("channel"):
                case None:
                    channel = None
                case value:
                    channel = Profile.Channel.from_json(value)

            return Profile.Training(
                batch_size=json.get("batch_size", 512),
                epochs=json.get("epochs", 50),
                parameters=_convert_parameters(json.get("parameters", {})),
                parameter_distributions=Profile.Training._convert_parameter_distributions(
                    json.get("parameter_distributions", {})
                ),
                optimizer_parameters=json.get("optimizer_parameters", {}),
                channel=channel,
                tensorboard=json.get("tensorboard")
            )

        @staticmethod
        def _convert_parameter_distributions(parameter_distributions: dict[str, Any]) -> dict[str, tf.Tensor]:
            parameter_distributions = dict(parameter_distributions)
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

            return parameter_distributions

    @dataclass
    class BEREstimation:
        symbol_count: int
        batch_size: int
        n_iterations: int
        parameter_name: str
        start: int | float
        end: int | float
        step: int | float
        channel: Optional["Profile.Channel"]
        parameters: dict[str, Any]

        @staticmethod
        def from_json(json: dict[str, Any]) -> "Profile.BEREstimation":
            match json.get("channel"):
                case None:
                    channel = None
                case value:
                    channel = Profile.Channel.from_json(value)

            return Profile.BEREstimation(
                symbol_count=json["symbol_count"],
                batch_size=json["batch_size"],
                n_iterations=json["n_iterations"],
                parameters=_convert_parameters(json.get("parameters", {})),
                parameter_name=json["parameter_name"],
                start=json["start"],
                end=json["end"],
                step=json["step"],
                channel=channel
            )

    system: System
    default_channel: Channel
    training: Optional[Training]
    ber_estimation: Optional[BEREstimation]

    @staticmethod
    def from_json(json: dict[str, Any]) -> "Profile":
        match json.get("training"):
            case None:
                training = None
            case value:
                training = Profile.Training.from_json(value)

        match json.get("ber_estimation"):
            case None:
                ber_estimation = None
            case value:
                ber_estimation = Profile.BEREstimation.from_json(value)

        return Profile(
            system=Profile.System.from_json(json["system"]),
            default_channel=Profile.Channel.from_json(json["default_channel"]),
            training=training,
            ber_estimation=ber_estimation
        )

# -*- coding: utf-8
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

import typing
from typing import Any, Union

if typing.TYPE_CHECKING:
    from .components import Component
    import tensorflow as tf

__all__ = ["CommunicationSystem"]


class CommunicationSystem:
    """
        A protocol for communication systems.
        It is composed of an encoder, a decoder, a transmitter and a receiver.
        It doesn't need to be callable, as the channel is specified separately.
        The implementation can be a trainable model.

        :ivar encoder: `Component`, component that encodes the bit sequence
        :ivar decoder: `Component`, component that decodes the received signal
        :ivar channel: `Component`, communication channel
        :ivar transmitter: `Component`, component that transmits encoded signal
        :ivar receiver: `Component`, component that receives the transmitted signal
        :ivar sampling_rate: `int | tf.Tensor`, sampling rate of the system
        :ivar bits_per_symbol: `int | tf.Tensor`, number of bits per symbol
        :ivar component_parameters: ``dict[str, Any]`, component parameters for the system
        :ivar parameter_distributions: `dict[str, Any]`, distributions for parameter
                                       randomization to be used during training
    """

    encoder: "Component"
    decoder: "Component"

    channel: "Component"

    transmitter: "Component"
    receiver: "Component"

    sampling_rate: Union[float, "tf.Tensor"]
    bits_per_symbol: Union[int, "tf.Tensor"]

    component_parameters: dict[str, Any]
    parameter_distributions: dict[str, Any]

    def setup(self, channel: "Component", component_parameters: dict[str, Any]) -> None:
        """
            Sets the channel and component parameters,
            initializes the system for the provided channel and parameters.
            This could include channel estimation.
            This method is not meant to be used during training.

            :param channel: `Component`, communication channel that will be used from now on
            :param component_parameters: `dict[str, Any]`, component parameters
        """

        self.channel = channel
        self.component_parameters.clear()
        self.component_parameters.update(component_parameters)

    def model_setup(
        self,
        channel: "Component",
        component_parameters: dict[str, Any],
        parameter_distributions: dict[str, Any]
    ) -> None:
        """
            This method is called before model is compiled and possibly trained but after
            all the relevant parameters are known. This could used to set the
            channel, pre-compute some constants or initialize some variables.

            This method is relevant only for trainable models.

            :param channel: `Component`, communication channel that will be used from now on
            :param component_parameters: `dict[str, Any]`, component parameters
            :param parameter_distributions: `dict[str, Any]`, distributions for parameter
                                            randomization to be used during training
        """

        self.channel = channel
        self.component_parameters.clear()
        self.component_parameters.update(component_parameters)

        self.parameter_distributions.clear()
        self.parameter_distributions.update(parameter_distributions)

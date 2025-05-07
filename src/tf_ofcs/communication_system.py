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

from typing import Any, Optional

import keras  # type: ignore[import-untyped]
from .components import Composition, Component, Substitute, Identity

__all__ = ["CommunicationSystem"]


class CommunicationSystem(Composition):
    """
        A base class for communication systems.
        It is composed of an encoder, a decoder, a transmitter and a receiver.
        It doesn't need to be callable, as the channel is specified separately.
        The implementation can be a trainable model.

        :ivar encoder: `Component`, component that encodes the bit sequence
        :ivar decoder: `Component`, component that decodes the received signal
        :ivar channel: `Component`, communication channel
        :ivar transmitter: `Component`, component that transmits encoded signal
        :ivar receiver: `Component`, component that receives the transmitted signal
        :ivar sampling_rate: `int`, sampling rate of the system
        :ivar bits_per_symbol: `int`, number of bits per symbol
        :ivar component_parameters: ``dict[str, Any]`, component parameters for the system
        :ivar model: `Optional[keras.models.Model]`, underlying Keras model (if applicable).
                     Must implement `AutoencoderModel` protocol if not `None`.
    """

    encoder: "Component"
    decoder: "Component"

    channel: "Component"

    transmitter: "Component"
    receiver: "Component"

    bits_per_symbol: int

    component_parameters: dict[str, Any]

    model: Optional[keras.models.Model]

    def __init__(
        self,
        *,
        bits_per_symbol: int,
        encoder: Component,
        decoder: Component,
        transmitter: Component,
        receiver: Component,
        component_parameters: Optional[dict[str, Any]] = None,
        model: Optional[keras.models.Model] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__([], name=name)

        if component_parameters is None:
            component_parameters = {}

        self.component_parameters = component_parameters
        self.model = model

        self.bits_per_symbol = bits_per_symbol
        self.encoder = encoder
        self.decoder = decoder
        self.transmitter = transmitter
        self.receiver = receiver
        self.channel = Substitute(Identity())

        self.components = [
            self.encoder,
            self.transmitter,
            self.channel,
            self.receiver,
            self.decoder
        ]

    def setup(
        self,
        channel: Component,
        component_parameters: dict[str, Any]
    ) -> None:
        """
            Sets the channel and component parameters, initializes the system
            for the provided channel and parameters.
            This could include channel estimation.
            This method is not meant to be used during training.

            :param channel: `Component`, communication channel that will be used from now on
            :param component_parameters: `dict[str, Any]`, component parameters
            :param training: `bool`, specifies if this method is called during model training
        """

        self.channel.substitute = channel  # type: ignore[attr-defined]
        self.component_parameters.clear()
        self.component_parameters.update(component_parameters)

    def setup_model(
        self,
        channel: Component,
        component_parameters: dict[str, Any],
        validation_component_parameters: dict[str, Any],
        parameter_distributions: dict[str, Any],
        validation_parameter_distributions: dict[str, Any]
    ) -> None:
        """
            Sets the channel, component parameters and parameter distributions
            to be used during model training.

            :param channel: `Component`, communication channel that will be used from now on
            :param component_parameters: `dict[str, Any]`, component parameters
            :param validation_component_parameters: `dict[str, Any`, component parameters
                to be used for validation
            :param parameter_distributions: `dict[str, Any]`, distributions for parameter
                randomization to be used during training
            :param validation_parameter_distributions: `dict[str, Any]`, distributions
                for parameter randomization to be used during validation
        """

        raise NotImplementedError

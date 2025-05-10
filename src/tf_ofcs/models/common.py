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

from typing import TypedDict

__all__ = ["TransmitterConfig", "ReceiverConfig"]


class TransmitterConfig(TypedDict, total=False):
    """
        Configuration parameters for the transmitter section of the autoencoder model.

        :param dsp_sampling_rate: `float`, DSP sampling rate (samples per second).
        :param dac_sampling_rate: `float`, DAC sampling rate (samples per second).
        :param N: `int`, number of total symbols (main symbol + adjacent symbols) sent by the transmitter.
        :param n_layers: `int`, number of transmitter layers in the neural network.
        :param trainable: `bool`, specifies whether the transmitter layers are trainable.
        :param enable_batch_normalization: `bool`, specifeis whether to enable batch normalization.
        :param l2_regularization: `float`, L2 regularization factor.
    """

    dsp_sampling_rate: float
    dac_sampling_rate: float
    N: int
    n_layers: int
    trainable: bool
    enable_batch_normalization: bool
    l2_regularization: float


class ReceiverConfig(TypedDict, total=False):
    """
        Configuration parameters for the receiver section of the autoencoder model.

        :param adc_sampling_rate: `float`, ADC sampling rate (samples per second).
        :param N: `int`, number of symbols received by the receiver (sliding window length).
        :param n_layers: `int`, number of receiver layers in the neural network.
        :param trainable: `bool`, specifies whether the receiver layers are trainable.
        :param enable_batch_normalization: `bool`, specifeis whether to enable batch normalization.
        :param l2_regularization: `float`, L2 regularization factor.
    """

    adc_sampling_rate: float
    N: int
    n_layers: int
    trainable: bool
    enable_batch_normalization: bool
    l2_regularization: float

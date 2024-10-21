# -*- coding: utf-8 -*-

import tensorflow as tf

from optical_communication_system.components import (
    IQEncoder, IQDecoder, IQSymbolEncoder, IQSymbolDecoder, IQModulator,
    IQReceiver, Composition, Laser, BandPassFilter, OpticalAWGN, OpticalFiber,
    Photodiode, Quantizer, MapComponent
)
from optical_communication_system.utils import dbm_to_w, from_db, wavelength_nm_to_frequency


def test_qam4_system() -> None:
    data_rate = 40e9
    sampling_rate = 10 * data_rate
    bits_per_symbol = 2
    input_power = dbm_to_w(2.0)
    batch_size = 3

    lpf = BandPassFilter(min_frequency=0.0, max_frequency=0.5 * data_rate)

    rng = tf.random.Generator.from_seed(1234, alg="philox")
    snr = 30.0

    attenuation = 0.22
    length = 40.0
    expected_symbol_scale = input_power * from_db(-attenuation * length)

    mapping = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]

    laser = Laser(power=input_power, batch_size=batch_size)
    local_oscillator = laser

    expected_symbol_scale = tf.sqrt(input_power * from_db(-attenuation * length))

    system = Composition([
        IQEncoder(
            sampling_rate=sampling_rate,
            data_rate=data_rate,
            symbol_encoder=IQSymbolEncoder(
                bits_per_symbol=bits_per_symbol,
                mapping=mapping,
                adjust_for_mzm=True
            )
        ),
        MapComponent(lpf),
        IQModulator(laser),
        OpticalAWGN(relative_db=-snr, rng=rng),
        OpticalFiber(
             attenuation=attenuation,
             chromatic_dispersion=4.0,
             eff_non_linear_coeff=1.76,
             length=length,
             central_frequency=wavelength_nm_to_frequency(1550.0),
             step_count=100
        ),
        IQReceiver(
            photodiode=Composition([Photodiode(), lpf]),
            local_oscillator=local_oscillator
        ),
        MapComponent(Quantizer(1e-6)),
        IQDecoder(
            data_rate=data_rate,
            symbol_decoder=IQSymbolDecoder(
                bits_per_symbol=bits_per_symbol,
                mapping=mapping
            ),
            symbol_scale=expected_symbol_scale
        )
    ])

    input_sequence = tf.constant([
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    ])

    output_sequence = system(input_sequence, parameter_dims=2)
    expected_output_sequence = tf.tile(tf.expand_dims(input_sequence, 1), [1, 2, 1])

    assert isinstance(output_sequence, tf.Tensor)
    assert output_sequence.dtype is tf.int32
    assert tf.math.reduce_all(tf.shape(output_sequence) == (3, 2, 16))  # type: ignore

    assert tf.math.reduce_all(expected_output_sequence == output_sequence)  # type: ignore

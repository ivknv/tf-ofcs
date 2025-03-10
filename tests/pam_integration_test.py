# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import (
    PAMSymbolEncoder, PAMSymbolDecoder, Composition, Laser, BandPassFilter,
    MachZehnderModulator, OpticalAWGN, OpticalFiber, Photodiode,
    Gain, Fork, MapComponents, Identity, DigitalToAnalogConverter,
    AnalogToDigitalConverter, MZMPreDistortion, PhotodiodePreDistortion,
    Upsample, Downsample
)
from tf_ofcs.utils import dbm_to_w, from_db, wavelength_nm_to_frequency


def test_pam4_system() -> None:
    data_rate = 20e9
    bits_per_symbol = 2
    symbol_rate = data_rate / bits_per_symbol
    input_power = dbm_to_w(2.0)
    maximum_power = dbm_to_w(tf.constant([-1.0]))
    dac_step = 1.0 / 256.0
    adc_step = maximum_power / 256.0
    adc_sampling_rate = symbol_rate * 1

    lpf = BandPassFilter(min_frequency=[0.0], max_frequency=[1 * data_rate])

    rng = tf.random.Generator.from_seed(1234, alg="philox")
    snr = 40.0
    batch_size = 3

    attenuation = 0.2
    length = 30.0
    expected_symbol_scale = input_power * from_db(-attenuation * length)

    mapping = tf.constant([0.0, 1 / 3, 2 / 3, 1.0])

    system = Composition([
        PAMSymbolEncoder(
            bits_per_symbol=bits_per_symbol,
            mapping=mapping,
            symbol_rate=symbol_rate
        ),  # DigitalSignal
        PhotodiodePreDistortion(),  # DigitalSignal
        MZMPreDistortion(),  # DigitalSignal
        DigitalToAnalogConverter(
            step=dac_step,
            min_level=0.0,
            max_level=1.0,
            sampling_rate=adc_sampling_rate
        ),  # ElectricalSignal

        Upsample(4), # ElectricalSignal
        lpf,  # ElectricalSignal
        Fork(shape=[2]),  # (ElectricalSignal, ElectricalSignal)
        MapComponents([
            Laser(power=[input_power], batch_size=batch_size),
            Identity()
        ]),  # (OpticalSignal, ElectricalSignal)
        MachZehnderModulator(),  # OpticalSignal

        OpticalAWGN(relative_db=[-snr], rng=rng),  # OpticalSignal
        OpticalFiber(
             attenuation=[attenuation],
             chromatic_dispersion=[4.0],
             eff_non_linear_coeff=[1.76],
             length=[length],
             central_frequency=wavelength_nm_to_frequency(1550.0),
             step_count=3
        ),  # OpticalSignal

        Photodiode(
            dark_current=dbm_to_w(tf.constant([-15.0])),
            maximum_power=maximum_power
        ),  # ElectricalSignal
        lpf,  # ElectricalSignal

        AnalogToDigitalConverter(
            step=adc_step,
            min_level=0.0,
            max_level=maximum_power,
            sampling_rate=adc_sampling_rate
        ),  # DigitalSignal
        Downsample(factor=int(adc_sampling_rate / symbol_rate), time_offset=0.5 * symbol_rate),
        Gain(1.0 / expected_symbol_scale),  # DigitalSignal
        PAMSymbolDecoder(
            bits_per_symbol=bits_per_symbol,
            mapping=mapping
        )  # tf.Tensor, dtype=tf.int32
    ])

    input_sequence = tf.constant([
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    ])

    output_sequence = system(input_sequence)
    expected_output_sequence = input_sequence

    assert isinstance(output_sequence, tf.Tensor)
    assert output_sequence.dtype is tf.int32
    assert tf.math.reduce_all(tf.shape(output_sequence) == (3, 16))  # type: ignore[operator]

    assert tf.math.reduce_all(expected_output_sequence == output_sequence)  # type: ignore[operator]

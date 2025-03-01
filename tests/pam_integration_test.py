# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import (
    PAMEncoder, PAMDecoder, PAMSymbolEncoder, PAMSymbolDecoder,
    Composition, Laser, BandPassFilter, MachZehnderModulator, OpticalAWGN,
    OpticalFiber, Photodiode, Quantizer, ElectricalGain, Fork, MapComponents,
    Identity
)
from tf_ofcs.utils import dbm_to_w, from_db, wavelength_nm_to_frequency


def test_pam2_system() -> None:
    data_rate = 25e9
    sampling_rate = 10 * data_rate
    bits_per_symbol = 2
    input_power = dbm_to_w(2.0)
    batch_size = 3

    lpf = BandPassFilter(min_frequency=[0.0], max_frequency=[1 * data_rate])

    rng = tf.random.Generator.from_seed(1234, alg="philox")
    snr = 30.0

    attenuation = 0.22
    length = 40.0
    expected_symbol_scale = input_power * from_db(-attenuation * length)

    mapping = tf.sqrt([0.0, 1 / 3, 2 / 3, 1.0])

    system = Composition([
        PAMEncoder(
            sampling_rate=sampling_rate,
            data_rate=data_rate,
            symbol_encoder=PAMSymbolEncoder(
                bits_per_symbol=bits_per_symbol,
                mapping=mapping,
                adjust_for_mzm=True
            )
        ),  # ElectricalSignal
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
             step_count=100
        ),  # OpticalSignal
        Photodiode(
            sensitivity=dbm_to_w(tf.constant([-15.0])),
            maximum_power=dbm_to_w(tf.constant([-1.0]))
        ),  # ElectricalSignal
        lpf,  # ElectricalSignal
        Quantizer(step=[1e-6]),  # ElectricalSignal
        ElectricalGain(tf.constant([1.0]) / expected_symbol_scale),  # ElectricalSignal
        PAMDecoder(
            data_rate=data_rate,
            symbol_decoder=PAMSymbolDecoder(
                bits_per_symbol=bits_per_symbol,
                mapping=mapping ** 2
            )
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

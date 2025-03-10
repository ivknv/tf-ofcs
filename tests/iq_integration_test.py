# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import (
    IQEncoder, IQDecoder, IQSymbolEncoder, IQSymbolDecoder, IQModulator,
    IQReceiver, Composition, Laser, BandPassFilter, OpticalAWGN, OpticalFiber,
    Photodiode, MapComponents, ElectricalGain, Fork, Reindex, Identity
)
from tf_ofcs.utils import dbm_to_w, from_db, wavelength_nm_to_frequency


def test_qam4_system() -> None:
    data_rate = 40e9
    sampling_rate = 10 * data_rate
    bits_per_symbol = 2
    input_power = dbm_to_w(2.0)
    batch_size = 3

    lpf = BandPassFilter(min_frequency=[0.0], max_frequency=[0.5 * data_rate])

    rng = tf.random.Generator.from_seed(1234, alg="philox")
    snr = 30.0

    attenuation = 0.22
    length = 40.0
    expected_symbol_scale = input_power * from_db(-attenuation * length)

    mapping = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]

    laser = Laser(power=[input_power], batch_size=batch_size)
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
        ),  # (ElectricalSignal, ElectricalSignal)
        MapComponents(lpf),  # (ElectricalSignal, ElectricalSignal)
        Fork(shape=[2]),  # ((ElectricalSignal, ElectricalSignal), (ElectricalSignal, ElectricalSignal))
        MapComponents([
            Composition([
                Reindex(indices=(0,)),  # ElectricalSignal
                laser  # OpticalSignal
            ]),
            Identity()  # (ElectricalSignal, ElectricalSignal)
        ]),  # (OpticalSignal, (ElectricalSignal, ElectricalSignal))
        IQModulator(),  # OpticalSignal
        OpticalAWGN(relative_db=[-snr], rng=rng),  # OpticalSignal
        OpticalFiber(
             attenuation=[attenuation],
             chromatic_dispersion=[4.0],
             eff_non_linear_coeff=[1.76],
             length=[length],
             central_frequency=wavelength_nm_to_frequency(1550.0),
             step_count=100
        ),  # OpticalSignal
        IQReceiver(
            photodiode=Composition([Photodiode(), lpf]),
            local_oscillator=local_oscillator
        ),  # (ElectricalSignal, ElectricalSignal)
        MapComponents(ElectricalGain([1.0 / expected_symbol_scale])),  # (ElectricalSignal, ElectricalSignal)
        IQDecoder(
            data_rate=data_rate,
            symbol_decoder=IQSymbolDecoder(
                bits_per_symbol=bits_per_symbol,
                mapping=mapping
            )
        )  # tf.Tensor, dtype=tf.int32
    ])

    input_sequence = tf.constant([
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    ])

    output_sequence = system.trace(input_sequence)
    expected_output_sequence = input_sequence

    assert isinstance(output_sequence, tf.Tensor)
    assert output_sequence.dtype is tf.int32
    assert tf.math.reduce_all(tf.shape(output_sequence) == (3, 16))  # type: ignore[operator]

    assert tf.math.reduce_all(expected_output_sequence == output_sequence)  # type: ignore[operator]

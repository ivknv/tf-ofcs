# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import Composition, OpticalAWGN, OpticalFiber
from tf_ofcs.reference_systems import PAMSystem
from tf_ofcs.utils import wavelength_nm_to_frequency


def test_pam4_system() -> None:
    data_rate = 42e9
    snr = 30.0

    system = PAMSystem(bits_per_symbol=2, data_rate=data_rate, n_taps=13)
    system.rng.reset_from_seed(1234)

    attenuation = 0.2
    length = 10.0

    channel = Composition([
        OpticalAWGN(relative_db=[-snr], rng=system.rng),
        OpticalFiber(
             attenuation=[attenuation],
             chromatic_dispersion=[4.0],
             eff_non_linear_coeff=[1.76],
             length=[length],
             central_frequency=wavelength_nm_to_frequency(1550.0),
             step_count=1
        )
    ])

    input_sequence = tf.constant([
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
    ])

    system.setup(channel, {})

    output_sequence = system(input_sequence)
    expected_output_sequence = input_sequence

    assert isinstance(output_sequence, tf.Tensor)
    assert output_sequence.dtype is tf.int32
    assert tf.math.reduce_all(tf.shape(output_sequence) == (3, 16))  # type: ignore[operator]

    assert tf.math.reduce_all(expected_output_sequence == output_sequence)  # type: ignore[operator]

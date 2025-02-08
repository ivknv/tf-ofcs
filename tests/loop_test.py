# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import Loop, Lambda


def test_loop() -> None:
    loop = Loop(Lambda(lambda x: x + 1), n=10)

    assert tf.math.reduce_all(loop(0) == 10)  # type: ignore[operator]
    assert tf.math.reduce_all(loop(3, loop__n=50) == 53)  # type: ignore[operator]

# -*- coding: utf-8 -*-

import math
import tensorflow as tf

__all__ = ["c", "pi", "e"]

#: Speed of light in vacuum in m/s
c = tf.Variable(tf.constant(2.99792458E8))

pi = tf.Variable(tf.constant(math.pi))
e = tf.Variable(tf.constant(math.e))

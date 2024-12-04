# -*- coding: utf-8
# Copyright (C) 2024 Ivan Konovalov

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

import argparse

import keras
import tensorflow as tf

from .models import IMDDAutoencoder


def main(args: list) -> int:
    parser = argparse.ArgumentParser(
        description="Train an IM/DD autoencoder"
    )

    parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for training")
    parser.add_argument("--store", help="store trained model in a .keras file")
    parser.add_argument("--load", help="load trained model from a .keras file")
    parser.add_argument("--device", default="/CPU:0", help="device to run the model on")
    parser.add_argument("-M", default=64, help="M value")
    parser.add_argument("-N", default=11, help="N value")
    parser.add_argument("-n", default=48, help="n value")

    ns = parser.parse_args(args)

    with tf.device(ns.device):
        model = IMDDAutoencoder(M=int(ns.M), n=int(ns.n), N=int(ns.N))
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="binary_crossentropy")
        model.build((1, model.M))

        model.summary()

        # Create a training dataset of ~256000 input messages
        tile_factor = 256000 // model.M
        dataset = tf.tile(
            tf.cast(tf.one_hot(tf.range(model.M), model.M), tf.float32),
            [tile_factor, 1]
        )

        if ns.load:
            model.load_weights(ns.load)

        model.fit(
            dataset,
            dataset,
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            shuffle=True
        )

        if ns.store:
            model.save_weights(ns.store, overwrite=True)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))

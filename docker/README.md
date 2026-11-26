This directory contains Docker configs and some helper scripts to make it
easier to run [tf-ofcs](https://github.com/ivknv/tf-ofcs) in a Docker
container. The setup is made to run on AMD GPUs with ROCm. The docker-compose
config also specifies an environment variable
`HSA_OVERRIDE_GFX_VERSION=10.3.0`. This is needed to run the setup on RX
6600. All scripts pass `--device=/GPU:0`, this makes TensorFlow use the first
GPU.

It is assumed that the `tf-ofcs` main repo is located in the parent directory.

`./output/` directory can be used for persistent file I/O.

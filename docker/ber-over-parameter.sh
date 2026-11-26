#!/bin/bash

docker-compose -f docker/docker-compose.yaml run --rm tf-ofcs ber-over-parameter --device=/GPU:0 $@

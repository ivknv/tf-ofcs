#!/bin/bash

docker-compose -f docker/docker-compose.yaml run --rm tf-ofcs confusion-matrix --device=/GPU:0 $@

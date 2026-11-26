#!/bin/bash

docker-compose -f docker/docker-compose.yaml run --rm tf-ofcs train --device=/GPU:0 $@

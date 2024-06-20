#!/usr/bin/env bash

./build.sh

docker save joeranbosma/dragon_cltl_medroberta.nl_domain_specific:latest | gzip -c > dragon_cltl_medroberta.nl_domain_specific.tar.gz

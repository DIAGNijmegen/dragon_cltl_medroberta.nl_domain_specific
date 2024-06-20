#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t joeranbosma/dragon_cltl_medroberta.nl_domain_specific:latest -t joeranbosma/dragon_cltl_medroberta.nl_domain_specific:v0.2.1 "$SCRIPTPATH"

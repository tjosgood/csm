#!/usr/bin/env bash

rm -rf latest
git clone --depth=1 https://github.com/tjosgood/csm.git latest
cd latest || exit
rm  test_input.json

python -u rp_handler.py
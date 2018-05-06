#!/bin/sh
for k in 1 9 16
do
    echo "Running $k value"
    python train.py --config ../config/chinna/char_lm/flat_memory/k_values/ptb_$k.yaml --force_restart True
done

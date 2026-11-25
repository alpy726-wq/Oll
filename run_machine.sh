#!/bin/sh
# Train + evaluate the machine and its baselines (4 single-thread jobs at a time).
python3 machine.py persistence
run() { python3 machine.py train --kind $1 --seed $2 > runs/machine/$1_s$2.log 2>&1 && python3 machine.py eval --kind $1 --seed $2 >> runs/machine/$1_s$2.log 2>&1; }
run machine 0 & run machine 1 & run machine 2 & run gru_mdn 0 & wait
run gru_mdn 1 & run gru_mdn 2 & run ct_mdn 0 & run ct_gauss 0 & wait
python3 machine.py report > runs/machine/summary.txt
echo MACHINE_DONE

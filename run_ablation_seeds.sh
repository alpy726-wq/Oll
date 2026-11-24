#!/bin/sh
A="--R 128 --L 6 --K 128 --B 16 --T 256 --steps 1000 --eval_every 1000 --eval_n 48 --lr 6e-3"
for s in 1 2; do
  for pair in "sqrt_p log_p" "random geometric"; do
    set -- $pair
    THREADS=2 python3 train_lm.py $A --seed $s --family $1 --out runs/abl_${1}_s$s > runs/abl_${1}_s$s.log 2>&1 &
    THREADS=2 python3 train_lm.py $A --seed $s --family $2 --out runs/abl_${2}_s$s > runs/abl_${2}_s$s.log 2>&1
    wait
  done
  THREADS=4 python3 train_lm.py $A --seed $s --family table --out runs/abl_table_s$s > runs/abl_table_s$s.log 2>&1
done
echo SEEDS_DONE

#!/bin/sh
# Stage 2: everything that needs runs/main.pt, then the basis ablation and recall comparison.
set -x
C="--init runs/main --B 8 --T 512 --steps 300 --eval_every 300 --eval_n 32 --eval_domains prose,code,json,vimdoc,cheader"
THREADS=2 python3 train_lm.py $C --train phase --domains vimdoc --lr 4e-3 --out runs/adapt_phase_vim > runs/adapt_phase_vim.log 2>&1 &
THREADS=2 python3 train_lm.py $C --train dense --domains vimdoc --lr 5e-3 --phase_lr_mult 1 --out runs/adapt_dense_vim > runs/adapt_dense_vim.log 2>&1
wait
THREADS=2 python3 train_lm.py $C --train phase --domains cheader --lr 4e-3 --out runs/adapt_phase_ch > runs/adapt_phase_ch.log 2>&1 &
THREADS=2 python3 train_lm.py $C --train all --domains prose --lr 2e-3 --out runs/branch_prose > runs/branch_prose.log 2>&1
wait
THREADS=3 python3 train_lm.py $C --train all --domains code --lr 2e-3 --out runs/branch_code > runs/branch_code.log 2>&1
python3 exp_adapt.py 3 > runs/E7.log 2>&1
python3 exp_props.py 3 > runs/E2.log 2>&1
A="--R 128 --L 6 --K 128 --B 16 --T 256 --steps 1000 --eval_every 1000 --eval_n 48 --lr 6e-3"
for pair in "sqrt_p log_p" "random geometric"; do
  set -- $pair
  THREADS=2 python3 train_lm.py $A --family $1 --out runs/abl_$1 > runs/abl_$1.log 2>&1 &
  THREADS=2 python3 train_lm.py $A --family $2 --out runs/abl_$2 > runs/abl_$2.log 2>&1
  wait
done
THREADS=2 python3 train_lm.py $A --family table --out runs/abl_table > runs/abl_table.log 2>&1 &
python3 exp_recall.py 2 1500 prime > runs/E8_prime.log 2>&1
wait
python3 exp_recall.py 3 1500 transformer > runs/E8_tf.log 2>&1
echo STAGE2_DONE

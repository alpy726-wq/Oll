#!/bin/sh
# Issues 2, 4, 5: joint vs staged curricula and component ablations, 3 seeds each, real data only.
cd "$(dirname "$0")"
{
for s in 0 1 2; do
  echo "--schedule joint --seed $s --out runs/real/joint_s$s"
  echo "--schedule staged --seed $s --out runs/real/staged_s$s"
done
for ab in memory phaselinear kuramoto flatreadout; do
  for s in 0 1 2; do echo "--schedule joint --ablate $ab --seed $s --out runs/real/abl_${ab}_s$s"; done
done
} | xargs -P 2 -I{} sh -c 'python3 real_train.py {} > /dev/null 2>&1; echo done {}'
echo QUEUE_DONE

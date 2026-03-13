set -euo pipefail

# Baseline: 1 gradient step per batch
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -lr 0.001 -rtg \
  --exp_name qb2_b5000_lr0.001_ngspb1 --num_grad_steps_per_batch 1 \
  > data/qb2_b5000_lr0.001_ngspb1.log 2>&1 &

# 5 gradient steps per batch
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -lr 0.001 -rtg \
  --exp_name qb2_b5000_lr0.001_ngspb5 --num_grad_steps_per_batch 5 \
  > data/qb2_b5000_lr0.001_ngspb5.log 2>&1 &

# 25 gradient steps per batch
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -lr 0.001 -rtg \
  --exp_name qb2_b5000_lr0.001_ngspb25 --num_grad_steps_per_batch 25 \
  > data/qb2_b5000_lr0.001_ngspb25.log 2>&1 &

wait
echo "All jobs finished."

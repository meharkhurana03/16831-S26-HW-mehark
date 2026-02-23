set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 \
  --exp_name q4_b30000_r0.02 > data/q4_b30000_r0.02.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 -rtg \
  --exp_name q4_b30000_r0.02_rtg > data/q4_b30000_r0.02_rtg.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 --nn_baseline \
  --exp_name q4_b30000_r0.02_nnbaseline > data/q4_b30000_r0.02_nnbaseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline \
  --exp_name q4_b30000_r0.02_rtg_nnbaseline > data/q4_b30000_r0.02_rtg_nnbaseline.log 2>&1 &

wait
echo "All jobs finished."

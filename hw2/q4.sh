
batch_sizes=(10000 30000 50000)
lr_values=(0.005 0.01 0.02)

set -euo pipefail

pids=()

for b in "${batch_sizes[@]}"; do
  for lr in "${lr_values[@]}"; do
    if [ ${b} -eq 50000 ]; then
      CUDA_VISIBLE_DEVICES=1 python rob831/scripts/run_hw2.py \
        --env_name HalfCheetah-v4 \
        --ep_len 150 \
        --discount 0.95 \
        -n 100 -l 2 -s 32 \
        -b ${b} -lr ${lr} \
        -rtg \
        --nn_baseline \
        --exp_name q4_search_b${b}_lr${lr}_rtg_nnbaseline \
        > data/q4_b${b}_lr${lr}.log 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=0 python rob831/scripts/run_hw2.py \
        --env_name HalfCheetah-v4 \
        --ep_len 150 \
        --discount 0.95 \
        -n 100 -l 2 -s 32 \
        -b ${b} -lr ${lr} \
        -rtg \
        --nn_baseline \
        --exp_name q4_search_b${b}_lr${lr}_rtg_nnbaseline \
        > data/q4_b${b}_lr${lr}.log 2>&1 &
    fi
    pids+=($!)
    echo "Launched b=${b} lr=${lr} (pid $!)"
  done
done

echo "Waiting for ${#pids[@]} jobs..."
for pid in "${pids[@]}"; do
  wait "$pid" && echo "Job $pid done" || echo "Job $pid failed"
done
echo "All jobs finished."

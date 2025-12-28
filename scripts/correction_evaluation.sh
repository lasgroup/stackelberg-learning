#!/bin/bash

model_names=(
  "Qwen2.5"
  "RLOO"
  "Nash"
  "Stackelberg"
)
model_paths=(
  "unsloth/Qwen2.5-0.5B-Instruct"
  "path/to/rloo/checkpoint"
  "path/to/nash/checkpoint"
  "path/to/stackelberg/checkpoint"
)
datasets=(
  "path/to/qwen/generation/dataset"
  "path/to/rloo/generation/dataset"
  "path/to/nash/generation/dataset"
  "path/to/stackelberg/generation/dataset"
)

output_dir="$(pwd)/data/experiments/correction_comparison"

# Iterate over the model names twice with indices
for i in "${!model_names[@]}"; do
  for j in "${!model_names[@]}"; do
    leader_name="${model_names[$i]}"
    follower_name="${model_names[$j]}"
    data_path="${datasets[$i]}"
    model_path="${model_paths[$j]}"
    output_path="${output_dir}/Leader-${leader_name}__Follower-${follower_name}"

    # Check that output_path exists if yes continue
    if [ -d "${output_path}__rewards" ]; then
      echo "${job_name} rewards already exists at: "${output_path}__rewards""
      continue
    fi

    slurm_job_name="correction_evaluation_Leader-${leader_name}__Follower-${follower_name}"
    echo "Submitting job: $slurm_job_name"
    port=$((base_port + i * 10 + j + 1))


    python src/evaluation/text_generation.py \
    --dataset_name "${data_path}" \
    --model_name_or_path "${model_path}" \
    --output_path "${output_path}" \
    --torch_dtype bfloat16 \
    --num_return_sequences 2 \
    --temperature 0.9 \
    --max_new_tokens 512 \
    --follower_prompt "Improve the previous answer. Phrase it as if it was the original response." \
    --given_leader_completions=True \
    --batch_size 4

    python src/evaluation/reward_evaluation.py \
    --reward_model_name_or_path unsloth/Qwen2.5-1.5B-Instruct \
    --reward_model_adapters_path "path/to/reward/model/adapters" \
    --dataset_path "${output_path}" \
    --batch_size 1
  done
done
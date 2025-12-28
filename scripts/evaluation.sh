#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Experiment directory
experiment_dir="$(pwd)/data/experiments/experiment_name"

# Model path
checkpoint_name="checkpoint-x"
model_path="${experiment_dir}/${checkpoint_name}"
reward_adapters_path="path/to/reward/model/adapters"

# Output Path
output_path="${experiment_dir}/generation__${checkpoint_name}"

# Inputs and configs
dataset_path="path/to/preprocessed_helpsteer2_dataset"
# If you would like to generate improvements based on previously generated completions, set this to True
# and provide the path to the dataset with the generated completions (output of the text_generation.py script).
given_leader_completions="False"

echo "Model path: ${model_path}"
echo "Output path: ${output_path}"

echo "Creating texts with the model ${model_path}"
python src/evaluation/text_generation.py \
  --dataset_name "${dataset_path}" \
  --model_name_or_path "${model_path}" \
  --output_path "${output_path}" \
  --torch_dtype bfloat16 \
  --num_return_sequences 5 \
  --temperature 0.9 \
  --max_new_tokens 512 \
  --batch_size 4 \
  --given_leader_completions="${given_leader_completions}"
#  --follower_prompt "Improve the previous answer. Phrase it as if it was the original response."

# Uncomment the last line if you want to use the follower prompt and iteratively generate responses

echo "Evaluating the model ${model_path}"
python src/evaluation/reward_evaluation.py \
  --reward_model_name_or_path unsloth/Qwen2.5-1.5B-Instruct \
  --reward_model_adapters_path "${reward_adapters_path}" \
  --dataset_path "${output_path}" \
  --batch_size 1
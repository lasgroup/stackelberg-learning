#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
export WANDB_API_KEY=$(cat "${HOME}/.wandb-api-key")
export WANDB_PROJECT="nash_md_training_wandb_project_name"
run_name="nash_md_model_training"

python src/finetuning/nash_md.py \
  --model_name_or_path unsloth/Qwen2.5-0.5B-Instruct \
  --reward_model_path unsloth/Qwen2.5-1.5B-Instruct \
  --reward_model_adapters_path "path/to/reward/model/adapters" \
  --dataset_name "path/to/preprocessed_helpsteer2_dataset" \
  --dataset_test_split validation \
  --max_prompt_length 1024 \
  --output_dir "data/experiments/${run_name}" \
  --run_name "${run_name}" \
  --mixture_coef 0.75 \
  --beta 0.001 \
  --missing_eos_penalty 1.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 512 \
  --max_steps 2000 \
  --learning_rate 1.0e-5 \
  --lr_scheduler_type constant \
  --logging_steps 5 \
  --eval_steps 50 \
  --save_steps 250 \
  --per_device_eval_batch_size 16 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --bf16_full_eval True \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.1 \
  --lora_task_type CAUSAL_LM

#!/bin/bash
export WANDB_API_KEY=$(cat "${HOME}/.wandb-api-key")
export WANDB_PROJECT="reward_model_wandb_project_name"
export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "Running for attribute: $1"
python src/reward_modeling/reward_model_training.py \
    --model_name_or_path unsloth/Qwen2.5-1.5B-Instruct \
    --dataset_name "path/to/preprocessed_helpsteer2_dataset" \
    --dataset_test_split validation \
    --label_name "$1" \
    --output_dir "path/to/reward_model_directory" \
    --run_name "wandb_run_name" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 5 \
    --learning_rate 1.0e-4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 50 \
    --per_device_eval_batch_size 32 \
    --save_strategy steps \
    --save_steps 50 \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --bf16_full_eval True \
    --gradient_checkpointing True \
    --center_rewards_coefficient 0.01 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_task_type SEQ_CLS
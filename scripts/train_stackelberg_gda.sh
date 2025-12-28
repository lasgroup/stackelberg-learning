#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
export WANDB_API_KEY=$(cat "${HOME}/.wandb-api-key")
export WANDB_PROJECT="stackelberg_gda_training_wandb_project_name"
run_name="stackelberg_gda_model_training"

python src/finetuning/stackelberg_gda.py \
  --seed 42 \
  --model_name_or_path unsloth/Qwen2.5-0.5B-Instruct \
  --reward_model_path unsloth/Qwen2.5-1.5B-Instruct \
  --reward_model_adapters_path "path/to/reward/model/adapters" \
  --dataset_name "path/to/preprocessed_helpsteer2_dataset" \
  --dataset_test_split validation \
  --max_prompt_length 1024 \
  --output_dir "data/experiments/${run_name}" \
  --run_name ${run_name} \
  --follower_weight 5.0 \
  --leader_update_frequency 1 \
  --beta 0.001 \
  --standard_follower_kl_regularization true \
  --top_k 0 \
  --top_p 1.0 \
  --generation_temperature 0.9 \
  --kl_estimator k3 \
  --missing_eos_penalty 1.0 \
  --missing_eos_probability_penalty 0.0 \
  --score_baseline 0.5 \
  --rloo_baseline false \
  --max_clip_grad_norm 200.0 \
  --separate_follower_model false \
  --save_safetensors true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 512 \
  --max_steps 2000 \
  --learning_rate 1.0e-5 \
  --lr_scheduler_type constant \
  --adam_epsilon 1.0e-6 \
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
#!/bin/bash

JUDGE_NAME_OR_PATH="pasztorb/Skywork-Critic-Llama-3.1-70B-4bit-bnb"
BASE_MODEL="allenai/Llama-3.1-Tulu-3-8B-SFT"
run_name="experiment_v1"
per_device_train_batch_size=2
gradient_accumulation_steps=1

python src/finetuning/stackelberg_gda.py \
      --seed 42 \
      --judge ${JUDGE_NAME_OR_PATH} \
      --model_name_or_path ${BASE_MODEL} \
      --dataset_name "allenai/llama-3.1-tulu-3-8b-preference-mixture" \
      --dataset_test_split "validation" \
      --max_prompt_length 1024 \
      --output_dir "data/experiments/${run_name}" \
      --run_name ${run_name} \
      --follower_weight 2.0 \
      --leader_update_frequency 1 \
      --beta 0.001 \
      --standard_follower_kl_regularization true \
      --top_k 0 \
      --top_p 1.0 \
      --generation_temperature 0.9 \
      --kl_estimator k1 \
      --score_baseline 0.5 \
      --rloo_baseline false \
      --max_clip_grad_norm 100.0 \
      --separate_follower_model false \
      --save_safetensors true \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --gradient_accumulation_steps ${gradient_accumulation_steps} \
      --max_new_tokens 1024 \
      --num_train_epochs 1 \
      --learning_rate "5.0e-6" \
      --lr_scheduler_type linear \
      --warmup_ratio 0.1 \
      --adam_epsilon 1.0e-6 \
      --logging_steps 5 \
      --save_steps 100 \
      --torch_dtype bfloat16 \
      --bf16 True
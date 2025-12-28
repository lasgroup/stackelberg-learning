import torch
import wandb
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import (
    RLOOConfig,
    get_peft_config,
    ModelConfig,
)
from src.trainer.rloo_trainer_modified import RLOOTrainerModified
from src.utils import load_dataset, load_adapter_model
from src.reward_modeling.preprocessing_arguments import PreprocessingArguments
from trl import ScriptArguments
from typing import Optional, List
from peft import get_peft_model
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


@dataclass
class TrainingConfig:
    disable_dropout: bool = False
    gradient_checkpointing: bool = True


# Modified PPOConfig to allow for several reward model adapters
@dataclass
class RLOOConfigModified(RLOOConfig):
    reward_model_adapters_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Glob or regex for reward model adapter paths, e.g. path/to/adapters/*/checkpoint-*"
        },
    )
    reward_sampling_strategy: str = field(
        default="fixed",
        metadata={
            "help": "Reward adapter sampling strategy: 'fixed', 'random', or 'round_robin'"
        },
    )
    default_reward_adapter: Optional[str] = field(
        default="verbosity", metadata={"help": "Used when sampling strategy is 'fixed'"}
    )
    max_new_tokens: int = field(
        default=1024, metadata={"help": "Max tokens to generate during training"}
    )
    kl_estimator: str = field(
        default="k1", metadata={"help": "KL divergence estimator to use"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ScriptArguments, PreprocessingArguments, RLOOConfigModified, ModelConfig)
    )
    script_args, preprocessing_args, training_args, model_args = (
        parser.parse_args_into_dataclasses()
    )

    # Set default dtype
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    torch.set_default_dtype(torch_dtype)

    # Prepare model kwargs
    model_kwargs = {
        "revision": model_args.model_revision,
        "device_map": None,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "quantization_config": None,
    }

    adapter_paths_prefix = training_args.reward_model_adapters_path.split("(")[0]
    adapter_paths_suffix = training_args.reward_model_adapters_path.split(")")[-1]
    reward_labels = [
        "coherence",
        "complexity",
        "correctness",
        "helpfulness",
        "verbosity",
    ]

    adapter_paths_dict = {
        label: f"{adapter_paths_prefix}{label}{adapter_paths_suffix}"
        for label in reward_labels
    }

    # Load reward model with adapters
    reward_tokenizer, reward_model = load_adapter_model(
        training_args.reward_model_path,
        adapter_paths_dict,
        model_args,
        model_kwargs,
    )
    reward_model.eval().requires_grad_(False)

    # Load policy and reference models
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    if model_args.use_peft:
        model_args.peft_target_modules = (
            TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                ref_model.config.model_type
            ]
        )
        peft_config = get_peft_config(model_args)
        model = get_peft_model(ref_model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
    # model.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)

    assert set(adapter_paths_dict.keys()) == set(
        ["coherence", "complexity", "correctness", "helpfulness", "verbosity"]
    )

    print("Loaded adapters:", reward_model.peft_config.keys())

    print("======== EOS Token Diagnostics ========")
    print("EOS token:", tokenizer.eos_token)
    print("EOS token ID:", tokenizer.eos_token_id)
    print(
        "Decoded EOS token:",
        (
            tokenizer.decode([tokenizer.eos_token_id])
            if tokenizer.eos_token_id is not None
            else "None"
        ),
    )
    print("=======================================")
    assert tokenizer.eos_token_id is not None, "Missing eos_token_id"
    assert tokenizer.pad_token_id is not None, "Missing pad_token_id"

    dataset_preprocessed = load_dataset(script_args, prompts_only=True)

    # Filter out long prompts to avoid OOMs
    def prepare_dataset(dataset):
        # Filter out long inputs to avoid OOM errors
        dataset = dataset.filter(
            lambda x: len(tokenizer.apply_chat_template(x["prompt"]))
            <= preprocessing_args.max_prompt_length
        )
        return dataset

    train_dataset = prepare_dataset(dataset_preprocessed["train"])
    eval_dataset = prepare_dataset(dataset_preprocessed[script_args.dataset_test_split])

    def format_prompt(example):
        prompt_ids = tokenizer.apply_chat_template(
            example["prompt"],  # Already a list of messages
            tokenize=True,
            add_generation_prompt=True,  # Use False if you're scoring completions
            return_tensors=None,
        )
        return {"input_ids": prompt_ids}

    train_dataset = dataset_preprocessed["train"].map(
        format_prompt,
        remove_columns=["prompt", "prompt_id"],
    )

    eval_dataset = dataset_preprocessed["validation"].map(
        format_prompt,
        remove_columns=["prompt", "prompt_id"],
    )

    # Initialize trainer
    trainer = RLOOTrainerModified(
        config=training_args,
        processing_class=tokenizer,
        policy=model,
        ref_policy=ref_model,
        reward_model=reward_model,
        tokenizer=reward_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("==== Starting RLOO Training Loop ... ====")

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    wandb.finish()

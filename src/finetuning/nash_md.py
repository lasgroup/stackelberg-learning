import os
import torch
import wandb
from trl import (
    ModelConfig,
    ScriptArguments,
    get_peft_config,
    get_quantization_config,
    NashMDConfig,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)

from src.reward_modeling.preprocessing_arguments import PreprocessingArguments
from src.utils import load_dataset, load_adapter_model
from src.trainer.nash_md_modified import NashMDTrainerModified
from src.judges.MultiAdapterJudge import MultiAdapterJudge
from accelerate import Accelerator
from dataclasses import dataclass, field


@dataclass
class NashMDConfigModified(NashMDConfig):
    reward_model_adapters_path: str = field(
        default=None,
        metadata={
            "help": "regex expression for the reward_model_adapters_paths",
        },
    )


if __name__ == "__main__":
    accelerator = Accelerator()
    accelerator.print("Accelerator num_processes: ", accelerator.num_processes)
    accelerator.print(
        "Accelerator local_process_index: ", accelerator.local_process_index
    )
    accelerator.print("Accelerator precision: ", accelerator.mixed_precision)
    accelerator.print("Default float dtype: ", torch.get_default_dtype())

    parser = HfArgumentParser(
        (ScriptArguments, PreprocessingArguments, NashMDConfigModified, ModelConfig)
    )
    script_args, preprocessing_args, training_args, model_args = (
        parser.parse_args_into_dataclasses()
    )
    accelerator.print("--- ARGUMENTS ---")
    accelerator.print(script_args)
    accelerator.print(preprocessing_args)
    accelerator.print(training_args)
    accelerator.print(model_args)

    # GPU cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    torch.set_default_dtype(torch_dtype)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )

    # Load data as DatasetDict
    # Schema: "train" and "validation" splits each with columns ['prompt', 'prompt_id']
    dataset_preprocessed = load_dataset(script_args, prompts_only=True)

    # Load preference oracle model
    if training_args.reward_model_adapters_path is None:
        accelerator.print("Loading reward model without adapters!")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            num_labels=1,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        if hasattr(reward_model, "score"):
            reward_model.score.original_module.weight.data.zero_()
            if reward_model.score.original_module.bias is not None:
                reward_model.score.original_module.bias.data.zero_()
        else:
            raise NotImplementedError
        reward_model.eval()
        judge = None
    else:
        accelerator.print(
            "Loading reward model with adapters and constructing a judge!"
        )
        reward_model = None
        adapter_paths_prefix = training_args.reward_model_adapters_path.split("(")[0]
        adapter_paths_suffix = training_args.reward_model_adapters_path.split(")")[-1]
        adapter_paths = {
            x: adapter_paths_prefix + x + adapter_paths_suffix
            for x in training_args.reward_model_adapters_path.split("(")[1]
            .split(")")[0]
            .split("|")
        }
        reward_model_tokenizer, multi_adapter_reward_model = load_adapter_model(
            training_args.reward_model_path, adapter_paths, model_args, model_kwargs
        )

        # Setup Judge
        judge = MultiAdapterJudge(
            model=multi_adapter_reward_model,
            tokenizer=reward_model_tokenizer,
            missing_eos_penalty=training_args.missing_eos_penalty,
        )
        training_args.missing_eos_penalty = (
            None  # Have to overwrite otherwise the trainer throws an error
        )

    # Setup model to train
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Datasets
    def prepare_nashmd_dataset(dataset):
        # Filter out long inputs to avoid OOM errors
        dataset = dataset.filter(
            lambda x: len(tokenizer.apply_chat_template(x["prompt"]))
            <= preprocessing_args.max_prompt_length
        )
        return dataset

    train_dataset = prepare_nashmd_dataset(dataset_preprocessed["train"])
    eval_dataset = prepare_nashmd_dataset(
        dataset_preprocessed[script_args.dataset_test_split]
    )
    accelerator.print("Train dataset size: ", len(train_dataset))
    accelerator.print("Eval dataset size: ", len(eval_dataset))
    accelerator.print("First entry of the training dataset: ", train_dataset[0])
    accelerator.print("First entry of the eval dataset: ", eval_dataset[0])

    # Setup training
    trainer = NashMDTrainerModified(
        model=model,
        # ref_model=ref_model,  # ref_model is needed when PEFT config is used, otherwise NashMDTrainer breaks
        ref_model=None,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )
    if judge is not None:
        judge.add_accelerator(
            trainer.accelerator
        )  # Workaround for consistent accelerator usage
    accelerator.print("Trainer created!")
    trainer.train(
        resume_from_checkpoint=(
            True if os.getenv("WANDB_RESUME") in ["allow", "must"] else None
        )
    )
    accelerator.print("Training completed!")
    trainer.accelerator.wait_for_everyone()
    wandb.finish()

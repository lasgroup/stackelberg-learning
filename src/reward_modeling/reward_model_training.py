import datasets
import torch
import os
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    RewardConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from accelerate import PartialState, Accelerator

from src.preprocessing.tldr import load_dataset as tldr_load_dataset
from src.preprocessing.tldr import prepare_dataset as tldr_prepare_dataset
from src.preprocessing.helpsteer2 import prepare_dataset as helpsteer2_prepare_dataset
from src.reward_modeling.data_collator_with_padding import (
    RewardDataCollatorWithPaddingModified,
)
from src.reward_modeling.preprocessing_arguments import PreprocessingArguments
from src.reward_modeling.reward_trainer_weighted_loss import RewardTrainerWeightedLoss

if __name__ == "__main__":
    accelerator = Accelerator()
    with PartialState().local_main_process_first():
        accelerator.print("Accelerator num_processes: ", accelerator.num_processes)
        accelerator.print(
            "Accelerator local_process_index: ", accelerator.local_process_index
        )
        accelerator.print("Accelerator device: ", accelerator.device)
        accelerator.print("Accelerator precision: ", accelerator.mixed_precision)
        accelerator.print("Default float dtype: ", torch.get_default_dtype())

    parser = HfArgumentParser(
        (ScriptArguments, PreprocessingArguments, RewardConfig, ModelConfig)
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
        device_map=(get_kbit_device_map() if quantization_config is not None else None),
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    accelerator.print("Model kwargs:", model_kwargs)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    accelerator.print("Tokenizer loaded!")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    # Set score head's weight and bias to zero to have consistency between adapter trained
    if hasattr(model, "score"):
        model.score.weight.data.zero_()
        if model.score.bias is not None:
            model.score.bias.data.zero_()
    else:
        raise NotImplementedError
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()
    accelerator.print("Model loaded!")

    # Load dataset
    with PartialState().local_main_process_first():
        if script_args.dataset_name == "openai/summarize_from_feedback":
            dataset, _ = tldr_load_dataset(
                script_args.dataset_name,
                script_args.dataset_config,
                preprocessing_args.min_annotation_per_worker,
                worker_id=preprocessing_args.worker_id,
            )

            dataset_preprocessed = tldr_prepare_dataset(
                dataset,
                tokenizer,
                max_prompt_length=preprocessing_args.max_prompt_length,
                max_response_length=preprocessing_args.max_response_length,
                seed=training_args.seed,
            )
            training_args.output_dir = os.path.join(
                training_args.output_dir,
                model_args.model_name_or_path.replace("/", "_") + "__tldr",
            )
            if preprocessing_args.worker_id:
                dataset_preprocessed["train"] = dataset_preprocessed["train"].map(
                    lambda x: {
                        "loss_weight": (
                            preprocessing_args.loss_weight
                            if x["worker"] == preprocessing_args.worker_id
                            else 1.0
                        )
                    }
                )
                # If worker_id is given, overwrite training_args.run_name/output_dir to include worker_id
                training_args.run_name = (
                    f"{training_args.run_name}_{preprocessing_args.worker_id}"
                )
                training_args.output_dir = (
                    f"{training_args.output_dir}__{preprocessing_args.worker_id}"
                )
            else:
                training_args.run_name = f"{training_args.run_name}__all_workers"
                training_args.output_dir = f"{training_args.output_dir}__all_workers"
        elif (
            os.path.exists(script_args.dataset_name)
            and "HelpSteer2" in script_args.dataset_name
        ):
            dataset = datasets.load_from_disk(script_args.dataset_name)
            accelerator.print(f"Dataset loaded from path: {script_args.dataset_name}")
            dataset_preprocessed = helpsteer2_prepare_dataset(
                dataset,
                label_name=preprocessing_args.label_name,
            )
            accelerator.print(dataset_preprocessed)
            training_args.run_name = (
                f"{training_args.run_name}_{preprocessing_args.label_name}"
            )
            training_args.output_dir = os.path.join(
                training_args.output_dir,
                (
                    model_args.model_name_or_path.replace("/", "_")
                    + f"__helpsteer2__{preprocessing_args.label_name}"
                ),
            )
            accelerator.print(f"Run name: {training_args.run_name}")
            accelerator.print(f"Output dir: {training_args.output_dir}")
        elif script_args.dataset_name == "nvidia/HelpSteer2":
            raise ValueError(
                "Dataset has to be split first for training and validation sets before it can be used for training"
            )
        else:
            raise ValueError("Dataset not supported")
        accelerator.print("Number of datapoints after preprocessing")
        for dataset_name, data in dataset_preprocessed.items():
            accelerator.print(dataset_name, len(data))
        accelerator.print("Example input: ", dataset_preprocessed["train"][0])

    # Run the training
    trainer = RewardTrainerWeightedLoss(
        model=model,
        processing_class=tokenizer,
        data_collator=RewardDataCollatorWithPaddingModified(tokenizer),
        args=training_args,
        train_dataset=dataset_preprocessed[script_args.dataset_train_split],
        eval_dataset=(
            dataset_preprocessed[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    # Final evaluation
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save the model
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    wandb.finish()

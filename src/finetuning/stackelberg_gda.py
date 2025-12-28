import torch
import wandb
import os
from accelerate import Accelerator
from trl import (
    ModelConfig,
    ScriptArguments,
    get_peft_config,
    get_quantization_config,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)
from peft import get_peft_model
from trl.trainer import empty_cache

from src.reward_modeling.preprocessing_arguments import PreprocessingArguments
from src.utils import load_dataset, load_adapter_model
from src.judges.MultiAdapterJudge import MultiAdapterJudge
from src.judges.OpenAIPairwiseJudge import OpenAIPairwiseJudge, DEFAULT_PAIRWISE_SYSTEM_PROMPT, SKYWORKS_SYSTEM_PROMPT
from src.trainer.stackelberg_gda_trainer import StackelbergGDATrainer
from src.trainer.DualModel import DualModel
from src.trainer.stackelberg_gda_config import StackelbergGDAConfig


def setup_judge(model_args, training_args, accelerator):
    accelerator.print("Setting up judge model...")
    if training_args.judge is not None:
        if not training_args.judge.startswith("localhost"):
            raise NotImplementedError()
        accelerator.print("Using OpenAIPairwiseJudge with localhost judge!")
        if "-" not in training_args.judge:
            api_base="http://127.0.0.1:8000/v1"
        else:
            api_base=f"http://{training_args.judge.split('-')[1]}/v1"
        judge = OpenAIPairwiseJudge(
            openai_api_base=api_base,
            temperature=training_args.judge_temperature,
            # system_prompt=SKYWORKS_SYSTEM_PROMPT,
        )
        reward_model = None
    elif training_args.reward_model_adapters_path is None:
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
    return reward_model, judge


if __name__ == "__main__":
    accelerator = Accelerator()
    accelerator.print("Accelerator num_processes: ", accelerator.num_processes)
    accelerator.print(
        "Accelerator local_process_index: ", accelerator.local_process_index
    )
    accelerator.print("Accelerator precision: ", accelerator.mixed_precision)

    parser = HfArgumentParser(
        (ScriptArguments, PreprocessingArguments, StackelbergGDAConfig, ModelConfig)
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
    accelerator.print("Default float dtype: ", torch.get_default_dtype())
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    quantization_config = get_quantization_config(model_args)
    if quantization_config is None and hasattr(model_config, "quantization_config"):
        quantization_config = model_config.quantization_config
        print("Using default quantization config: ", quantization_config)
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
    reward_model, judge = setup_judge(model_args, training_args, accelerator)

    # Setup model to train
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if training_args.separate_follower_model:
        accelerator.print("Using separate adapter for the follower model!")
        assert (
            ~training_args.save_safetensors
        ), "Cannot save DualModel as safetensors! set --save_safetensors=False"

        leader_base = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        leader_base.config.pad_token_id = tokenizer.pad_token_id
        follower_base = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        follower_base.config.pad_token_id = tokenizer.pad_token_id
        leader_model = get_peft_model(
            leader_base, get_peft_config(model_args), adapter_name="leader"
        )
        follower_model = get_peft_model(
            follower_base, get_peft_config(model_args), adapter_name="follower"
        )
        model = DualModel(leader_model, follower_model)

        leader_params = []
        follower_params = []
        for n, p in model.named_parameters():
            if ("leader" in n) and ("lora" in n) and p.requires_grad:
                leader_params.append(p)
            if ("follower" in n) and ("lora" in n) and p.requires_grad:
                follower_params.append(p)
        assert len(leader_params) == len(follower_params)
        accelerator.print("Number of parameters in leader_params: ", len(leader_params))
        accelerator.print(
            "Number of parameters in follower_params: ", len(follower_params)
        )
        grouped_params = [
            {
                "params": leader_params,
                "lr": training_args.learning_rate,
            },
            {
                "params": follower_params,
                "lr": training_args.follower_weight * training_args.learning_rate,
            },
        ]

        if training_args.optim.lower() == "adamw_torch":
            optimizer = torch.optim.AdamW(
                grouped_params,
                betas=(training_args.adam_beta1, training_args.adam_beta1),
                eps=training_args.adam_epsilon,
            )
            accelerator.print("Parameter group optimizer created: ", optimizer)
        else:
            raise NotImplementedError
        peft_config = None
    else:
        accelerator.print("Using a shared model for both Leader and Follower")
        optimizer = None
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        peft_config = get_peft_config(model_args)
        accelerator.print("Peft config: ", peft_config)
    model.train()

    # Datasets
    def prepare_dataset(dataset):
        # Filter out long inputs to avoid OOM errors
        initial_dataset_size = len(dataset)
        dataset = dataset.filter(
            lambda x: len(tokenizer.apply_chat_template(x["prompt"]))
            <= preprocessing_args.max_prompt_length
        )
        removed_samples = initial_dataset_size - len(dataset)
        accelerator.print(
            "Filtered out {} samples ({:.2f}%) that were too long.".format(
                removed_samples,
                100 * removed_samples / initial_dataset_size,
            )
        )
        return dataset

    train_dataset = prepare_dataset(dataset_preprocessed["train"])
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    accelerator.print("Train dataset size: ", len(train_dataset))
    accelerator.print("First entry of the training dataset: ", train_dataset[0])
    if (
        script_args.dataset_test_split
        and script_args.dataset_test_split in dataset_preprocessed
    ):
        eval_dataset = prepare_dataset(
            dataset_preprocessed[script_args.dataset_test_split]
        )
        accelerator.print("Eval dataset size: ", len(eval_dataset))
        accelerator.print("First entry of the eval dataset: ", eval_dataset[0])
    else:
        accelerator.print(
            "No test split defined or the dataset does not contain a test split."
        )
        eval_dataset = None

    # Setup training
    accelerator.print("Save Safetensors: ", training_args.save_safetensors)
    trainer = StackelbergGDATrainer(
        model=model,
        ref_model=None,
        reward_model=reward_model,
        judge=judge,
        optimizers=(optimizer, None),
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    if (judge is not None) & (hasattr(judge, "add_accelerator")):
        judge.add_accelerator(
            trainer.accelerator
        )  # Workaround for consistent accelerator usage
        empty_cache()
    accelerator.print("Trainer created!")
    trainer.train(
        resume_from_checkpoint=(
            True if os.getenv("WANDB_RESUME") in ["allow", "must"] else None
        )
    )
    accelerator.print("Training completed!")
    trainer.accelerator.wait_for_everyone()
    wandb.finish()

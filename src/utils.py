import datasets
import os
from datasets import load_from_disk

from peft import PeftModel, PeftModelForSequenceClassification
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
)

from src.preprocessing.utils import chat_structure
from src.preprocessing.helpsteer2 import (
    prepare_dataset as helpsteer2_prepare_dataset,
)
from src.preprocessing.tldr import (
    load_dataset as tldr_load_dataset,
    prepare_dataset as tldr_prepare_dataset,
)

from typing import Union, Dict

def load_dataset(
    script_args,
    preprocessing_args=None,
    seed=None,
    tokenizer=None,
    prompts_only=False,
):
    if script_args.dataset_name == "openai/summarize_from_feedback":
        dataset, _ = tldr_load_dataset(
            script_args.dataset_name,
            script_args.dataset_config,
            preprocessing_args.min_annotation_per_worker,
            worker_id=preprocessing_args.worker_id,
        )
        if prompts_only:
            raise NotImplementedError
        else:
            dataset_preprocessed = tldr_prepare_dataset(
                dataset,
                tokenizer,
                max_prompt_length=preprocessing_args.max_prompt_length,
                max_response_length=preprocessing_args.max_response_length,
                seed=seed,
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
    elif (
        os.path.exists(script_args.dataset_name)
        and "helpsteer2" in script_args.dataset_name.lower()
    ):
        dataset = load_from_disk(script_args.dataset_name)
        if prompts_only:
            dataset_preprocessed = dataset.map(
                chat_structure,
                remove_columns=["labels", "completions"],
            )
        else:
            dataset_preprocessed = helpsteer2_prepare_dataset(
                dataset,
                label_name=preprocessing_args.label_name,
            )
    elif script_args.dataset_name == "allenai/llama-3.1-tulu-3-8b-preference-mixture":
        dataset = datasets.load_dataset(script_args.dataset_name)
        if prompts_only:
            dataset_preprocessed = dataset.select_columns(
                ["id", "prompt"]
            ).rename_column("id", "prompt_id")
            dataset_preprocessed = dataset_preprocessed.map(
                lambda x: {"prompt": [{"role": "user", "content": x["prompt"]}]},
            )
        else:
            dataset_preprocessed = dataset.select_columns(
                ["id", "chosen", "rejected"]
            ).rename_column("id", "prompt_id")
    elif script_args.dataset_name == "tatsu-lab/alpaca_eval":
        dataset_preprocessed = datasets.load_dataset(
            script_args.dataset_name, "alpaca_eval", trust_remote_code=True
        )["eval"]
        dataset_preprocessed = dataset_preprocessed.rename_columns(
            {"instruction": "prompt"}
        ).remove_columns(["output", "generator"])
        dataset_preprocessed = dataset_preprocessed.add_column(
            "prompt_id", range(len(dataset_preprocessed))
        )
        dataset_preprocessed = dataset_preprocessed.map(
            lambda x: {"prompt": [{"role": "user", "content": x["prompt"]}]},
        )
        dataset_preprocessed = datasets.DatasetDict(
            {"validation": dataset_preprocessed}
        )
    elif script_args.dataset_name in ["google/IFEval", "valpy/ifeval_ood3"]:
        dataset_preprocessed = datasets.load_dataset(
            script_args.dataset_name, trust_remote_code=True
        )["train"]
        dataset_preprocessed = dataset_preprocessed.add_column(
            "prompt_id", range(len(dataset_preprocessed))
        )
        dataset_preprocessed = dataset_preprocessed.map(
            lambda x: {"prompt": [{"role": "user", "content": x["prompt"]}]},
        )
        if prompts_only:
            dataset_preprocessed = dataset_preprocessed.select_columns(
                ["prompt_id", "prompt"]
            )
        dataset_preprocessed = datasets.DatasetDict(
            {"validation": dataset_preprocessed}
        )
    elif script_args.dataset_name == "nvidia/HelpSteer2":
        raise ValueError(
            "Dataset has to be split first for training and validation sets before it can be used for training"
        )
    else:
        raise ValueError(f"Dataset not supported at path: {script_args.dataset_name}")
    print("Number of datapoints after preprocessing")
    for dataset_name, data in dataset_preprocessed.items():
        print(dataset_name, len(data))
    print("Example input: ", next(iter(dataset_preprocessed.values()))[0])
    return dataset_preprocessed


def load_adapter_model(
    base_model_name_or_path: str,
    adapter_names_or_paths: Dict[str, str],
    model_args,
    model_kwargs: dict = {},
) -> Union[PreTrainedTokenizer, PeftModelForSequenceClassification]:
    """
    Loads the base model and the corresponding adapters.
    :param base_model_name_or_path:
    :param adapter_names_or_paths:
    :param model_args:
    :param model_kwargs:
    :return:
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        torch_dtype=model_args.torch_dtype,
    )

    # Load the base model once
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name_or_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # Set score head's weight and bias to zero to have consistency between adapter trained
    if hasattr(base_model, "score"):
        base_model.score.weight.data.zero_()
        if base_model.score.bias is not None:
            base_model.score.bias.data.zero_()
    else:
        raise NotImplementedError

    # Load adapters one by one, but with weight sharing
    first_name = list(adapter_names_or_paths.keys())[0]
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_names_or_paths[first_name],
        local_files_only=True,
        adapter_name=first_name,
        inference_mode=True,
        **model_kwargs,
    )

    # Add other adapters
    for i, (adapter_name, path) in enumerate(adapter_names_or_paths.items(), 1):
        if adapter_name == first_name:
            continue
        peft_model.load_adapter(path, adapter_name=adapter_name)

    print(f"Successfully loaded {len(adapter_names_or_paths)} adapter models")
    return tokenizer, peft_model

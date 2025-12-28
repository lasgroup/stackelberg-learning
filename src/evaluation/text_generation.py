import torch
import tqdm
from datasets import DatasetDict, Dataset, load_from_disk
from dataclasses import dataclass, field
from accelerate import Accelerator
from accelerate.utils import gather_object
from trl import (
    ModelConfig,
    ScriptArguments,
    get_quantization_config,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.models.utils import unwrap_model_for_generation
from peft import PeftConfig, PeftModel

from transformers import (
    Qwen2ForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    GenerationConfig,
    AutoConfig,
)
from safetensors import safe_open
from safetensors.torch import load_file

from typing import List, Union

import os
import sys

# Add project root to sys.path to avoid import errors
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to sys.path")

from src.utils import load_dataset
from src.reward_modeling.preprocessing_arguments import PreprocessingArguments

"""
Example script:
python src/evaluation/text_generation.py \
    --dataset_name data/datasets/HelpSteer2/preprocessed_dataset \
    --model_name_or_path unsloth/Qwen2.5-0.5B-Instruct \
    --output_path data/experiments/nash_md/generation_Qwen2.5-0.5B-Instruct \
    --torch_dtype bfloat16 \
    --num_return_sequences 10 \
    --max_new_tokens 1024
"""


@dataclass
class GenerationArguments:
    num_samples: int = field(
        default=None,
        metadata={
            "help": "Number of prompts to generate for. If None, all prompts will be used."
        },
    )

    max_new_tokens: int = field(
        default=None,
        metadata={
            "help": "The maximum number of new tokens to generate. If None, the model will generate until it reaches the end of the sequence."
        },
    )

    num_return_sequences: int = field(
        default=1,
        metadata={
            "help": "The maximum number of new tokens to generate. If None, the model will generate until it reaches the end of the sequence."
        },
    )

    batch_size: int = field(
        default=1,
        metadata={"help": ""},
    )

    temperature: float = field(
        default=1.0,
        metadata={
            "help": "The temperature to use for sampling. Higher values mean more random samples."
        },
    )

    top_k: int = field(
        default=50,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."
        },
    )

    top_p: float = field(
        default=1.0,
        metadata={
            "help": "The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling."
        },
    )

    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling or not."},
    )

    output_path: str = field(
        default=None,
        metadata={
            "help": "",
        },
    )

    follower_prompt: str = field(
        # default="Improve the previous answer. Phrase it as if it was the original response.",
        default=None,
        metadata={
            "help": "The prompt to use for the follower model to generate completions based on the leader's completion."
        },
    )

    given_leader_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to use pregenerated leader completions or not. If True, the script will use the completions from the dataset."
        },
    )


def generate_text(
    model,
    tokenizer,
    accelerator,
    batch,
    generation_config,
) -> Union[List[str], List[List[str]]]:
    """
    Generate text using the model and tokenizer.
    :return: List[str] or List[List[str]]
    """
    is_conversational_batch = is_conversational(batch[0])
    # Preprocess the batch
    prompts = [maybe_apply_chat_template(x, tokenizer)["prompt"] for x in batch]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")

    with torch.no_grad(), unwrap_model_for_generation(
        model, accelerator
    ) as unwrapped_model:
        generated_ids = unwrapped_model.generate(
            input_ids=encoded["input_ids"].to(model.device),
            attention_mask=encoded["attention_mask"].to(model.device),
            generation_config=generation_config,
        )
    context_length = encoded["input_ids"].shape[1]
    generated_ids = generated_ids[:, context_length:]

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if is_conversational_batch:
        generated_texts = [
            [{"role": "assistant", "content": text}] for text in generated_texts
        ]

    # Create list of lists based on how many generations were created
    if generation_config.num_return_sequences > 1:
        generated_texts = [
            generated_texts[i : i + generation_config.num_return_sequences]
            for i in range(
                0,
                len(generated_texts),
                generation_config.num_return_sequences,
            )
        ]

    del generated_ids, context_length, encoded
    torch.cuda.empty_cache()
    return generated_texts


def follower_generate_text(
    model,
    tokenizer,
    accelerator,
    batch,
    generation_config,
    follower_prompt,
    num_return_sequences,
) -> Union[List[str], List[List[str]]]:
    """
    Generate responses iteratively. First using the Leader and then using the Follower's correction
    """
    assert (
        generation_config.num_return_sequences == 1
    ), "Generation config must have num_return_sequences=1 if Follower correction generation is used."
    assert (
        num_return_sequences > 1
    ), "Number of return sequences must be larger than 1. Otherwise Follower correction generation is meaningless."

    if "leader_completion" in batch[0]:
        completions = [[x["leader_completion"] for x in batch]]
    else:
        completions = [
            generate_text(
                model,
                tokenizer,
                accelerator,
                batch,
                generation_config,
            )
        ]

    if is_conversational(batch[0]):
        follower_prompt_maybe_conversational = [
            {"role": "user", "content": follower_prompt}
        ]
    else:
        follower_prompt_maybe_conversational = follower_prompt

    for _ in range(num_return_sequences - 1):
        follower_input_batch = [
            {"prompt": b["prompt"] + c + follower_prompt_maybe_conversational}
            for b, c in zip(batch, completions[-1])
        ]
        completions.append(
            generate_text(
                model,
                tokenizer,
                accelerator,
                follower_input_batch,
                generation_config,
            )
        )
    # Transpose completions
    completions = list(map(list, zip(*completions)))
    return completions


def generate_dataset_completions(
    accelerator,
    dataset,
    model,
    tokenizer,
    generation_args: GenerationArguments,
):
    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=generation_args.max_new_tokens,
        temperature=generation_args.temperature,
        top_p=generation_args.top_p,
        top_k=generation_args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=(
            generation_args.num_return_sequences
            if generation_args.follower_prompt is None
            else 1
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=generation_args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,
    )
    dataloader = accelerator.prepare(dataloader)
    completions = []
    for batch in tqdm.tqdm(dataloader, desc=f"Generating completions"):
        if generation_args.follower_prompt is None:
            generated_texts = generate_text(
                model, tokenizer, accelerator, batch, generation_config
            )
        else:
            generated_texts = follower_generate_text(
                model,
                tokenizer,
                accelerator,
                batch,
                generation_config,
                generation_args.follower_prompt,
                generation_args.num_return_sequences,
            )

        accelerator.wait_for_everyone()
        generated_texts = gather_object(generated_texts)
        if accelerator.is_main_process:
            completions.extend(generated_texts)
    accelerator.print("Number of generated completions: ", len(completions))
    return completions


def main():
    accelerator = Accelerator()
    parser = HfArgumentParser(
        (ScriptArguments, PreprocessingArguments, GenerationArguments, ModelConfig)
    )
    script_args, preprocessing_args, generation_args, model_args = (
        parser.parse_args_into_dataclasses()
    )
    accelerator.print("--- ARGUMENTS ---")
    accelerator.print(script_args)
    accelerator.print(preprocessing_args)
    accelerator.print(generation_args)
    accelerator.print(model_args)

    # GPU cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Setup the Model
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
        use_cache=True,
        torch_dtype=torch_dtype,
    )
    if os.path.exists(
        os.path.join(model_args.model_name_or_path, "adapter_config.json")
    ):
        accelerator.print("Loading model with PEFT adapter")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        model = PeftModel(
            base_model,
            peft_config,
        )

        # Load and correct naming if necessary (RLOO algorithm can save model weights weirdly)
        adapter_weights = load_file(
            os.path.join(model_args.model_name_or_path, "adapter_model.safetensors")
        )
        corrected_adapter_weights = {}
        for key, tensor in adapter_weights.items():
            corrected_name = key
            if key.startswith("module."):
                corrected_name = corrected_name[7:]  # Remove 'module.' prefix
            if "lora_A.weight" in key:
                corrected_name = corrected_name.replace(
                    "lora_A.weight", "lora_A.default.weight"
                )
            if "lora_B.weight" in key:
                corrected_name = corrected_name.replace(
                    "lora_B.weight", "lora_B.default.weight"
                )
            corrected_adapter_weights[corrected_name] = tensor
        del adapter_weights
        model.load_state_dict(corrected_adapter_weights, strict=False)
    elif os.path.exists(model_args.model_name_or_path):
        accelerator.print("Loading model without PEFT adapter")
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = Qwen2ForCausalLM(config)
        original_path = os.path.join(model_args.model_name_or_path, "model.safetensors")
        # Create new state dict with corrected keys
        new_state_dict = {}
        with safe_open(original_path, framework="pt", device="cpu") as f:
            # Get all tensor names in the file
            tensor_names = f.keys()
            # Process each tensor
            for key in tensor_names:
                tensor = f.get_tensor(key)
                # Remove 'module.' prefix if present
                if key.startswith("module."):
                    new_key = key[7:]  # Remove 'module.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = tensor
        model.load_state_dict(
            new_state_dict, strict=False
        )  # Example if you're just initializing
        accelerator.print("State dict loaded and updated")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left"
    )
    if tokenizer.pad_token is None:
        print("Missing padding token. Adding pad token as eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    model, tokenizer = accelerator.prepare(model, tokenizer)
    # Load data
    if generation_args.given_leader_completions:
        dataset_preprocessed = load_from_disk(
            script_args.dataset_name,
        )
        dataset_preprocessed = dataset_preprocessed.map(
            lambda x: {"leader_completion": x["completions"][0]},
            remove_columns=["completions"],
        )
    else:
        dataset_preprocessed = load_dataset(
            script_args,
            prompts_only=True,
        )  # Columns: "prompt", "prompt_id", "answer" (optional), "split" (optional)
    if generation_args.num_samples is not None:
        for dataset_name, dataset in dataset_preprocessed.items():
            dataset_preprocessed[dataset_name] = dataset.select(
                range(min(generation_args.num_samples, len(dataset)))
            )
        accelerator.print(
            "Generating completions for the first {} samples".format(
                generation_args.num_samples
            )
        )

    completions_dict = {}
    for data_split_name, data_split in dataset_preprocessed.items():
        accelerator.print(f"\n--- Processing {data_split_name} split ---")
        accelerator.print(
            f"Number of datapoints in {data_split_name}: {len(data_split)}"
        )
        split_completions = generate_dataset_completions(
            accelerator,
            data_split,
            model,
            tokenizer,
            generation_args,
        )
        if accelerator.is_main_process:
            num_datapoints = len(dataset_preprocessed[data_split_name])
            assert (
                len(split_completions) >= num_datapoints
            ), f"Not enough completions generated, num generated completions: {len(split_completions)}, num datapoints: {num_datapoints}"
            split_completions = split_completions[:num_datapoints]
            split_completions = Dataset.from_dict(
                {
                    "prompt_id": dataset_preprocessed[data_split_name]["prompt_id"],
                    "prompt": dataset_preprocessed[data_split_name]["prompt"],
                    "completions": split_completions,
                }
            )
            if "answer" in dataset_preprocessed[data_split_name].column_names:
                split_completions = split_completions.add_column(
                    "answer", dataset_preprocessed[data_split_name]["answer"]
                )
            if "split" in dataset_preprocessed[data_split_name].column_names:
                split_completions = split_completions.add_column(
                    "split", dataset_preprocessed[data_split_name]["split"]
                )
            completions_dict[data_split_name] = split_completions

    # Save Completions
    if accelerator.is_main_process:
        completions_dict = DatasetDict(completions_dict)
        completions_dict.save_to_disk(generation_args.output_path)


if __name__ == "__main__":
    main()

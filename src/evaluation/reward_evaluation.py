import torch
import argparse
from accelerate import Accelerator
from datasets import load_from_disk, Dataset, DatasetDict
import tqdm
from collections import defaultdict

from transformers import DataCollatorWithPadding
from trl.models.utils import unwrap_model_for_generation
from trl.data_utils import maybe_apply_chat_template
from trl import ModelConfig
import os
import sys

# Add project root to sys.path to avoid import errors
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added {project_root} to sys.path")
from src.utils import load_adapter_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_name_or_path", type=str, required=True)
    parser.add_argument("--reward_model_adapters_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Print args
    accelerator.print("Command-line Arguments: ", args)

    # GPU cleanup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Setup the Model
    adapter_paths_prefix = args.reward_model_adapters_path.split("(")[0]
    adapter_paths_suffix = args.reward_model_adapters_path.split(")")[-1]
    adapter_paths = {
        x: adapter_paths_prefix + x + adapter_paths_suffix
        for x in args.reward_model_adapters_path.split("(")[1].split(")")[0].split("|")
    }
    reward_model_tokenizer, multi_adapter_reward_model = load_adapter_model(
        args.reward_model_name_or_path,
        adapter_paths,
        model_args=ModelConfig(
            torch_dtype="auto",
            trust_remote_code=True,
        ),
    )
    multi_adapter_reward_model.eval()
    reward_model_tokenizer, multi_adapter_reward_model = accelerator.prepare(
        reward_model_tokenizer, multi_adapter_reward_model
    )

    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    assert isinstance(dataset, DatasetDict), "Only DatasetDict is supported for now"
    accelerator.print("Dataset loaded: ", dataset)

    # Flatten the completions
    dataset_flattened = {}
    n_completions = None
    for dataset_split_name, dataset_split in dataset.items():
        flat_data = {
            "prompt_id": [],
            "prompt": [],
            "completion": [],
        }
        for element in dataset_split:
            if n_completions is None:
                n_completions = len(element["completions"])
            for completion in element["completions"]:
                flat_data["prompt_id"].append(element["prompt_id"])
                flat_data["prompt"].append(element["prompt"])
                flat_data["completion"].append(completion)
        dataset_flattened[dataset_split_name] = Dataset.from_dict(flat_data)
        del flat_data
    dataset_flattened = DatasetDict(dataset_flattened)
    accelerator.print("Dataset flattened: ", dataset_flattened)

    # Tokenize the dataset
    dataset_flattened = dataset_flattened.map(
        lambda x: maybe_apply_chat_template(x, reward_model_tokenizer)
    )
    dataset_inputs = dataset_flattened.map(
        lambda x: reward_model_tokenizer(x["prompt"] + x["completion"]),
        remove_columns=["prompt", "completion"],
    )  # Columns: prompt_id, input_ids, attention_mask
    accelerator.print("Dataset inputs: ", dataset_inputs)

    # Run the reward model
    data_collator = DataCollatorWithPadding(tokenizer=reward_model_tokenizer)
    rewards = {}
    for data_split_name, data_split in dataset_inputs.items():
        accelerator.print(f"\n--- Processing {data_split_name} split ---")
        accelerator.print(
            f"Number of datapoints in {data_split_name}: {len(data_split)}"
        )
        dataloader = torch.utils.data.DataLoader(
            data_split,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=data_collator,
        )
        dataloader = accelerator.prepare(dataloader)
        data_split_rewards = defaultdict(list)
        for i, batch in enumerate(
            tqdm.tqdm(dataloader, desc=f"Generating {data_split_name} rewards")
        ):
            prompts = accelerator.gather(batch["prompt_id"]).cpu().tolist()
            if accelerator.is_main_process:
                data_split_rewards["prompt_id"].extend(prompts)
            with torch.no_grad(), unwrap_model_for_generation(
                multi_adapter_reward_model, accelerator
            ) as unwrapped_model:
                for adapter_name in unwrapped_model.peft_config.keys():
                    unwrapped_model.set_adapter(adapter_name)
                    output = unwrapped_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    accelerator.wait_for_everyone()
                    output = accelerator.gather(output.logits.reshape(-1)).cpu()
                    if accelerator.is_main_process:
                        data_split_rewards[adapter_name].extend(output.tolist())
                    del output
                if i == 0 and accelerator.is_main_process:
                    accelerator.print(
                        f"Output after first iteration: {data_split_rewards}"
                    )
            torch.cuda.empty_cache()
        if accelerator.is_main_process:
            # Truncate the output to the number of input samples to remove dummy rows due to Accelerator
            data_split_rewards = {
                key: values[: len(data_split)]
                for key, values in data_split_rewards.items()
            }
            for key, value in data_split_rewards.items():
                print(f"{key} length: {len(value)}")

            # Unflatten the rewards
            data_split_rewards = {
                key: [
                    values[i : i + n_completions]
                    for i in range(0, len(values), n_completions)
                ]
                for key, values in data_split_rewards.items()
            }
            for key, value in data_split_rewards.items():
                print(f"{key} length: {len(value)}")
            single_prompt_id = []
            for x in data_split_rewards["prompt_id"]:
                if len(set(x)) == 1:
                    single_prompt_id.append(x[0])
                else:
                    raise ValueError(
                        "Prompt IDs are not the same for all completions: ", x
                    )
            data_split_rewards["prompt_id"] = single_prompt_id
            assert all(
                len(values) == len(dataset[data_split_name])
                for values in data_split_rewards.values()
            ), (
                "The number of rewards does not match the number of completions: "
                + ", ".join(
                    [
                        f"{key}: {len(values)}"
                        for key, values in data_split_rewards.items()
                    ]
                )
            )
            data_split_rewards = Dataset.from_dict(data_split_rewards)
            rewards[data_split_name] = data_split_rewards
    rewards = DatasetDict(rewards)
    accelerator.print(rewards)

    # Save the rewards
    output_path = args.dataset_path + "__rewards"
    rewards.save_to_disk(output_path)

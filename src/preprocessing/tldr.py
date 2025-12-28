import datasets
from typing import Dict, Tuple, Union

import pandas as pd
from transformers import PreTrainedTokenizerBase

from src.preprocessing.utils import chat_structure

"""
Example command to run:
python src/preprocessing/tldr.py --output_dir data/datasets/tldr

Preprocess the TL;DR dataset for training and get the SFT, validation, and test datasets to a common format:
{
    "id": str, # The unique identifier
    "prompt": str, # The TL;DR prompt
    "completion": Union[str, Dict[str, str], List[str]], # The completion
}
the completion is either a string, a list of strings, a dictionary with the following format:
{
    "chosen: str, # The chosen completion
    "rejected": str, # The rejected completion
}
"""


def construct_tldr_prompt(
    example_info: dict,
) -> str:
    """Construct the TL;DR prompt."""
    if example_info["site"] is not None:
        query = (
            "SITE: " + example_info["site"] + "\n\nARTICLE: " + example_info["article"]
        )
    elif example_info["subreddit"] is not None:
        query = (
            "SUBREDDIT: r/"
            + example_info["subreddit"]
            + "\n\nTITLE: "
            + example_info["title"]
            + "\n\nPOST: "
            + example_info["post"]
        )
    else:
        raise ValueError
    return query


def truncate_prompt_and_extract_summaries(
    element: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_length: int = None,
) -> Dict[str, str]:
    """Truncate the prompt and extract the summaries."""
    if "prompt" in element:
        prompt = element["prompt"]
    else:
        prompt = construct_tldr_prompt(element["info"])
    if max_prompt_length is not None:
        while len(tokenizer(prompt)["input_ids"]) > max_prompt_length:
            prompt = "\n".join(prompt.split("\n")[:-1])
    prompt = prompt + "\n\nTL;DR:"
    return {
        "prompt": prompt,
        "prompt_id": element["prompt_id"] if "prompt_id" in element else element["id"],
    }


def load_dataset(
    dataset_name_or_path: str,
    dataset_config_name: str,
    min_annotation_per_worker: int,
    worker_id: str = None,
) -> Tuple[Union[datasets.DatasetDict, datasets.Dataset], pd.DataFrame]:
    print("Loading dataset...")
    dataset = datasets.load_dataset(dataset_name_or_path, dataset_config_name)
    print("Number of datapoints")
    for dataset_name, data in dataset.items():
        print(dataset_name, len(data))

    # Filter tha validations for workers with sufficient annotations, keep all datapoints in the training set
    annotation_counts_train = pd.Series(dataset["train"]["worker"]).value_counts()
    annotation_counts_valid = pd.Series(dataset["validation"]["worker"]).value_counts()
    workers_used = list(
        annotation_counts_train.index[
            annotation_counts_train >= min_annotation_per_worker
        ]
    )
    workers_used = [x for x in workers_used if x in annotation_counts_valid.index]
    if worker_id is not None:
        assert (
            worker_id in workers_used
        ), f"Worker {worker_id} is either not present in the dataset or does not have enough annotations."
        workers_used = [worker_id]
    print("Workers used: ", workers_used)
    print("Number of samples per worker: ")
    df = pd.concat([annotation_counts_train, annotation_counts_valid], axis=1)
    df.columns = ["train", "valid"]
    print(df.loc[workers_used, :])
    dataset["validation"] = dataset["validation"].filter(
        lambda x: x["worker"] in workers_used
    )
    print("Number of datapoints after filtering workers")
    for dataset_name, data in dataset.items():
        print(dataset_name, len(data))
    return dataset, df


def prepare_dataset(
    dataset: Union[datasets.Dataset, datasets.DatasetDict],
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_length: int,
    max_response_length: int,
    seed: int,
):
    """pre-tokenize the dataset before training; only collate during training"""

    def truncate_prompt_and_extract_summaries(example):
        prompt = construct_tldr_prompt(example["info"])
        choice = example["choice"]
        while len(tokenizer(prompt)["input_ids"]) > max_prompt_length:
            prompt = "\n".join(prompt.split("\n")[:-1])
        prompt = prompt + "\n\nTL;DR:"
        return {
            "prompt": prompt,
            "prompt_id": example["info"]["id"],
            "chosen": example["summaries"][choice]["text"],
            "rejected": example["summaries"][1 - choice]["text"],
        }

    return (
        dataset.map(
            truncate_prompt_and_extract_summaries,
            remove_columns=["info", "summaries", "choice", "extra", "split", "batch"],
        )
        .filter(
            lambda x: not x["prompt"].endswith("\n\nPOST:\n\nTL;DR:")
        )  # Remove if prompt is non-existent
        .filter(
            lambda x: all(
                [
                    len(tokenizer(x[summary_name])["input_ids"]) <= max_response_length
                    for summary_name in ["chosen", "rejected"]
                ]
            )
        )  # Keep only short summaries
        .map(chat_structure, remove_columns=["prompt"])
        .shuffle(seed=seed)
    )

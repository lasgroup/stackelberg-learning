import datasets
import numpy as np
import torch

from typing import Iterable

from src.preprocessing.utils import chat_structure


def load_dataset(
    label_names: Iterable = (
        "helpfulness",
        "correctness",
        "coherence",
        "complexity",
        "verbosity",
    ),
    seed: int = 42,
    train_validation_split: float = 0.8,
) -> datasets.Dataset:
    """
    Load dataset:

    Returns a Dataset with the following format:
    {
        "prompt_id": int,
        "prompt": str,
        "completions": List[
            {
                "text": str,
                "helpfulness": List[float],
                "correctness": List[float],
                "coherence": List[float],
                "complexity": List[float],
                "verbosity": List[float],
            }
        ],
        "labels": {
            "helpfulness": int,
            "correctness": int,
            "coherence": int,
            "complexity": int,
            "verbosity": int,
        },
    }
    Labels are 1 if the first completion wins and 0 otherwise.
    Labels are calculated based on the average score across each dimension.
    In case of ties, the lower standard deviation wins.
    In case of ties in both mean and standard deviation, a random label is assigned.
    """
    # Load dataset
    dataset = datasets.load_dataset("nvidia/HelpSteer2", data_dir="disagreements")[
        "train"
    ]

    # Get the prompt and completions
    ds = {"prompt": [], "completions": []}
    for i in range(0, len(dataset), 2):
        assert (
            dataset[i]["prompt"] == dataset[i + 1]["prompt"]
        ), "Prompts of consequent entries must be equal"
        ds["prompt"].append(dataset[i]["prompt"])
        ds["completions"].append(
            [
                {
                    "text": dataset[i + j]["response"],
                    **{
                        label_name: dataset[i + j][label_name]
                        for label_name in label_names
                    },
                }
                for j in [0, 1]
            ]
        )
    ds["prompt_id"] = list(range(len(ds["prompt"])))
    dataset = datasets.Dataset.from_dict(ds)

    # Calculate the labels
    def get_labels(element):
        labels = {}
        for name in label_names:
            mean_diff = np.mean(element["completions"][0][name]) - np.mean(
                element["completions"][1][name]
            )
            std_diff = np.std(element["completions"][0][name]) - np.std(
                element["completions"][1][name]
            )
            if np.abs(mean_diff) <= 1e-8:  # Tie in the means
                # Still tie then labels are the same -> random label
                if std_diff <= 1e-8:
                    labels[name] = torch.randint(0, 2, size=()).item()
                elif std_diff > 0:  # Second completion wins
                    labels[name] = 0
                else:  # First completion wins
                    labels[name] = 1
            elif mean_diff > 0:
                labels[name] = 1
            else:
                labels[name] = 0

            # If label name is "verbosity" or "complexity" flip the label
            if name in ["verbosity", "complexity"]:
                labels[name] = 1 - labels[name]
        return {"labels": labels}

    dataset = dataset.map(get_labels)

    # Split the dataset into train and validation
    dataset = dataset.train_test_split(
        train_size=train_validation_split,
        seed=seed,
    )
    # Rename test to validation
    dataset["validation"] = dataset["test"]
    del dataset["test"]
    return dataset


def prepare_dataset(
    dataset: datasets.Dataset,
    label_name: str,
) -> datasets.DatasetDict:
    """
    Select the label from the dataset and choose chosen/rejected completions accordingly
    """

    # Choose completions based on label
    def get_completions(element):
        chosen_idx = 1 - element["labels"][label_name]  # 1 if the first completion wins
        return {
            "chosen": element["completions"][chosen_idx]["text"],
            "rejected": element["completions"][1 - chosen_idx]["text"],
        }

    dataset = dataset.map(get_completions, remove_columns=["completions", "labels"])

    # Get chat structure
    dataset = dataset.map(chat_structure, remove_columns=["prompt"])

    return dataset

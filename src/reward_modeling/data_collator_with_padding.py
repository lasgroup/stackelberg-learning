from dataclasses import dataclass
from typing import Union, Optional, Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class RewardDataCollatorWithPaddingModified:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        pad_to_multiple_of (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        loss_weight = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        has_loss_weight = "loss_weight" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                    "input_ids_chosen" not in feature
                    or "input_ids_rejected" not in feature
                    or "attention_mask_chosen" not in feature
                    or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])
            if has_loss_weight:
                loss_weight.append(feature["loss_weight"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        if has_loss_weight:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float)
            batch["loss_weight"] = loss_weight
        return batch



@dataclass
class PreferenceDataCollatorWithPadding:
    r"""
    Preference DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        pad_to_multiple_of (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_messages = []
        labels_messages = []
        margin = []
        loss_weight = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        has_loss_weight = "loss_weight" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                    "input_ids_messages" not in feature
                    or "attention_mask_messages" not in feature
                    or "labels" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_messages`, `attention_mask_messages` and `labels`"
                )

            features_messages.append(
                {
                    "input_ids": feature["input_ids_messages"],
                    "attention_mask": feature["attention_mask_messages"],
                }
            )
            labels_messages.append(feature["labels"])
            if has_margin:
                margin.append(feature["margin"])
            if has_loss_weight:
                loss_weight.append(feature["loss_weight"])
        batch_messages = self.tokenizer.pad(
            features_messages,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_messages": batch_messages["input_ids"],
            "attention_mask_messages": batch_messages["attention_mask"],
            "labels": torch.tensor(labels_messages, dtype=torch.float),
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        if has_loss_weight:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float)
            batch["loss_weight"] = loss_weight
        return batch

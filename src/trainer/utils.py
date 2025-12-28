import torch
import torch.nn.functional as F
from collections.abc import Mapping, Sequence


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def multiply_list(arr, multiplicty):
    return [x for x in arr for _ in range(multiplicty)]


def move_state_dict_to_cpu_backup(data):
    """
    Recursively traverses a state dictionary (or part of it).
    Clones, detaches, and moves any found torch.Tensor to the CPU.
    Handles nested dicts, lists, and tuples.
    """
    if isinstance(data, torch.Tensor):
        # The core operation for backup: clone, detach, move to CPU
        return data.clone().detach().cpu()
    elif isinstance(data, Mapping): # Checks for dict or dict-like objects
        # Recursively process dictionary values
        return {k: move_state_dict_to_cpu_backup(v) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)): # Checks for list, tuple, etc., but not strings/bytes
        # Recursively process sequence items and preserve sequence type (list or tuple)
        return type(data)(move_state_dict_to_cpu_backup(item) for item in data)
    else:
        # Return non-tensor, non-collection types as is (int, float, str, bool, None, etc.)
        return data

# --- Helper Function: Move state dict tensors to a target device for restore ---
def move_state_dict_to_device(data, device):
    """
    Recursively traverses a state dictionary (or part of it).
    Moves any found torch.Tensor to the specified target device.
    Handles nested dicts, lists, and tuples.
    """
    if isinstance(data, torch.Tensor):
        # The core operation for restore: move to target device
        return data.to(device)
    elif isinstance(data, Mapping):
        # Recursively process dictionary values
        return {k: move_state_dict_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        # Recursively process sequence items and preserve sequence type
        return type(data)(move_state_dict_to_device(item, device) for item in data)
    else:
        # Return non-tensor, non-collection types as is
        return data

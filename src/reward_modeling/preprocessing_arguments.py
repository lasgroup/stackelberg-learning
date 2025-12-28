from dataclasses import dataclass, field


@dataclass
class PreprocessingArguments:
    max_prompt_length: int = field(
        default=None, metadata={"help": "Max token length of the prompt"}
    )
    max_response_length: int = field(
        default=None, metadata={"help": "Max token length of the responses"}
    )
    min_annotation_per_worker: int = field(
        default=None,
        metadata={
            "help": "Minimum number of annotations per each worker in the preference dataset."
        },
    )
    worker_id: str = field(
        default=None,
        metadata={
            "help": "Worker ID to filter the dataset by. Default is None and all workers are used."
        },
    )
    loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the loss of the worker_id. Default is 1.0."},
    )
    label_name: str = field(
        default=None, metadata={"help": "Label name to use for the HelpSteer2 dataset."}
    )

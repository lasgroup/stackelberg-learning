from dataclasses import dataclass, field
from typing import Optional

from trl.trainer.online_dpo_config import OnlineDPOConfig


@dataclass
class StackelbergGDAConfig(OnlineDPOConfig):
    r"""
    Configuration class for the [`StackelbergPGTrainer`].

    Subclass of [`OnlineDPOConfig`] we can use all its arguments and add the following:

    Parameters:
        follower_weight (`float`, *optional*, defaults to 2.0):
    """

    reward_model_adapters_path: str = field(
        default=None,
        metadata={
            "help": "regex expression for the reward_model_adapters_paths",
        },
    )

    judge_temperature: float = field(
        default=0.9,
        metadata={
            "help": "The temperature to use for the judge model when scoring the leader and follower completions."
        },
    )

    follower_weight: float = field(
        default=2.0,
        metadata={
            "help": "Relative weight in the loss of the follower's loss compared to the leader's loss."
            "It should be set larger than 1.0 to enable convergence.",
        },
    )

    follower_prompt: str = field(
        default="Improve the previous answer. Phrase it as if it was the original response.",
        metadata={
            "help": "The prompt to use for the follower model to generate completions based on the leader's completion."
        },
    )

    follower_beta: float = field(
        default=None,
        metadata={
            "help": "KL-divergence coefficient for the follower. If not set, it uses the 'beta' argument."
        },
    )

    score_baseline: float = field(
        default=0.5,
        metadata={
            "help": "The baseline score for the follower model to use when calculating the reward."
            "This is used to normalize the reward signal.",
        },
    )

    top_k: int = field(
        default=0,
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

    generation_temperature: float = field(
        default=0.01,
        metadata={
            "help": "The temperature to use for sampling. Lower values make the model more deterministic."
        },
    )

    missing_eos_probability_penalty: float = field(
        default=0.0,
        metadata={
            "help": "The penalty to apply for missing the end of sequence token in the follower's response."
            "This is used to encourage the follower model to generate complete responses.",
        },
    )

    rloo_baseline: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the Leave-one-out (RLOO) baseline in the scoring function."
        },
    )

    separate_follower_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to use a separate PEFT adapter for the follower model."
        },
    )

    standard_follower_kl_regularization: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the standard KL divergence in the Follower's loss or without the Leader's response."
        },
    )

    kl_estimator: str = field(
        default="k1",
        metadata={
            "help": "The method to use for estimating the KL divergence in the Follower's loss. More info: http://joschu.net/blog/kl-approx.html"
            "Options: 'k1', 'k2', or 'k3'."
        },
    )

    max_clip_grad_norm: float = field(
        default=0.0,
        metadata={
            "help": "The maximum norm for gradient clipping. Set to 0 to disable gradient clipping."
        },
    )

    leader_update_frequency: int = field(
        default=1,
        metadata={
            "help": "The frequency of updating the leader model. Set to 1 to update every step."
        },
    )

    use_vllm: str = field(
        default=None,
        metadata={
            "help": "Whether to use vLLM for generation. If set, it will use vLLM for generating completions. Options: 'server` or 'local'."
        },
    )

    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )

    vllm_tis_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "Threshold for vLLM Token Importance Sampling (TIS). If `None` (default), TIS is disabled."
        },
    )

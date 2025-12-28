from typing import Any, Callable, Optional, Union
from contextlib import nullcontext, ExitStack
import os
import textwrap

import torch
import wandb
from accelerate.utils import (
    broadcast_object_list,
    gather_object,
    is_peft_model,
)

import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datasets import Dataset, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available, is_deepspeed_available
from trl.extras.vllm_client import VLLMClient
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.judges import BasePairwiseJudge
from trl.trainer.online_dpo_trainer import OnlineDPOTrainer
from trl.trainer.utils import empty_cache, truncate_right, pad, generate_model_card, get_comet_experiment_url
from trl.extras.profiling import profiling_context, profiling_decorator
import trl

from .DualModel import DualModel, switch_model, is_ddp_wrapped
from .stackelberg_gda_config import StackelbergGDAConfig

# TODO: Add apex support
# if is_apex_available():
#     from apex import amp

if is_wandb_available():
    pass

if is_deepspeed_available():
    import deepspeed


def toggle_peft_adapters(model, active_adapter_name):
    """
    Sets requires_grad=True for the active adapter and False for others.
    Works for PEFT/LoRA models wrapped in DDP.
    """
    # Unwrap DDP/FSDP if necessary to access named_parameters purely for name checking
    # (though we set attributes on the actual model params which persists through DDP)

    for name, param in model.named_parameters():
        # PEFT parameter names usually look like: ...layers.0.self_attn.q_proj.lora_A.leader
        if "lora_" in name or "adapter_" in name:  # Standard PEFT naming
            if f".{active_adapter_name}" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


class StackelbergGDATrainer(OnlineDPOTrainer):
    r"""
    Initialize NashMDTrainer as a subclass of [`OnlineDPOConfig`].

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        reward_model (`transformers.PreTrainedModel`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`NashMDConfig`):
            The NashMD config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        peft_config (`dict`):
            The peft config to use for training.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "stackelberg-pg"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[StackelbergGDAConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> None:
        # Sanity checks
        use_vllm = args.use_vllm
        args.use_vllm = False  # Overwrite to avoid issues in the parent class
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            judge=judge,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_class=processing_class,  # for now, NashMDTrainer can't use any reward model
            peft_config=peft_config,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.args.use_vllm = use_vllm
        assert self.args.use_vllm in [None, False, "server"], "args.use_vllm must be one of None, False, or 'server', 'local' support is not yet implemented"

        self._follower_weight = self.args.follower_weight
        self._follower_prompt = self.args.follower_prompt
        self.generation_config.top_k = self.args.top_k
        self.generation_config.top_p = self.args.top_p

        if args.use_vllm == "server":
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                # if trl version >= 0.19.1, pass device to init_communicator
                if trl.__version__ >= "0.19.1":
                    base_url = (
                        f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    )
                    self.vllm_client = VLLMClient(
                        base_url=base_url, connection_timeout=240.0,
                    )
                    self.vllm_client.init_communicator()
                else:
                    self.vllm_client = VLLMClient(
                        args.vllm_server_host,
                        args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()

            self._last_loaded_step = (
                -1
            )  # tag to avoid useless loading during grad accumulation

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()

        # Overwrite the stats dictionary to include NashMD specific statistics
        self.stats = {
            "rewards/probabilities": [],
            "rewards/margins": [],
            "loss/loss_leader": [],
            "loss/loss_follower": [],
            "loss/score_leader": [],
            "loss/score_follower": [],
            "loss/kl_leader": [],
            "loss/kl_follower": [],
            "logps/leader": [],
            "logps/follower": [],
            "objective/entropy_leader": [],
            "objective/entropy_follower": [],
            "val/leader_contain_eos_token": [],
            "val/follower_contain_eos_token": [],
            "val/leader_completion_length": [],
            "val/follower_completion_length": [],
            "beta": [],
            "follower_weight": [],
            # "rewards/accuracies": [],
        }
        if self.reward_model is not None:
            self.stats["rewards/chosen"] = []
            self.stats["rewards/rejected"] = []

    def _get_follower_prompt(self, prompt_texts, leader_completion_texts):
        follower_prompts = []
        for prompt, leader_text in zip(prompt_texts, leader_completion_texts):
            if is_conversational({"prompt": prompt}):
                follower_prompt = (
                    prompt
                    + [{"role": "assistant", "content": leader_text}]
                    + [{"role": "user", "content": self._follower_prompt}]
                )
                follower_prompts.append({"prompt": follower_prompt})
            else:
                follower_prompts.append(prompt + leader_text + self._follower_prompt)
        return follower_prompts

    def _generate_completions(self, model, prompts):
        """
        Generate completions for the given prompts using the model.
        :param model:
        :param prompts: Dict of the structure:
            prompts = {
                "input_ids": Tensor, Shape: (batch_size, input_length),
                "attention_mask": Tensor, Shape: (batch_size, input_length),
                "raw": str or List[Dict[str, str]],  # Raw prompt text (maybe conversational structure)
            }
        :return:
        """
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            if self.args.separate_follower_model:
                assert ~model.module.use_follower
            # Generate Leader Data
            with profiling_context(self, "HF.generate_leader"):
                leader_output = unwrapped_model.generate(
                    input_ids=prompts["input_ids"],
                    attention_mask=prompts["attention_mask"],
                    generation_config=self.generation_config,
                )  # Tensor, Shape: (batch_size, input_length+output_length)
            leader_data = self._process_completions(leader_output, prompts)
            del leader_output

            # Generate Prompt for the Follower
            follower_prompts = self._get_follower_prompt(
                prompts["raw"], leader_data["completion_text"]
            )

            follower_inputs = [
                maybe_apply_chat_template(x, self.processing_class)
                for x in follower_prompts
            ]
            follower_inputs = [
                self.tokenize_row(
                    x, self.model.config.is_encoder_decoder, self.processing_class
                )
                for x in follower_inputs
            ]
            follower_inputs = self.data_collator(follower_inputs)
            follower_inputs = {
                key: value.to(self.model.device)
                for key, value in follower_inputs.items()
            }
            follower_prompts = {
                "input_ids": follower_inputs["prompt_input_ids"],
                "attention_mask": follower_inputs["prompt_attention_mask"],
                "raw": follower_prompts,
            }
            if self.args.separate_follower_model:
                switch_model(unwrapped_model, "follower")
                assert unwrapped_model.use_follower
            with profiling_context(self, "HF.generate_follower"):
                follower_output = unwrapped_model.generate(
                    input_ids=follower_inputs["prompt_input_ids"],
                    attention_mask=follower_inputs["prompt_attention_mask"],
                    generation_config=self.generation_config,
                )  # Tensor, Shape: (batch_size, input_length+output_length)
            if self.args.separate_follower_model:
                switch_model(unwrapped_model, "leader")
                assert ~unwrapped_model.use_follower

            follower_data = self._process_completions(follower_output, follower_prompts)
            del follower_output, follower_inputs, follower_prompts
            empty_cache()
        return leader_data, follower_data

    def _generate_vllm_completions(self, prompts):
        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_prompts_text = gather_object(prompts["processed_text"])
        if self.accelerator.is_main_process:
            outputs = self.vllm_client.generate(
                prompts=all_prompts_text,
                n=1,
                repetition_penalty=1.0,
                temperature=self.args.generation_temperature,
                top_p=self.args.top_p,
                top_k=-1 if self.args.top_k == 0 else self.args.top_k,
                min_p=0.0,
                max_tokens=self.args.max_new_tokens,
                guided_decoding_regex=self.guided_decoding_regex,
            )
            if trl.__version__ >= "0.23.0":
                completion_ids = outputs["completion_ids"]
                if self.args.vllm_tis_threshold:
                    logprobs = outputs["logprobs"]
                else:
                    logprobs = [None]*len(all_prompts_text)
            else:
                completion_ids = outputs
                logprobs = [None]*len(all_prompts_text)
        else:
            completion_ids = [None] * len(all_prompts_text)
            logprobs = [None] * len(all_prompts_text)
        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # corresponding slice.
        completion_ids_all, logprobs_all = broadcast_object_list(
            [completion_ids, logprobs], from_process=0
        )
        process_slice = slice(
            self.accelerator.process_index * len(prompts["processed_text"]),
            (self.accelerator.process_index + 1) * len(prompts["processed_text"]),
        )
        completion_ids_local = completion_ids_all[process_slice]
        del completion_ids_all
        completions_text = self.processing_class.batch_decode(
            completion_ids_local, skip_special_tokens=True
        )
        completions_text = [
            completion.strip() for completion in completions_text
        ]

        # Pad the completions, and concatenate them with the prompts
        completion_ids_local = [
            torch.tensor(ids, device=self.model.device) for ids in completion_ids_local
        ]
        completion_ids_local = pad(
            completion_ids_local, padding_value=self.processing_class.pad_token_id
        )
        completion_mask = (
            completion_ids_local != self.processing_class.pad_token_id
        ).to(self.model.device)
        completion_ids_local = torch.cat(
            [prompts["input_ids"], completion_ids_local], dim=1
        )
        completion_mask = torch.cat(
            [prompts["attention_mask"], completion_mask], dim=1
        )

        # Get logprobs if needed for TIS
        if self.args.vllm_tis_threshold:
            logprobs_local = logprobs_all[process_slice]
            del logprobs_all
            logprobs_local = [
                torch.tensor(lp, device=self.model.device) for lp in logprobs_local
            ]
            logprobs_local = pad(logprobs_local, padding_value=torch.nan)
        else:
            logprobs_local = None
        return completions_text, completion_ids_local, completion_mask, logprobs_local

    def _generate_completions_server_vllm(self, prompts):
        """
        Generate completions for the given prompts using vLLM server.
        This method is a placeholder and should be implemented if vLLM server support is required.
        """
        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        with profiling_context(self, "vLLM.generate_leader"):
            leader_completions_text, leader_completion_ids, leader_completion_mask, leader_vllm_logprobs = self._generate_vllm_completions(prompts)
        leader_data = {
            "input_ids": leader_completion_ids,
            "attention_mask": leader_completion_mask,
            "raw": prompts["raw"],
            "completion_text": leader_completions_text,
            "context_length": prompts["input_ids"].shape[1],
            "generation_logprobs": leader_vllm_logprobs,
        }

        follower_prompts = self._get_follower_prompt(
            prompts["raw"], leader_completions_text
        )
        follower_prompts_processed = [
            maybe_apply_chat_template(x, self.processing_class)
            for x in follower_prompts
        ]
        follower_inputs = [
            self.tokenize_row(
                x, self.model.config.is_encoder_decoder, self.processing_class
            )
            for x in follower_prompts_processed
        ]
        follower_inputs = self.data_collator(follower_inputs)
        follower_inputs = {
            key: value.to(self.model.device) for key, value in follower_inputs.items()
        }
        follower_prompts = {
            "input_ids": follower_inputs["prompt_input_ids"],
            "attention_mask": follower_inputs["prompt_attention_mask"],
            "raw": follower_prompts,
            "processed_text": [x["prompt"] for x in follower_prompts_processed],
        }

        with profiling_context(self, "vLLM.generate_follower"):
            follower_completions_text, follower_completion_ids, follower_completion_mask, follower_vllm_logprbs = self._generate_vllm_completions(follower_prompts)
        follower_data = {
            "input_ids": follower_completion_ids,
            "attention_mask": follower_completion_mask,
            "raw": follower_prompts["raw"],
            "completion_text": follower_completions_text,
            "context_length": follower_prompts["input_ids"].shape[1],
            "generation_logprobs": follower_vllm_logprbs,
        }
        return leader_data, follower_data

    @profiling_decorator
    def _move_model_to_vllm(self):
        """
        Method to move model weights to the vLLM server when args.use_vllm == "server".
        Sourced from the GRPOTrainer: https://github.com/huggingface/trl/blob/v0.17.0/trl/trainer/grpo_trainer.py
        """
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = (
            deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        )

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(
                        ".base_layer", ""
                    )
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    def _process_completions(self, output, prompts):
        """
        Process the completions generated by the model.
        Adds completion mask and pads out all non-padding token after the first eos_token
        """
        context_length = prompts["input_ids"].shape[1]
        completion_ids = output[:, context_length:]
        completion_ids, completion_mask = truncate_right(
            completion_ids,
            self.processing_class.eos_token_id,
            self.processing_class.pad_token_id,
        )
        completion_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        completion_text = [completion.strip() for completion in completion_text]
        completion_ids = torch.cat((prompts["input_ids"], completion_ids), dim=1)
        completion_mask = torch.cat((prompts["attention_mask"], completion_mask), dim=1)
        return {
            "input_ids": completion_ids,
            "attention_mask": completion_mask,
            "raw": prompts["raw"],
            "completion_text": completion_text,
            "context_length": context_length,
        }

    def _compute_rewards(self, leader_data, follower_data):
        raise NotImplementedError

    @profiling_decorator
    def _compute_judge(self, leader_data, follower_data):
        prompts = leader_data["raw"]
        leader_data_completions = leader_data["completion_text"]
        follower_data_completions = follower_data["completion_text"]
        if is_conversational({"prompt": prompts[0]}):
            leader_data_completions = [
                [{"role": "assistant", "content": completion}]
                for completion in leader_data_completions
            ]
            follower_data_completions = [
                [{"role": "assistant", "content": completion}]
                for completion in follower_data_completions
            ]
        if hasattr(self.judge, "missing_eos_penalty") and (
            self.judge.missing_eos_penalty is not None
        ):
            leader_contain_eos = torch.any(
                leader_data["input_ids"][:, leader_data["context_length"] :]
                == self.processing_class.eos_token_id,
                dim=-1,
            ).tolist()
            follower_contain_eos = torch.any(
                follower_data["input_ids"][:, follower_data["context_length"] :]
                == self.processing_class.eos_token_id,
                dim=-1,
            ).tolist()
            contain_eos_tokens = list(zip(leader_contain_eos, follower_contain_eos))
        else:
            contain_eos_tokens = None
        probability = self.judge.judge(
            prompts,
            list(zip(leader_data_completions, follower_data_completions)),
            contain_eos_tokens=contain_eos_tokens,
            return_scores=True,
        )
        return torch.tensor(probability, device=leader_data["input_ids"].device)

    @profiling_decorator
    def _compute_logprobs(
        self, model, model_data, with_grad=True, disable_adapters=False
    ):
        context_length = model_data["context_length"]
        contexts = []
        if not with_grad:
            contexts.append(torch.no_grad())
        if disable_adapters:
            contexts.append(model.disable_adapter())
        with ExitStack() as stack:
            for ctx in contexts:
                stack.enter_context(ctx)
            output = model(
                model_data["input_ids"], attention_mask=model_data["attention_mask"]
            )
        logits = output.logits[:, context_length - 1 : -1]
        del output

        logits.div_(self.args.temperature + 1e-7)
        logprobs = []
        for i in range(logits.shape[0]):  # loop over batch to avoid CUDA OOM
            logprobs.append(
                F.log_softmax(logits[i], dim=-1)
            )
        logprobs = torch.stack(logprobs, dim=0)
        del logits
        empty_cache()

        target_ids = model_data["input_ids"][:, context_length:].unsqueeze(-1)
        model_logprobs_model_data = torch.gather(logprobs, 2, target_ids).squeeze(-1)
        del target_ids, logprobs
        empty_cache()

        # Mask padding tokens
        model_padding_mask = model_data["attention_mask"][:, context_length:] == 0
        model_logprobs_model_data = model_logprobs_model_data.masked_fill(
            model_padding_mask, 0.0
        )
        del model_padding_mask
        empty_cache()
        return model_logprobs_model_data

    @profiling_decorator
    def _compute_losses(self, model_logprobs, ref_logprobs, probability, generation_logprobs=None, beta=None):
        # reinforce score where score_baseline (default: 0.5) is a control variate
        if self.args.score_baseline is not None:
            probability = probability - self.args.score_baseline
        if self.args.rloo_baseline:
            prob_rloo = (probability.sum() - probability) / (probability.shape[0] - 1)
            probability = probability - prob_rloo

        score = probability * model_logprobs.detach().sum(1)

        # kl divergence via reinforce
        # Different definitions of estimating the KL divergence following: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            logr = ref_logprobs - model_logprobs
            if self.args.kl_estimator == "k1":
                log_ratio = -logr
            elif self.args.kl_estimator == "k2":
                log_ratio = 0.5*(logr**2)
            elif self.args.kl_estimator == "k3":
                logr = torch.clamp(logr, -40.0, 40.0)
                log_ratio = torch.expm1(logr) - logr
            else:
                raise ValueError("kl_estimator must be one of k1, k2, or k3")
            kl_div_log = log_ratio.sum(1)

        # final loss
        if beta is None:
            beta = self.beta
        loss = (beta*log_ratio - probability.unsqueeze(1)) * model_logprobs
        if generation_logprobs is not None:
            assert self.args.vllm_tis_threshold is not None and self.args.vllm_tis_threshold > 0.0, "generation_logprobs should only be passed when TIS is used with vLLM"
            # Add TIS penalty if applicable
            tis_penalty =  torch.clamp(
                torch.exp(model_logprobs.detach() - generation_logprobs), max=self.args.vllm_tis_threshold
            )
            loss *= tis_penalty
        loss = loss.sum(1)

        return loss.mean(), score, kl_div_log

    @profiling_decorator
    def _log_statistics(
        self,
        leader_data,
        follower_data,
        model_logprobs_leader_data,
        model_logprobs_follower_data,
        probability,
        loss_leader,
        loss_follower,
        score_leader,
        score_follower,
        kl_div_leader,
        kl_div_follower,
    ):
        # Helper to compute local means (keeps on device)
        def mean(x):
            return x.mean()

        # Precompute sums and derived values once (per-sample sums -> per-batch means)
        with torch.no_grad():
            model_logprobs_leader_sum = model_logprobs_leader_data.sum(1)
            model_logprobs_follower_sum = model_logprobs_follower_data.sum(1)

            entropy_leader = -model_logprobs_leader_sum
            entropy_follower = -model_logprobs_follower_sum
            margin = model_logprobs_leader_sum - model_logprobs_follower_sum

            leader_eos = (
                (
                    leader_data["input_ids"][:, leader_data["context_length"] :]
                    == self.processing_class.eos_token_id
                )
                .any(dim=1)
                .float()
            )
            follower_eos = (
                (
                    follower_data["input_ids"][:, follower_data["context_length"] :]
                    == self.processing_class.eos_token_id
                )
                .any(dim=1)
                .float()
            )
            leader_completion_length = (
                leader_data["attention_mask"][:, leader_data["context_length"] :].sum(1).float()
            )
            follower_completion_length = (
                follower_data["attention_mask"][:, follower_data["context_length"] :].sum(1).float()
            )

        # Build a single tensor of local metrics (float32 on current device)
        metrics_local = torch.stack(
            [
                mean(score_leader),
                mean(score_follower),
                mean(kl_div_leader),
                mean(kl_div_follower),
                mean(loss_leader),
                mean(loss_follower),
                mean(model_logprobs_leader_sum),
                mean(model_logprobs_follower_sum),
                mean(probability),
                mean(entropy_leader),
                mean(entropy_follower),
                mean(margin),
                mean(leader_eos),
                mean(follower_eos),
                mean(leader_completion_length),
                mean(follower_completion_length),
            ]
        ).unsqueeze(0)

        # Gather metrics from all processes (accelerator-specific)
        metrics_gathered = self.accelerator.gather_for_metrics(metrics_local).mean(0)
        metrics_mean_list = metrics_gathered.cpu().tolist()

        (
            score_leader_m,
            score_follower_m,
            kl_leader_m,
            kl_follower_m,
            loss_leader_m,
            loss_follower_m,
            logp_leader_m,
            logp_follower_m,
            prob_m,
            ent_leader_m,
            ent_follower_m,
            margin_m,
            eos_leader_m,
            eos_follower_m,
            leader_len_m,
            follower_len_m,
        ) = metrics_mean_list

        # Store results (already CPU scalars)
        self.stats["loss/score_leader"].append(score_leader_m)
        self.stats["loss/score_follower"].append(score_follower_m)
        self.stats["loss/kl_leader"].append(kl_leader_m)
        self.stats["loss/kl_follower"].append(kl_follower_m)
        self.stats["loss/loss_leader"].append(loss_leader_m)
        self.stats["loss/loss_follower"].append(loss_follower_m)
        self.stats["logps/leader"].append(logp_leader_m)
        self.stats["logps/follower"].append(logp_follower_m)
        self.stats["rewards/probabilities"].append(prob_m)
        self.stats["objective/entropy_leader"].append(ent_leader_m)
        self.stats["objective/entropy_follower"].append(ent_follower_m)
        self.stats["rewards/margins"].append(margin_m)
        self.stats["val/leader_contain_eos_token"].append(eos_leader_m)
        self.stats["val/follower_contain_eos_token"].append(eos_follower_m)
        self.stats["val/leader_completion_length"].append(leader_len_m)
        self.stats["val/follower_completion_length"].append(follower_len_m)

        # Scalars local to this process (no gather needed)
        self.stats["beta"].append(self.beta)
        self.stats["follower_weight"].append(self._follower_weight)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        # Apply chat template and tokenize the input
        batch_size = len(next(iter(inputs.values())))
        prompts = inputs["prompt"]
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        processed_prompts = [
            maybe_apply_chat_template(x, self.processing_class) for x in inputs
        ]
        input_ids = [
            self.tokenize_row(
                x, self.model.config.is_encoder_decoder, self.processing_class
            )
            for x in processed_prompts
        ]
        input_ids = self.data_collator(input_ids)
        input_ids = self._prepare_inputs(input_ids)
        prompts = {
            "input_ids": input_ids["prompt_input_ids"],
            "attention_mask": input_ids["prompt_attention_mask"],
            "raw": prompts,
            "processed_text": [x["prompt"] for x in processed_prompts],
        }
        del inputs, input_ids, processed_prompts
        if self.args.use_vllm == "server":
            leader_data, follower_data = self._generate_completions_server_vllm(
                prompts
            )
        else:
            # Sample completions from both the leader and the follower
            leader_data, follower_data = self._generate_completions(model, prompts)
        del prompts
        empty_cache()

        # Compute rewards
        if self.reward_model is not None:
            model_scores, mixture_scores = self._compute_rewards(
                leader_data, follower_data
            )
            # probability of the model data vs the mixture data
            probability = F.sigmoid(model_scores - mixture_scores)
            del model_scores, mixture_scores
        else:
            model_scores, mixture_scores = None, None
            probability = self._compute_judge(leader_data, follower_data)

        # Learning prep
        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        # Compute reference logprobs
        ref_logprobs_follower_data, ref_logprobs_leader_data = (
            self._compute_ref_model_logprobs(follower_data, leader_data, model)
        )

        # Compute the loss for the Leader
        if self.args.separate_follower_model:
            switch_model(model, "leader")
            toggle_peft_adapters(model, "leader")
            assert ~model.module.use_follower
        model_logprobs_leader_data = self._compute_logprobs(
            model, leader_data, with_grad=True
        )
        leader_contain_eos = torch.detach(
            torch.any(
                leader_data["input_ids"][:, leader_data["context_length"] :]
                == self.processing_class.eos_token_id,
                dim=-1,
            )
        )  # Tensor, Shape: (batch_size,)
        loss_leader, score_leader, kl_div_leader = self._compute_losses(
            model_logprobs_leader_data,
            ref_logprobs_leader_data,
            probability
            - self.args.missing_eos_probability_penalty * (~leader_contain_eos).float(),
        )
        del ref_logprobs_leader_data, leader_contain_eos
        empty_cache()

        # Compute loss for the Follower
        if self.args.separate_follower_model:
            # Do a backward pass for the Leader first
            if self.args.n_gpu > 1:
                leader_loss = loss_leader.mean()
            else:
                leader_loss = loss_leader

            if self.use_apex:
                raise NotImplementedError("Apex support is not yet implemented.")
            else:
                self.accelerator.backward(leader_loss, **kwargs)

            # Switch to follower adapter
            toggle_peft_adapters(model, "follower")
            switch_model(model, "follower")
            assert model.module.use_follower
            model_logprobs_follower_data = self._compute_logprobs(
                model, follower_data, with_grad=True
            )

            follower_contain_eos = torch.detach(
                torch.any(
                    follower_data["input_ids"][:, follower_data["context_length"] :]
                    == self.processing_class.eos_token_id,
                    dim=-1,
                )
            )

            loss_follower, score_follower, kl_div_follower = self._compute_losses(
                model_logprobs_follower_data,
                ref_logprobs_follower_data,
                1.0
                - probability
                - self.args.missing_eos_probability_penalty
                * (~follower_contain_eos).float(),
                beta=self.args.follower_beta,
            )

            if self.args.n_gpu > 1:
                follower_loss = loss_follower.mean()
            else:
                follower_loss = loss_follower

            # Apply backward pass for follower model
            self.accelerator.backward(follower_loss, **kwargs)

            # Calculate total loss for reporting
            total_loss = (leader_loss + follower_loss) / 2.0

            # Switch back to leader adapter for next iteration
            switch_model(model, "leader")
        else:
            model_logprobs_follower_data = self._compute_logprobs(
                model, follower_data, with_grad=True
            )
            # switch_peft_adapter(model, "leader")
            follower_contain_eos = torch.detach(
                torch.any(
                    follower_data["input_ids"][:, follower_data["context_length"] :]
                    == self.processing_class.eos_token_id,
                    dim=-1,
                )
            )  # Tensor, Shape: (batch_size,)
            loss_follower, score_follower, kl_div_follower = self._compute_losses(
                model_logprobs_follower_data,
                ref_logprobs_follower_data,
                1.0
                - probability  # Negate the probability for the follower
                - self.args.missing_eos_probability_penalty
                * (~follower_contain_eos).float(),
            )
            del ref_logprobs_follower_data, follower_contain_eos
            empty_cache()

            assert (
                self.args.leader_update_frequency >= 1
            ), "leader_update_frequency must be >= 1"
            if self.args.leader_update_frequency == 1:
                total_loss = loss_leader + self._follower_weight * loss_follower
            elif (self.state.global_step + 1) % self.args.leader_update_frequency == 0:
                total_loss = loss_leader
            else:
                total_loss = self._follower_weight * loss_follower

            if self.args.n_gpu > 1:
                total_loss = (
                    total_loss.mean()
                )  # mean() to average on multi-gpu parallel training
            if self.use_apex:
                raise NotImplementedError("Apex support is not yet implemented.")
                # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
            else:
                with profiling_context(self, "accelerator.backward"):
                    self.accelerator.backward(total_loss, **kwargs)

        # Training update preparation
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        probability = probability.detach()
        model_logprobs_leader_data = model_logprobs_leader_data.detach()
        self._log_statistics(
            leader_data,
            follower_data,
            model_logprobs_leader_data,
            model_logprobs_follower_data,
            probability,
            loss_leader,
            loss_follower,
            score_leader,
            score_follower,
            kl_div_leader,
            kl_div_follower,
        )

        if self.accelerator.sync_gradients and self.args.max_clip_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_clip_grad_norm,
            )

        return total_loss.detach() / self.args.gradient_accumulation_steps

    @profiling_decorator
    def _compute_ref_model_logprobs(self, follower_data, leader_data, model):
        """
        Compute the reference model's logprobabilities for the Leader and Follower completions
        Used to calculate the KL-divergence regularization
        """

        if self.args.standard_follower_kl_regularization:
            ref_model_data = follower_data
        else:
            ref_model_data = {
                key_name: torch.cat(
                    [
                        leader_data[key_name][:, : leader_data["context_length"]],
                        follower_data[key_name][:, follower_data["context_length"] :],
                    ],
                    dim=1,
                )
                for key_name in ["input_ids", "attention_mask"]
            }
            ref_model_data["context_length"] = leader_data["context_length"]
        if self.ref_model is None:
            with unwrap_model_for_generation(model, self.accelerator) as ref_model:
                if isinstance(ref_model, DualModel):
                    ref_logprobs_leader_data = self._compute_logprobs(
                        ref_model.leader,
                        leader_data,
                        with_grad=False,
                        disable_adapters=True,
                    )
                    ref_logprobs_follower_data = self._compute_logprobs(
                        ref_model.follower,
                        ref_model_data,
                        with_grad=False,
                        disable_adapters=True,
                    )
                else:
                    ref_logprobs_leader_data = self._compute_logprobs(
                        ref_model, leader_data, with_grad=False, disable_adapters=True
                    )
                    ref_logprobs_follower_data = self._compute_logprobs(
                        ref_model,
                        ref_model_data,
                        with_grad=False,
                        disable_adapters=True,
                    )
        else:
            ref_logprobs_leader_data = self._compute_logprobs(
                self.ref_model, leader_data, with_grad=False
            )
            ref_logprobs_follower_data = self._compute_logprobs(
                self.ref_model, ref_model_data, with_grad=False
            )
        return ref_logprobs_follower_data, ref_logprobs_leader_data

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)
                    if self.args.use_vllm == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = (
            deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        )

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # Update vLLM weights while parameters are gathered
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(
                            ".base_layer", ""
                        )
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        if self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
        empty_cache()

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        citation = textwrap.dedent("")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Online DPO",
            trainer_citation=citation,
            paper_title="Direct Language Model Alignment from Online AI Feedback",
            paper_id="2402.04792",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))

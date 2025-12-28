import torch
import torch.nn.functional as F
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models.utils import unwrap_model_for_generation
from trl import NashMDTrainer


class NashMDTrainerModified(NashMDTrainer):
    """
    Modified NashMDTrainer with minor modifications on data handling
    """

    def _get_completions(self, data, context_length):
        """Decode and strip the completions from the model data."""
        data_completions = self.processing_class.batch_decode(
            data["input_ids"][:, context_length:], skip_special_tokens=True
        )
        data_completions = [completion.strip() for completion in data_completions]
        return data_completions

    def _get_reward(self, prompts, completions):
        """
        Get reward scores for a batch of input texts.
        """
        input_texts = [
            maybe_apply_chat_template(
                {"prompt": p, "completion": c}, self.processing_class
            )
            for p, c in zip(prompts, completions)
        ]
        input_texts = [x["prompt"] + x["completion"] for x in input_texts]
        inputs = self.processing_class(input_texts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.reward_model.device)
        self.reward_model.eval()
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            scores = outputs.logits.reshape(-1)
        del inputs, outputs
        torch.cuda.empty_cache()
        return scores

    def _compute_judge(self, model_data, mixture_data, context_length):
        """
        Overwriting the original method to remove the usage of the SIMPLE_CHAT_TEMPLATE
        """
        prompts = model_data["raw"]
        model_data_completions = self._get_completions(model_data, context_length)
        mixture_data_completions = self._get_completions(mixture_data, context_length)

        if is_conversational({"prompt": prompts[0]}):
            model_data_completions = [
                [{"role": "assistant", "content": completion}]
                for completion in model_data_completions
            ]
            mixture_data_completions = [
                [{"role": "assistant", "content": completion}]
                for completion in mixture_data_completions
            ]

        if self.judge.missing_eos_penalty is not None:
            model_contain_eos = torch.any(
                model_data["input_ids"][:, context_length:]
                == self.processing_class.eos_token_id,
                dim=-1,
            ).tolist()
            mixture_contain_eos = torch.any(
                mixture_data["input_ids"][:, context_length:]
                == self.processing_class.eos_token_id,
                dim=-1,
            ).tolist()
            contain_eos_tokens = list(zip(model_contain_eos, mixture_contain_eos))
        else:
            contain_eos_tokens = None
        probability = self.judge.judge(
            prompts=prompts,
            completions=list(zip(model_data_completions, mixture_data_completions)),
            contain_eos_tokens=contain_eos_tokens,
            return_scores=True,
        )
        return torch.tensor(probability, device=model_data["input_ids"].device)

    def _compute_rewards(self, model_data, mixture_data, context_length):
        """
        Overwriting the original method to make sure that the chat_template is correctly applied and the EOS token is only considered for the completion
        Instead of concatenating the output tokens to the input, this method decodes the completions first and then apply the chat template again if needed
        This makes sure that it is consistent with the reward model
        """
        prompts = model_data["raw"]
        model_data_completions = self._get_completions(model_data, context_length)
        mixture_data_completions = self._get_completions(mixture_data, context_length)
        if is_conversational({"prompt": prompts[0]}):
            model_data_completions = [
                [{"role": "assistant", "content": completion}]
                for completion in model_data_completions
            ]
            mixture_data_completions = [
                [{"role": "assistant", "content": completion}]
                for completion in mixture_data_completions
            ]
        model_scores = self._get_reward(prompts, model_data_completions)
        mixture_scores = self._get_reward(prompts, mixture_data_completions)

        # Apply EOS penalty if needed
        if self.args.missing_eos_penalty is not None:
            model_contain_eos = torch.any(
                model_data["input_ids"][:, context_length:]
                == self.processing_class.eos_token_id,
                dim=-1,
            )  # Changed to only consider EOS in the completion
            mixture_contain_eos = torch.any(
                mixture_data["input_ids"][:, context_length:]
                == self.processing_class.eos_token_id,
                dim=-1,
            )  # Changed to only consider EOS in the completion
            model_scores[~model_contain_eos] -= self.args.missing_eos_penalty
            mixture_scores[~mixture_contain_eos] -= self.args.missing_eos_penalty

        return model_scores, mixture_scores

    def _compute_logprobs(self, model, model_data, context_length):
        def compute_logprobs_for_data(m, data):
            output = m(data["input_ids"], attention_mask=data["attention_mask"])
            logits = output.logits[:, context_length - 1 : -1]
            logprobs = F.log_softmax(logits, dim=-1)
            token_logprobs = torch.gather(
                logprobs, 2, data["input_ids"][:, context_length:].unsqueeze(-1)
            ).squeeze(-1)
            return token_logprobs

        # Compute logprobs for model completions under the model
        model_logprobs_model_data = compute_logprobs_for_data(model, model_data)

        # Compute logprobs of model completions under the reference model
        with torch.no_grad():
            if self.ref_model is None:
                with torch.no_grad(), unwrap_model_for_generation(
                    model, self.accelerator
                ) as unwrapped_model:
                    with unwrapped_model.disable_adapter():
                        ref_logprobs_model_data = compute_logprobs_for_data(
                            model, model_data
                        )
            else:
                ref_logprobs_model_data = compute_logprobs_for_data(
                    self.ref_model, model_data
                )

        # Mask padding tokens
        model_padding_mask = model_data["attention_mask"][:, context_length:] == 0
        model_logprobs_model_data = model_logprobs_model_data.masked_fill(
            model_padding_mask, 0.0
        )
        ref_logprobs_model_data = ref_logprobs_model_data.masked_fill(
            model_padding_mask, 0.0
        )

        return (model_logprobs_model_data, ref_logprobs_model_data)

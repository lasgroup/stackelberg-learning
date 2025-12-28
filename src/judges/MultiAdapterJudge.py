import torch
from trl.data_utils import maybe_apply_chat_template
from trl.models.utils import unwrap_model_for_generation
from trl import BasePairwiseJudge
from transformers import PreTrainedTokenizer
from peft.peft_model import PeftModelForSequenceClassification
from accelerate import Accelerator


class MultiAdapterJudge(BasePairwiseJudge):
    def __init__(
        self,
        model: PeftModelForSequenceClassification,
        tokenizer: PreTrainedTokenizer,
        missing_eos_penalty: float = None,
    ):
        super().__init__()
        assert isinstance(
            model, PeftModelForSequenceClassification
        ), "Model must be a PeftModelForSequenceClassification"
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_names = list(self.model.peft_config.keys())
        self.accelerator = None
        self.missing_eos_penalty = missing_eos_penalty

    def add_accelerator(self, accelerator: Accelerator):
        """Add accelerator to the judge."""
        self.accelerator = accelerator
        self.model = self.model.to(accelerator.device)

    def judge(
        self,
        prompts,
        completions,
        contain_eos_tokens=None,
        shuffle_order=False,
        return_scores=False,
    ):
        """
        Judge which completion is better by sampling a random model and using its scores.

        Args:
            prompts: List of prompt strings
            completions: List of pairs of completion strings
            shuffle_order: Whether to shuffle order (unused in this implementation, approach is invariant to shuffling)
            return_scores: Whether to return scores instead of scores. (Not implemented yet.)

        Returns:
            List of 0s and 1s, where 1 means first completion is preferred
        """
        assert (
            self.accelerator is not None
        ), "Accelerator not set. Call add_accelerator() first."
        if contain_eos_tokens is not None:
            assert (
                self.missing_eos_penalty is not None
            ), "Missing eos token penalty not set. Set it in the constructor."

        # TODO: Add batched inference
        if return_scores:
            results = []
            with unwrap_model_for_generation(
                self.model, self.accelerator
            ) as unwrapped_model:
                for adapter_name in self.adapter_names:
                    unwrapped_model.set_adapter(adapter_name)
                    adapter_results = []
                    for i, (prompt, completion_pair) in enumerate(
                        zip(prompts, completions)
                    ):
                        score_0 = self._get_score(
                            {"prompt": prompt, "completion": completion_pair[0]},
                            unwrapped_model,
                        )
                        score_1 = self._get_score(
                            {"prompt": prompt, "completion": completion_pair[1]},
                            unwrapped_model,
                        )
                        if contain_eos_tokens is not None:
                            score_0 = (
                                score_0
                                - (~contain_eos_tokens[i][0]) * self.missing_eos_penalty
                            )
                            score_1 = (
                                score_1
                                - (~contain_eos_tokens[i][1]) * self.missing_eos_penalty
                            )
                        adapter_results.append(1 if score_0 >= score_1 else 0)
                    results.append(adapter_results)
            torch.cuda.empty_cache()

            # Average the results across adapters
            results = torch.tensor(results).float()
            results = results.mean(dim=0).tolist()
            return results
        else:
            results = []
            for i, (prompt, completion_pair) in enumerate(zip(prompts, completions)):
                random_idx = torch.randint(len(self.adapter_names), (1,)).item()
                adapter_name = self.adapter_names[random_idx]
                with unwrap_model_for_generation(
                    self.model, self.accelerator
                ) as unwrapped_model:
                    unwrapped_model.set_adapter(adapter_name)
                    score_0 = self._get_score(
                        {"prompt": prompt, "completion": completion_pair[0]},
                        unwrapped_model,
                    )
                    score_1 = self._get_score(
                        {"prompt": prompt, "completion": completion_pair[1]},
                        unwrapped_model,
                    )
                    if contain_eos_tokens is not None:
                        score_0 = (
                            score_0
                            - (~contain_eos_tokens[i][0]) * self.missing_eos_penalty
                        )
                        score_1 = (
                            score_1
                            - (~contain_eos_tokens[i][1]) * self.missing_eos_penalty
                        )
                    results.append(1 if score_0 >= score_1 else 0)
            torch.cuda.empty_cache()
            return results

    def _get_score(self, element, model):
        """Get score from a model for a given text.
        Override this method based on your specific model architecture.
        """
        # Example implementation for reward models
        text = maybe_apply_chat_template(element, self.tokenizer)
        text = text["prompt"] + text["completion"]
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        model.eval()
        with torch.no_grad():
            output = model(**inputs)
        del inputs
        return output.logits.item()

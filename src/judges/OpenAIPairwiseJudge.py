import numpy as np
from trl import BasePairwiseJudge
from typing import Optional
from openai import OpenAI
import concurrent.futures
import logging

DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "A",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "B",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''

SKYWORKS_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better without quotation marks, no spaces, no new lines, and nothing else.

[User Question]
{prompt}

[The Start of Assistant A's Answer]
{response0}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response1}
[The End of Assistant B's Answer]
"""


class OpenAIPairwiseJudge(BasePairwiseJudge):
    """
    ...
    """

    def __init__(
        self,
        openai_api_base: str = "http://127.0.0.1:8000/v1",
        openai_api_key: str = "EMPTY",
        system_instructions: Optional[str] = None,
        temperature: float = 1.0,
        system_prompt=DEFAULT_PAIRWISE_SYSTEM_PROMPT,
        *args,
        **kwargs,
    ):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        models = self.client.models.list()
        self.model = models.data[0].id
        print("Judge model used for the OpenAI client: ", self.model)
        self.system_prompt = system_prompt
        if "Skywork-Critic-Llama-3.1-70B" in self.model:
            self.system_prompt = SKYWORKS_SYSTEM_PROMPT
            print("Overwriting OpenAI client system prompt for the judge to SKYWORKS_SYSTEM_PROMPT")
        self.system_instructions = system_instructions
        self.temperature = temperature

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        *args,
        **kwargs,
    ) -> list[int]:
        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(
                prompt=prompt, response0=candidates[0], response1=candidates[1]
            )
            input_chat = [{"role": "user", "content": content}]
            if self.system_instructions:
                input_chat = [
                    {"role": "system", "content": self.system_instructions}
                ] + input_chat
            try:
                response = self.client.chat.completions.create(
                    messages=input_chat,
                    model=self.model,
                    stream=False,
                    max_tokens=1,  # Just need a few tokens for demonstration
                    logprobs=True,  # Request log probabilities
                    top_logprobs=20,  # Maximum allowed
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    } if "Qwen3" in self.model else None,
                )
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                model_identifiers = ["A", "B"]
                completion_logprobs = -np.inf * np.ones(len(model_identifiers))
                for item in logprobs:
                    if item.token in model_identifiers:
                        idx = model_identifiers.index(item.token)
                        completion_logprobs[idx] = item.logprob
                if np.any(np.isinf(completion_logprobs)):
                    logging.error("Warning: Inf detected in logprobs")
                completion_logprobs = np.exp(completion_logprobs/self.temperature)
                prob = completion_logprobs[0] / (sum(completion_logprobs) + 1.0e-6)
            except Exception as e:
                logging.error("Error during API call: %s", e)
                prob = 0.5
            return prob

        # # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [
                ranks[i] if not flip else 1 - ranks[i]
                for i, flip in enumerate(flip_mask)
            ]

        # Return the ranks
        return ranks

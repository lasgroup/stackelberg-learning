from argparse import ArgumentParser
from ifeval import Evaluator, InputExample, instruction_registry, read_input_examples, read_responses
import datasets
import json
import urllib.request

def add_arguments(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--is_ood",
        action="store_true",
        help="Whether to evaluate on the OOD version of IFEval",
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # Load IFEval dataset
    if args.is_ood:
        ifeval_input_data = datasets.load_dataset("valpy/ifeval_ood3")["train"]
        ifeval_input_data = read_input_examples(ifeval_input_data.to_list())
        print("OOD IFEval dataset loaded!")
    else:
        url = "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"
        ifeval_input_data = []
        with urllib.request.urlopen(url) as f:
            for line in f:
                ifeval_input_data.append(json.loads(line))   # Parse directly into Python dict/list
        ifeval_input_data = read_input_examples(ifeval_input_data)
        print("IFEval dataset loaded!")
    print(len(ifeval_input_data))
    print(ifeval_input_data[0])

    # Load generated responses
    responses = datasets.load_from_disk(args.dataset_path)["validation"]

    # Initialize evaluator
    evaluator = Evaluator(instruction_registry)

    # Standard strict/loose evaluation
    for completion_idx in range(len(responses[0]["completions"])):
        print(f"\n --- Evaluating completion #{completion_idx} ---")
        responses_single_completion = responses.map(lambda x: {
            "prompt": x["prompt"][0]["content"],
            "response": x["completions"][completion_idx][0]["content"]
        }).remove_columns(["prompt_id", "completions"]).to_list()
        print("First response: ", responses_single_completion[0])
        responses_single_completion = {x["prompt"]: x["response"] for x in responses_single_completion}
        report, all_outputs = evaluator.evaluate(ifeval_input_data, responses_single_completion)
        print("\n--- Strict prompt accuracy:", report["eval_results_strict"]["prompt_accuracy"])
        print("--- Loose prompt accuracy:", report["eval_results_loose"]["prompt_accuracy"])

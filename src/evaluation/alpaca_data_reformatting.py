import datasets
from argparse import ArgumentParser
import json

def add_arguments(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--generator",
        type=str,
        required=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # Load Dataset
    completions = datasets.load_from_disk(args.dataset_path)

    # Sample completion print
    idx = 0
    print(f"\nPrompt #{idx}: ", completions["validation"][idx]["prompt"])
    print("\n--- Leader completion: ---\n", completions["validation"][idx]["completions"][0])
    print("\n--- Follower completion: ---\n", completions["validation"][idx]["completions"][1])

    for i in range(len(completions["validation"][idx]["completions"])):
        output_path = f"{args.dataset_path}_completion_{i}.json"
        with open(output_path, "w") as f:
            json.dump(
                completions["validation"]
                .map(
                    lambda sample: {
                        "instruction": (
                            sample["prompt"]
                            if isinstance(sample["prompt"], str)
                            else sample["prompt"][0]["content"]
                        ),
                        "output": (
                            sample["completions"][i]
                            if isinstance(sample["completions"][i], str)
                            else sample["completions"][i][0]["content"]
                        ),
                        "generator": f"{args.generator}_completion_{i}",
                        "dataset": "alpaca_eval",
                        "datasplit": "eval",
                    }
                )
                .remove_columns(["prompt_id", "prompt", "completions"])
                .to_list(),
                f,
            )
        print("Saved at: ", output_path)

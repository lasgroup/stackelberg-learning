from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapter into base model")
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        required=True,
        help="The identifier of the PEFT model to merge",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
        help="The identifier of the base model to merge with the PEFT model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to save the merged model",
    )
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    print(f"Loaded base model from {args.base_model_name_or_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_name_or_path)
    print(f"Loaded PEFT model from {args.peft_model_name_or_path}")
    merged_model = model.merge_and_unload()
    print(f"Merged PEFT model into base model")
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Saved merged model to {args.output_path}")


if __name__ == "__main__":
    main()
    print("Merge completed successfully!")

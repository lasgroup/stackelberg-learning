import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, BitsAndBytesConfig
from trl import ModelConfig
from dataclasses import dataclass, field

"""
Example usage:
python src/preprocessing/model_quantization.py \
    --model_name_or_path model_name_or_path_on_huggingface \
    --huggingface_username your_hf_username
"""

@dataclass
class QuantizationConfig:
    huggingface_username: str = field(
        metadata={"help": "Huggingface username to push the model to the hub."}
    )

if __name__ == "__main__":
    parser = HfArgumentParser(ModelConfig, QuantizationConfig)
    model_args, hf_args = parser.parse_args_into_dataclasses()

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    torch.set_default_dtype(torch_dtype)
    quantization_config = quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=model_args.torch_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage="uint8",
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=None,
        quantization_config=quantization_config,
        use_cache=False,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    huggingface_hub_name=f"{hf_args.huggingface_username}/{model_args.model_name_or_path.split('/')[-1]}-4bit-bnb"
    tokenizer.push_to_hub(huggingface_hub_name)
    model.push_to_hub(huggingface_hub_name)

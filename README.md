# Stackelberg Learning from Human Feedback (SLHF)
[![arXiv](https://img.shields.io/badge/arXiv-2512.16626-b31b1b.svg)](https://arxiv.org/abs/2512.16626)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/pasztorb/Llama-3.1-Tulu-3-8B-SLHF)

This repository contains the code to reproduce the experiments in the paper
"Stackelberg Learning from Human Feedback: Preference Optimization as a Sequential Game" by Barna Pásztor, Thomas Kleine Buening, and Andreas Krause.

# Setup
Create a new virtual environment with Python version `3.12.3` and install the required packages in `requirements.txt`.
```commandline
python3 -m venv env
pip install --upgrade pip
pip install -r requirements.txt
```
If you are using cuda for training, make sure to install the `torch` version compatible with your CUDA version.

If you are planning to evaluate models on the [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) or [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#alpacaeval-20) benchmarks, install the respective packages from the linked repositories and follow their instructions.

Training runs are logged to Weights and Biases. To seamlessly log your experiments to your account,
save your api-key to `${HOME}/.wandb-api-key` file. You can find your api-key in your W&B account settings.

[//]: # (### Setup Euler)

[//]: # (Use the following modules and commands on Euler)

[//]: # (```commandline)

[//]: # (module load stack/2024-06 gcc/12.2.0)

[//]: # (module load python_cuda/3.11.6 cudnn/9.2.0)

[//]: # (module load eth_proxy)

[//]: # (export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME)

[//]: # (```)

[//]: # (and install the packages in `requirements.txt` in a virtualenv.)

# Training

### Dataset preparation
To run experiments on the `HelpSteer2` dataset using the custom judge aggregating separate reward models, follow these steps:
1. Preprocess the dataset and save the train-validation split using the following snippet:
```python
from src.preprocessing.helpsteer2 import load_dataset as helpsteer2_load_dataset
dataset = helpsteer2_load_dataset(seed=42, train_validation_split=0.8)
dataset.save_to_disk("path/to/preprocessed_helpsteer2_dataset")
```

2. Train the separate reward models for each attribute using the following script:
```bash
bash scripts/train_reward_model.sh attribute_name
```
for `attribute_name` in `helpfulness`, `correctness`, `coherence`, `complexity`, `verbosity`.
Before training, add the correct paths to the `scripts/train_reward_model.sh` script
and update the parameters according to the available resources.
We recommend setting the `output_dir` variable such that it includes the attribute name, e.g.,
`path/to/reward_model/attribute_name`.
All fine-tuning and evaluation scripts expects the `reward_model_adapters_path` argument to be set as follows
`path/to/reward_model/(attribute1|attribute2|attribute3|...)` where the attributes are separated by `|`.

# Training
## HelpSteer2 Fine-Tuning
### Training
The training scripts for `RLOO`, `Nash-MD`, and `StackelbergGDA` are located in the `scripts` folder named `train_{algorithm}.sh`.
Before executing the scripts, make sure to update the paths and parameters.
By default, the results are saved to `data/experiments/${run_name}`.
Execute each script from the root directory of the repository.

### Evaluation
To evaluate any given model, update the script `scripts/evaluation.sh` with the right models and datasets.
Generated responses are saved to `path/to/experiment/generation__checkpoint-x` and corresponding rewards for each
attribute are saved to `path/to/experiment/generation__checkpoint-x__rewards`.

You can create correction evaluations across multiple models by running the `correction_evaluation.sh` script.

## General Fine-Tuning
We provide a more generic purpose training script for fine-tuning models for chat applications at `scripts/train_slhf_chat.sh`.
Configs are set to reproduce results in Section 6.2 of the paper.

### Training Acceleration
The script is compatible with distributed training integrations in the TRL library.
For more information, refer to the [TRL documentation](https://huggingface.co/docs/trl/main/en/distributing_training).
Furthermore, we provide support for vLLM inference for both the judge and the policy model.
1. To setup vLLM for the judge model, start a vLLM server with the `vllm serve` command and set the `--judge` argument of the training script to `"localhost-{JUDGE_SERVER_NODE}:{JUDGE_PORT}"`.
where `JUDGE_SERVER_NODE` and `JUDGE_PORT` correspond to the node and port where the vLLM server is running.
If you run the judge on the same node as the training script you can set `JUDGE_SERVER_NODE` to `localhost`.
2. To setup vLLM for the policy model, first start a vLLM server using the `trl vllm serve` command.
For more information, see the official TRL documentation on [vLLM integration](https://huggingface.co/docs/trl/main/en/vllm_integration).
Then set the following arguments `--use_vllm server --vllm_server_host {POLICY_SERVER_NODE} --vllm_server_port {POLICY_PORT}`.
where `POLICY_SERVER_NODE` and `POLICY_PORT` correspond to the node and port where the vLLM server is running.

### Using new datasets
To use a new dataset for training, the dataset must be included in the `load_dataset` function in the `src.utils.py` file.
For the `StackelbergGDA` training, the returned datasets must include `prompt_id` and `prompt` columns where the `prompt` column includes a list of dictionaries representing chat dialog turns as standard in the TRL library.
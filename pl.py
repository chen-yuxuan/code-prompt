import argparse
import logging

from huggingface_hub import login

from code_prompt.experiments import run_hcc, run_sst, run_semeval, run_mrpc
from code_prompt.utils import seed_everything
from code_prompt.settings import HF_TOKEN

MODELS = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "google/codegemma-7b",
    "meta-llama/CodeLlama-13b-hf",
]

ALL_MODELS = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "google/codegemma-7b",
    "meta-llama/CodeLlama-13b-hf",
    "gpt-3.5-turbo-instruct",
    "deepseek-v3",
]

parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting 3 programming languages with 4 datasets."
)
parser.add_argument("--seed", type=int, default=42, help="The random seed.")
parser.add_argument(
    "--shots", type=int, default=4, help="Number of few-shot examples. 0 for zero-shot."
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="The model name/identifier.",
)
parser.add_argument(
    "--enforce_code_prompt",
    action="store_true",
    help="Enforce code prompt for all models.",
)
parser.add_argument(
    "--type_hint",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Include type hints (default: True)",
)
parser.add_argument(
    "--implement",
    type=lambda x: x.lower() == "true",
    default=False,
    help="Implement the function body (default: False)",
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logging.info(f"Arguments: {args}")
login(token=HF_TOKEN)

seed_everything(args.seed)
models = MODELS if args.model is None else [args.model]
runs = 3 if args.shots > 0 else 1
shots = 16 if args.shots > 16 else args.shots

for model in models:
    logging.info(f"Running HCCexperiment with model {model}")
    # try and if fails, print error and continue
    for run in range(runs):
        logging.info(f"Run {run+1}/{runs} for model {model}")
        try:
            run_hcc(
                model,
                enforce_code_prompt=args.enforce_code_prompt,
                shots=shots,
                seed=run,
                type_hint=args.type_hint,
                implement=args.implement,
            )
        except Exception as e:
            logging.error(f"HCC-Experiment with model {model} failed with error: {e}")
            continue
    logging.info(f"HCC Experiment with model {model} completed successfully.")

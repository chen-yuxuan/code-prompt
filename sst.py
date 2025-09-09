import argparse
import logging

from code_prompt.experiments.sst import run_sst
from code_prompt.utils import seed_everything

MODELS = [
    "Qwen/Qwen2.5-Coder-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-Coder-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-3B",
    "Qwen/Qwen2.5-3B-Instruct",
]

ALL_MODELS = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen2.5-Coder-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-Coder-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    "deepseek-ai/DeepSeek-V2-Lite",
    "google/codegemma-7b",
    "google/gemma-7b",
    "google/codegemma-2b",
    "google/gemma-2b",
    "meta-llama/CodeLlama-13b-hf",
    "meta-llama/CodeLlama-13b-Python-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/CodeLlama-7b-hf",
    "meta-llama/CodeLlama-7b-Python-hf",
    "meta-llama/Llama-2-7b-hf",
]

parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting with SST-2 dataset."
)
parser.add_argument("--seed", type=int, default=42, help="The random seed.")
parser.add_argument(
    "--shots", type=int, default=0, help="Number of few-shot examples. 0 for zero-shot."
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="The model name/identifier.",
)
parser.add_argument(
    "--enforce_code_prompt",
    type=bool,
    default=False,
    help="Whether to enforce code prompt for all models.",
)
parser.add_argument(
    "--type_hint",
    type=bool,
    default=True,
    help="Whether to include type hints in the code prompt.",
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

seed_everything(args.seed)
models = MODELS if args.model is None else [args.model]
for model in models:
    logging.info(f"Running experiment with model {model}")
    # try and if fails, print error and continue
    try:
        run_sst(
            model,
            enforce_code_prompt=args.enforce_code_prompt,
            shots=args.shots,
            seed=args.seed,
            type_hint=args.type_hint,
        )
    except Exception as e:
        logging.error(f"Experiment with model {model} failed with error: {e}")
        continue
    logging.info(f"Experiment with model {model} completed successfully.")

# another round only for coder models with no type hint
for model in models:
    if "code" in model.lower():
        logging.info(f"Running experiment with model {model} without type hint")
        # try and if fails, print error and continue
        try:
            run_sst(
                model,
                enforce_code_prompt=False,
                shots=args.shots,
                seed=args.seed,
                type_hint=False,
            )
        except Exception as e:
            logging.error(
                f"Experiment with model {model} without type hint failed with error: {e}"
            )
            continue
        logging.info(
            f"Experiment with model {model} without type hint completed successfully."
        )

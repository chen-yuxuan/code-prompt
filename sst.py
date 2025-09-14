import argparse
import logging

from code_prompt.experiments.sst import run_sst
from code_prompt.utils import seed_everything, clean_vllm_memory

MODELS = [
    # >=30B then all <30B including deepseek models
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen2.5-Coder-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-Coder-14B",
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
    "gpt-3.5-turbo-instruct",
    "deepseek-v3",
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
    action="store_true",
    help="Enforce code prompt for all models.",
)
parser.add_argument(
    "--type_hint",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Include type hints (default: True)",
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logging.info(f"Arguments: {args}")

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

# for code models, rerun without type hints
for model in models:
    if "code" in model.lower() and args.type_hint:
        logging.info(f"Rerunning experiment with model {model} without type hints")
        try:
            run_sst(
                model,
                shots=args.shots,
                seed=args.seed,
                type_hint=False,
            )
        except Exception as e:
            logging.error(f"Experiment with model {model} failed with error: {e}")
            continue
        logging.info(f"Experiment with model {model} completed successfully.")

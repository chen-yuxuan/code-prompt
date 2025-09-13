import argparse
import logging

from code_prompt.experiments.xnli import run_xnli
from code_prompt.utils import seed_everything

MODELS = [

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
    "gpt3.5-turbo-instruct",
    "deepseek-v3",
]

parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting with XNLI dataset."
)
parser.add_argument(
    "--language",
    type=str,
    default="en",
    help="The language to evaluate on. Should be one of the languages in XNLI.",
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
        run_xnli(
            model,
            language=args.language,
            enforce_code_prompt=args.enforce_code_prompt,
            shots=args.shots,
            seed=args.seed,
            type_hint=args.type_hint,
        )
    except Exception as e:
        logging.error(f"Experiment with model {model} failed with error: {e}")
        continue
    logging.info(f"Experiment with model {model} completed successfully.")


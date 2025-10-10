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

LANGUAGES = ["ruby"]

parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting 4 programming languages with 4 datasets."
)
parser.add_argument("--seed", type=int, default=0, help="The random seed.")
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="The model name/identifier.",
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logging.info(f"Arguments: {args}")
login(token=HF_TOKEN)

seed_everything(args.seed)
models = MODELS if args.model is None else [args.model]

for model in models:
    for language in LANGUAGES:
        for experiment in [run_hcc, run_sst, run_semeval, run_mrpc]:
            try:
                logging.info(
                    f"Running {experiment.__name__} experiment with model {model} and language {language}"
                )
                experiment(
                    model_name=model,
                    shots=0,
                    seed=args.seed,
                    enforce_code_prompt=True,
                    type_hint=True,
                    language=language,
                )
            except Exception as e:
                logging.error(
                    f"{experiment.__name__}-Experiment with model {model} and language {language} failed with error: {e}"
                )
        logging.info(
            f"Finished all experiments for model {model} and language {language}"
        )
    logging.info(f"Finished all experiments for model {model}")

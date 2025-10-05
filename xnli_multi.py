import argparse
import logging

from huggingface_hub import login

from code_prompt.experiments.xnli_multi import run_xnli_multi
from code_prompt.utils import seed_everything
from code_prompt.settings import HF_TOKEN


MODELS = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "google/codegemma-7b",
    "meta-llama/CodeLlama-13b-hf",
    # "gpt-3.5-turbo-instruct",
    # "deepseek-v3",
]
LANGUAGES = [
    "ar",
    "bg",
    "de",
    "el",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting with XNLI-Multi dataset."
)
parser.add_argument(
    "--lang",
    type=str,
    default=None,
    help="The language to evaluate on. Should be one of the languages in XNLI.",
)
parser.add_argument("--seed", type=int, default=0, help="The random seed.")
parser.add_argument(
    "--shots", type=int, default=0, help="Number of few-shot examples. 0 for zero-shot."
)
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
languages = LANGUAGES if args.lang is None else [args.lang]
shots = 4 if args.shots > 4 else args.shots
for model in models:
    for lang in languages:
        logging.info(f"Running experiment with model {model} on language {lang}")
        try:
            run_xnli_multi(
                model,
                lang=lang,
                shots=shots,
                seed=0,
            )
            logging.info(
                f"Experiment with model {model} on language {lang} completed successfully."
            )
        except Exception as e:
            logging.error(
                f"Experiment with model {model} on language {lang} failed with error: {e}"
            )
            continue
    logging.info(f"Experiment with model {model} completed.")

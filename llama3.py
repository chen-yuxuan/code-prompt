import argparse
import logging

from huggingface_hub import login

from code_prompt.experiments import (
    run_agnews,
    run_cola,
    run_iris,
    run_mrpc,
    run_mscinli,
    run_scierc,
    run_semeval,
    run_sst,
    run_xnli,
)
from code_prompt.utils import seed_everything
from code_prompt.settings import HF_TOKEN

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting with SciERC dataset."
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
models = MODEL if args.model is None else args.model
runs = 3 if args.shots > 0 else 1

# run each dataset
for experiment in [
    run_agnews,
    run_cola,
    run_iris,
    run_mrpc,
    run_mscinli,
    run_semeval,
    run_sst,
    run_xnli,
]:
    logging.info(f"Running experiment {experiment.__name__}")
    for model in models:
        logging.info(f"Running {experiment.__name__} with model {model}")
        # try and if fails, print error and continue
        for run in range(runs):
            logging.info(f"Run {run+1}/{runs} with {args.shots} shots")
            try:
                experiment(
                    model,
                    shots=args.shots,
                    seed=run,
                )
            except Exception as e:
                logging.error(f"Experiment with model {model} failed with error: {e}")
                continue
        logging.info(f"Experiment with model {model} completed successfully.")


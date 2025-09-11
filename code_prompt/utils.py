import contextlib
import gc
import random

import torch
import vllm
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)


def seed_everything(seed: int) -> None:
    """Sets random seed anywhere randomness is involved.

    This process makes sure all the randomness-involved operations yield the
    same result under the same `seed`, so each experiment is reproducible.
    In this function, we set the same random seed for the following modules:
    `random`, `numpy` and `torch`.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_vllm_memory(model):
    """Cleans up the GPU memory used by a vLLM model."""
    if isinstance(model, vllm.LLM):
        destroy_model_parallel()
        destroy_distributed_environment()
        # del model.llm_engine.model_executor
        del model
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
    elif model:
        del model

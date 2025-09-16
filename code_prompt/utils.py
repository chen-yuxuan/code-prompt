import contextlib
from collections import defaultdict
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
    """Safely clean GPU memory from vLLM or HF models."""
    try:
        if isinstance(model, vllm.LLM):
            # Explicitly shutdown vLLM engine
            with contextlib.suppress(Exception):
                model.llm_engine.shutdown()
        del model
    except Exception as e:
        print(f"[Warning] clean_vllm_memory: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
    return None


def few_shot_per_class(dataset, k=1, seed=0, label_column_name="label"):
    rng = random.Random(seed)
    # group indices by label
    indices_per_class = defaultdict(list)
    for i, ex in enumerate(dataset):
        indices_per_class[ex[label_column_name]].append(i)

    sampled_indices = []
    for label, indices in indices_per_class.items():
        if len(indices) < k:
            raise ValueError(
                f"Class {label} has only {len(indices)} examples, needs {k}"
            )
        sampled_indices.extend(rng.sample(indices, k=k))

    return dataset.select(sampled_indices)

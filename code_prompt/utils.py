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


def adjust_code_prompt_template(orig_prompt: str, model_name: str) -> str:
    """Adjusts the code prompt template based on the model name.
    `orig_prompt` is to prompt for future code on the right.
    For Code-Llama models, keep the original prompt.
    For DeepSeek-Coder models, use |fim_begin|, |fim_hole|, |fim_end| tokens.
    For CodeGemma models, use <fim_prefix>, <fim_middle>, <fim_suffix> tokens.
    """
    if "codellama" in model_name.lower() or "deepseek-coder" in model_name.lower():
        return orig_prompt
    elif "codegemma" in model_name.lower():
        return orig_prompt.replace("def ", "<fim_prefix>def ").replace(
            "pass", "<fim_middle>pass<fim_suffix>"
        )
    else:
        return orig_prompt


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

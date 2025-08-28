"Wrap different code generation and parsing functions"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GemmaTokenizer,
)
import torch


def code_complete_gemma(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: GemmaTokenizer,
    max_new_tokens: int = 32,
) -> str:
    """Code completion for Gemma models."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0][prompt_len:])
    return response


def code_complete_qwen(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 32,
) -> str:
    """Code completion for Qwen models."""
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    # Use `max_new_tokens` to control the maximum output length.
    eos_token_ids = [151659, 151661, 151662, 151663, 151664, 151643, 151645]
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_ids,
        )
    # The generated_ids include prompt_ids, we only need to decode the tokens after prompt_ids.
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return response


def code_complete(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 32,
) -> str:
    """Code completion for DeepSeek models."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return response

import json
import logging

from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GemmaTokenizer,
)
import torch

from ..prompts.xnli_multi import get_code_prompt
from ..models.codellm import (
    code_complete_gemma,
    code_complete_qwen,
    code_complete,
)
from ..models.llm import get_client, get_response
from ..utils import clean_vllm_memory, few_shot_per_class


def run_xnli_multi(
    model_name: str = "Qwen/Qwen2.5-32B-Instruct",
    lang: str = "de",
    shots: int = 0,
    seed: int = 0,
):
    logging.basicConfig(level=logging.INFO)

    testset = load_dataset("facebook/xnli", lang, split="test")
    if shots > 0:
        trainset = load_dataset("facebook/xnli", lang, split="train")
        few_shot_examples = few_shot_per_class(
            trainset, k=shots, seed=seed, label_column_name="label"
        )
    else:
        few_shot_examples = None

    if "code" in model_name.lower():
        if "gemma" in model_name.lower():
            tokenizer = GemmaTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32, device_map="auto"
            ).eval()
        elif "qwen" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                device_map="auto",
            ).eval()
        else:  # deepseek and llama coder
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
    else:
        beta = True if "deepseek-v3" in model_name.lower() else False
        model = get_client(model_name, beta=beta)

    examples = []
    for example in tqdm(testset):
        if "code" in model_name.lower():
            prompt = get_code_prompt(
                example=example,
                few_shot_examples=few_shot_examples,
                type_hint=True,
                model=model_name,
                lang=lang,
            )
            if "gemma" in model_name.lower():
                response = code_complete_gemma(
                    prompt,
                    model=model,
                    tokenizer=tokenizer,
                )
            elif "qwen" in model_name.lower():
                if "instruct" in model_name.lower():
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a code completion assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                response = code_complete_qwen(
                    prompt,
                    model,
                    tokenizer,
                )
            else:  # deepseek and llama coder
                response = code_complete(
                    prompt,
                    model,
                    tokenizer,
                )

        else:
            prompt = get_code_prompt(
                example=example,
                few_shot_examples=few_shot_examples,
                type_hint=True,
                model=model_name,
                lang=lang,
            )
            response = get_response(
                prompt,
                client=model,
                model=model_name,
                enforce_code_prompt=True,
            )
        examples.append(
            {
                "premise": example["premise"],
                "hypothesis": example["hypothesis"],
                "natural_language": lang,
                "label": example["label"],
                "response": response,
            }
        )

    model = clean_vllm_memory(model)
    _model_name = model_name.split("/")[-1].replace(".", "")
    if shots == 0:
        output_path = f"./outputs/xnli_multi_{lang}_{_model_name}_{shots}_shot"
    else:
        output_path = (
            f"./outputs/xnli_multi_{lang}_{_model_name}_{shots}_shot_seed_{seed}"
        )
    output_path += ".json"
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=4, ensure_ascii=True)
    logging.info(f"XNLI-{lang} results saved to {output_path}")

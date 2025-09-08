import json
import logging

from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GemmaTokenizer,
)
import torch
import vllm

from ..prompts.sst import get_nl_prompt, get_code_prompt
from ..settings import HF_TOKEN
from ..models.codellm import (
    code_complete_gemma,
    code_complete_qwen,
    code_complete,
)
from ..models.llm import get_client, get_response


def run_sst(
    model_name: str = "Qwen/Qwen2.5-32B-Instruct",
    enforce_code_prompt: bool = False,
    shots: int = 0,
    seed: int = 42,
    type_hint: bool = True,
    data_path: str = None,
):
    logging.basicConfig(level=logging.INFO)
    login(token=HF_TOKEN)

    testset = load_dataset(data_path, split="validation")
    if shots > 0:
        trainset = load_dataset(data_path, split="train")
        few_shot_examples = trainset.shuffle(seed=seed).select(range(shots))
    else:
        few_shot_examples = None

    if "code" in model_name.lower():
        if "gemma" in model_name.lower():
            tokenizer = GemmaTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map="auto"
            ).eval()
        elif "qwen" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
            ).eval()
        else:  # deepseek and llama coder
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
    else:
        model = get_client(model_name)

    examples = []
    for example in tqdm(testset):
        text = example["sentence"]
        if "code" in model_name.lower():
            prompt = get_code_prompt(
                text=text,
                few_shot_examples=few_shot_examples,
                type_hint=type_hint,
                model=model_name,
            )
            if "gemma" in model_name.lower():
                response = code_complete_gemma(
                    prompt,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=2,
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
                    max_new_tokens=2,
                )
            else:  # deepseek and llama coder
                response = code_complete(
                    prompt,
                    model,
                    tokenizer,
                    max_new_tokens=2,
                )

        else:
            if enforce_code_prompt:
                prompt = get_code_prompt(
                    text=text,
                    few_shot_examples=few_shot_examples,
                    type_hint=type_hint,
                    model=model_name,
                )
                response = get_response(
                    prompt,
                    client=model,
                    model=model_name,
                    enforce_code_prompt=True,
                    max_tokens=1,
                )
            else:
                prompt = get_nl_prompt(text=text, few_shot_examples=few_shot_examples)
                response = get_response(
                    prompt,
                    client=model,
                    model=model_name,
                )
        examples.append(
            {
                "id": example["idx"],
                "text": text,
                "label": example["label"],
                "response": response,
            }
        )

    _model_name = model_name.split("/")[-1].replace(".", "")
    output_path = f"./outputs/sst2_{_model_name}_{shots}_shot"
    if enforce_code_prompt:
        output_path += "_codeprompt"
    if not type_hint:
        output_path += "-notype"
    output_path += ".json"
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=4, ensure_ascii=True)

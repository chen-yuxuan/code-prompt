import argparse
import json
import logging

from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer
import transformers
import torch
from vllm import LLM, SamplingParams

from code_prompt.prompts.sst import NL_PROMPT, CODE_PROMPT


parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting with SST-2 dataset."
)
parser.add_argument("--seed", type=int, default=42, help="The random seed.")
parser.add_argument(
    "--shots", type=int, default=0, help="Number of few-shot examples. 0 for zero-shot."
)
parser.add_argument(
    "--model",
    type=str,
    default="deepseek-ai/DeepSeek-V2-Lite",
    help="The model name/identifier.",
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


testset = load_dataset("stanfordnlp/sst2", split="validation")
if args.shots > 0:
    trainset = load_dataset("stanfordnlp/sst2", split="train")

HF_TOKEN = "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"
login(token=HF_TOKEN)

if "code" in args.model.lower():
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = transformers.pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    sampling_params = SamplingParams(temperature=0)
    model = LLM(model=args.model, trust_remote_code=True)

examples = []
for example in tqdm(testset):
    text = example["sentence"]
    if "code" in args.model.lower():
        prompt = CODE_PROMPT.format(text=text)
        response = model(prompt, eos_token_id=tokenizer.eos_token_id, max_length=200)[0]
    else:
        prompt = NL_PROMPT.format(text=text)
        response = model.generate(prompt, sampling_params)[0].outputs[0].text
    examples.append(
        {
            "id": example["idx"],
            "text": text,
            "label": example["label"],
            "nl_response": response,
        }
    )

model_name = args.model.split("/")[-1].replace(".", "")
output_path = f"sst2_{model_name}_{args.shots}_shot.json"
with open(output_path, "w") as f:
    json.dump(examples, f, indent=4, ensure_ascii=True)

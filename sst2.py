import argparse
import json
import logging

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from code_prompt.prompts.sst import NL_PROMPT, CODE_PROMPT
from code_prompt.models.lmstudio import get_response


parser = argparse.ArgumentParser(
    description="Collect arguments for experimenting with SST-2 dataset."
)
parser.add_argument(
    "--seed", type=int, default=42, help="The random seed."
)
parser.add_argument(
    "--shots", type=int, default=0, help="Number of few-shot examples. 0 for zero-shot."
)
parser.add_argument(
    "--model", type=str, default="Qwen/Qwen3-32B", help="The model name/identifier."
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


testset = load_dataset("stanfordnlp/sst2", split="validation")
if args.shots > 0:
    trainset = load_dataset("stanfordnlp/sst2", split="train")
sampling_params = SamplingParams(temperature=0)
llm = LLM(model="Qwen/Qwen3-32B")

examples = []
for example in tqdm(testset):
    text = example["sentence"]
    nl_prompt = NL_PROMPT.format(text=text)
    response = llm.generate(nl_prompt, sampling_params)[0].outputs[0].text
    examples.append(
        {
            "id": example["idx"],
            "text": text,
            "label": example["label"],
            "nl_response": response,
        }
    )

output_path = f"sst2_{args.model.split("/")[-1]}_{args.shots}_shot.json"
with open(output_path, "w") as f:
    json.dump(examples, f, indent=4, ensure_ascii=True)

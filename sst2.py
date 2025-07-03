import json

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from code_prompt.prompts.sst import NL_PROMPT, CODE_PROMPT
from code_prompt.models.lmstudio import get_response


ds = load_dataset("stanfordnlp/sst2", split="validation")
sampling_params = SamplingParams(temperature=0)
llm = LLM(model="Qwen/Qwen3-32B")

examples = []
for example in tqdm(ds):
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
with open("sst2_examples.json", "w") as f:
    json.dump(examples, f, indent=4, ensure_ascii=True)
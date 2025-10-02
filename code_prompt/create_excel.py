import json
import pandas as pd

# define order of models to present in the excel file
MODELS = [
    "Qwen3-Coder-30B",
    "Qwen3-30B-A3B-Instruct",
    "Qwen25-Coder-32B",
    "Qwen25-32B-Instruct",
    "Qwen25-Coder-14B",
    "Qwen25-14B-Instruct",
    "Qwen25-Coder-7B",
    "Qwen25-7B-Instruct",
    "Qwen25-Coder-3B",
    "Qwen25-3B-Instruct",
    "deepseek-v3_0_shot_codeprompt",
    "deepseek-v3_0_shot.",
    "codegemma-7b",
    "_gemma-7b",
    "codegemma-2b",
    "_gemma-2b",
    "CodeLlama-13b-hf",
    "CodeLlama-13b-Python",
    "_Llama-2-13b",
    "CodeLlama-7b-hf",
    "CodeLlama-7b-Python",
    "_Llama-2-7b",
    "gpt-35-turbo-instruct_0_shot_codeprompt",
    "gpt-35-turbo-instruct_0_shot."
]


def create_excel(
    dataset: str="agnews",
    score: str="accuracy",
):
    read_path = f"eval/zeroshot-{dataset}-evaluation.json"
    # load json
    with open(read_path, "r") as f:
        data = json.load(f)
    
    results: list[dict[str, float]] = []
    # if a model is not found, return -1 score
    for model in MODELS:
        for item in data:
            if model in item["filename"]:               
                results.append({
                    "model": model,
                    score: round(item[score] * 100, 1)
                })
                break
    # save as excel
    df = pd.DataFrame(results)
    df.to_excel(f"eval/zeroshot-{dataset}-{score}.xlsx", index=False)
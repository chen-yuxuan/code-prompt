import json
import pandas as pd

# define order of models to present in the excel file
MODELS = [
    "qwen3-coder-30b-a3b-instruct",
    "qwen3-30b-a3b-instruct-2507",
    "qwen25-coder-32b",
    "qwen25-32b-instruct",
    "qwen25-coder-14b",
    "qwen25-14b-instruct",
    "qwen25-coder-7b",
    "qwen25-7b-instruct",
    "qwen25-coder-3b",
    "qwen25-3b-instruct",
    "deepseek-v3",
    "deepseek-v3_coder",
    "codegemma-7b",
    "gemma-7b",
    "codegemma-2b",
    "gemma-2b",
    "codellama-13b-hf",
    "codellama-13b-python-hf",
    "llama-2-13b-hf",
    "codellama-7b-hf",
    "codellama-7b-python-hf",
    "llama-2-7b-hf",
    "gpt-35-turbo-instruct_coder",
    "gpt-35-turbo-instruct",
]


def create_excel(
    dataset: str = "agnews",
    score: str = "accuracy",
    scenario: str = "zeroshot",
):
    read_path = f"eval/{scenario}-{dataset}-evaluation.json"
    # load json
    with open(read_path, "r") as f:
        data = json.load(f)

    results: list[dict[str, float]] = []
    # if a model is not found, return -1 score
    for model in MODELS:
        for item in data:
            if model == item["model"]:
                results.append({"model": model, score: round(item[score] * 100, 1)})
                break
    # save as excel
    df = pd.DataFrame(results)
    df.to_excel(f"eval/{scenario}-{dataset}-{score}.xlsx", index=False)

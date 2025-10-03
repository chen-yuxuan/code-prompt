import json
import os

from sklearn.metrics import matthews_corrcoef, f1_score


ALL_LABELS = {
    "sst": {
        0: "negative",
        1: "positive",
    },
    "agnews": {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech",
    },
    "cola": {
        0: "unacceptable",
        1: "acceptable",
    },
    "semeval": {
        "Cause-Effect(e1,e2)": "Cause-Effect",
        "Cause-Effect(e2,e1)": "Cause-Effect",
        "Component-Whole(e1,e2)": "Component-Whole",
        "Component-Whole(e2,e1)": "Component-Whole",
        "Content-Container(e1,e2)": "Content-Container",
        "Content-Container(e2,e1)": "Content-Container",
        "Entity-Destination(e1,e2)": "Entity-Destination",
        "Entity-Destination(e2,e1)": "Entity-Destination",
        "Entity-Origin(e1,e2)": "Entity-Origin",
        "Entity-Origin(e2,e1)": "Entity-Origin",
        "Instrument-Agency(e1,e2)": "Instrument-Agency",
        "Instrument-Agency(e2,e1)": "Instrument-Agency",
        "Member-Collection(e1,e2)": "Member-Collection",
        "Member-Collection(e2,e1)": "Member-Collection",
        "Message-Topic(e1,e2)": "Message-Topic",
        "Message-Topic(e2,e1)": "Message-Topic",
        "Product-Producer(e1,e2)": "Product-Producer",
        "Product-Producer(e2,e1)": "Product-Producer",
        "Other": "Other",
    },
    "scierc": {
        "COMPARE": "compare",
        "CONJUNCTION": "conjunction",
        "EVALUATE-FOR": "evaluate-for",
        "FEATURE-OF": "feature-of",
        "HYPONYM-OF": "hyponym-of",
        "PART-OF": "part-of",
        "USED-FOR": "used-for",
    },
    "xnli": {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    },
    "iris": {
        "Iris-setosa": "setosa",
        "Iris-versicolor": "versicolor",
        "Iris-virginica": "virginica",
    },
    "mscinli": {
        "neutral": "neutral",
        "entailment": "entailment",
        "contrasting": "contrasting",
        "reasoning": "reasoning",
    },
    "mrpc": {
        0: "not_equivalent",
        1: "equivalent",
    },
    "hcc": {
        True: "True",
        False: "False",
    },
}

ALL_LABEL_COLUMN_NAMES = {
    "sst": "label",
    "agnews": "label",
    "cola": "label",
    "semeval": "relation",
    "scierc": "label",
    "iris": "Species",
    "xnli": "label",
    "mscinli": "label",
    "mrpc": "label",
    "hcc": "label",
}



def response_to_label(
    response: str, target_classes: list[str | int]
) -> str | int | None:
    """Find the earliest occurrence of any target class in the response and return it.
    If none found, return None.
    Example:
        response = "Positive. Because no negative words found."
        target_classes = ["positive", "negative"]
        Returns: "positive"
    """
    response = response.lower()
    first_match = None
    first_pos = len(response) + 1  # start larger than any possible index

    for target in target_classes:
        target_str = str(target).lower()
        pos = response.find(target_str)
        if pos != -1 and pos < first_pos:
            first_match = target
            first_pos = pos

    return first_match


def redundancy_score(response: str, pred: str | int) -> float | None:
    """Calculate the redundancy score of the prediction in the response.
    Defined as REDUNDANCY := 1 - len(pred_label)/len(response)
    """
    # clean response
    response = (
        response.lower().strip().replace("\n", " ").replace('"', "").replace("'", "")
    )
    # special tokens from codegemma
    response = response.replace("<eos>", "").replace("<|file_separator|>", "")
    # markdown code blocks
    response = response.replace("```python", "").replace("```", "")
    
    if pred is None:
        if len(response) == 0:
            return None
        return 1

    pred_str = str(pred).lower()
    if len(response) == 0:
        return None
    return 1 - len(pred_str) / len(response)


def if_code_prompt(file_name: str) -> bool:
    return "code" in file_name.lower()


def evaluate_result(
    result: dict, target_classes: list | dict, label_column_name: str = "label"
) -> dict:
    """Evaluate a single result dictionary containing 'response' and 'label' keys.
    Adds 'pred', 'correct', and 'redundancy' keys to the dictionary.
    """
    response = result.get("response", "")
    label = result.get(label_column_name, None)
    if isinstance(target_classes, dict):
        # convert label to the corresponding class name
        label = target_classes[label]
        target_classes = target_classes.values()

    pred = response_to_label(response, target_classes)
    correct = 1 if pred == label else 0
    redundancy = redundancy_score(response, pred)

    result["label"] = label
    result["pred"] = pred
    result["correct"] = correct
    result["redundancy"] = redundancy
    return result


def get_dataset(file_name: str) -> str:
    if file_name.startswith("sst"):
        return "sst"
    elif file_name.startswith("agnews"):
        return "agnews"
    elif file_name.startswith("cola"):
        return "cola"
    elif file_name.startswith("semeval"):
        return "semeval"
    elif file_name.startswith("scierc"):
        return "scierc"
    elif file_name.startswith("mscinli"):
        return "mscinli"
    elif file_name.startswith("xnli"):
        return "xnli"
    elif file_name.startswith("iris"):
        return "iris"
    elif file_name.startswith("hcc"):
        return "hcc"
    elif file_name.startswith("iris"):
        return "iris"
    elif file_name.startswith("xnli"):
        return "xnli"
    elif file_name.startswith("mscinli"):
        return "mscinli"
    elif file_name.startswith("mrpc"):
        return "mrpc"
    else:
        raise ValueError(f"Unknown dataset for file: {file_name}")


def get_model(file_name: str) -> str:
    # model name is the substring between the first and second underscore
    parts = file_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid file name format: {file_name}")
    return parts[1].lower()


def get_shot(file_name: str) -> int:
    # shot is the number before "_shot" in the file name
    if "_shot" not in file_name:
        return "0"
    # find the index of "_shot"
    idx = file_name.index("_shot")
    idx = idx - 1
    return int(file_name[idx])


def get_seed(file_name: str) -> int | None:
    # example: outputs/sst2_gpt-35-turbo-instruct_4_shot_seed_1_codeprompt.json
    # return: 1
    if "_seed" not in file_name:
        return None
    idx = file_name.index("_seed")
    return int(file_name[idx + 5 : idx + 6])


def get_language(file_name: str) -> str:
    # programming language for code prompts
    if "code" not in file_name.lower():
        return "nl"
    if "js" in file_name.lower() or "javascript" in file_name.lower():
        return "js"
    elif "cpp" in file_name.lower():
        return "cpp"
    return "python"


def evaluate_results_zero_shot(file_path: str) -> list[dict]:
    """Evaluate a list of result dictionaries from a JSONL file.
    Each dictionary should contain 'response' and 'label' keys.
    Returns a list of evaluated result dictionaries.
    """
    file_name = file_path.split("/")[-1]
    dataset = get_dataset(file_name)
    target_classes = ALL_LABELS[dataset]
    label_column_name = ALL_LABEL_COLUMN_NAMES[dataset]
    
    examples = []
    with open(file_path, "r") as f:
        results = json.load(f)

    for example in results:
        evaluate_result(example, target_classes, label_column_name)
        examples.append(example)
    
    # compute overall accuracy and average redundancy
    total = len(examples)
    correct = sum(e["correct"] for e in examples)
    # compute average redundancy, ignoring None values
    redundancies = [e["redundancy"] for e in examples if e["redundancy"] is not None]
    avg_redundancy = sum(redundancies) / len(redundancies) if redundancies else 0.0
    accuracy = correct / total if total > 0 else 0.0

    report = {
        "filename": file_name,
        "dataset": dataset,
        "model": get_model(file_name),
        "shot": get_shot(file_name),
        "seed": get_seed(file_name),
        "language": get_language(file_name),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_redundancy": avg_redundancy,
    }
    if dataset == "cola":
        y_true = [e["label"] for e in examples]
        y_pred = [e["pred"] if e["pred"] is not None else -1 for e in examples]
        mcc = matthews_corrcoef(y_true, y_pred)
        report["matthews_corrcoef"] = mcc
    elif dataset in ["semeval", "scierc"]:
        y_true = [e["label"] for e in examples]
        y_pred = [e["pred"] if e["pred"] is not None else "Other" for e in examples]
        f1 = f1_score(y_true, y_pred, average="micro", labels=list(target_classes.values()))
        report["f1_micro"] = f1
    elif dataset in ["mscinli"]:
        y_true = [e["label"] for e in examples]
        y_pred = [e["pred"] if e["pred"] is not None else "neutral" for e in examples]
        f1 = f1_score(y_true, y_pred, average="macro", labels=list(target_classes.values()))
        report["f1_macro"] = f1
    return report


def evaluate_directory(dir_path: str) -> list[dict]:
    """Enumerate all JSON files in the directory and evaluate them.
    Returns a list of evaluation reports for each file.
    """
    reports = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".json") or file_name.endswith(".jsonl"):
            file_path = os.path.join(dir_path, file_name)
            report = evaluate_results(file_path)
            reports.append(report)
    # order by filename
    reports = sorted(reports, key=lambda x: x["filename"])
    return reports

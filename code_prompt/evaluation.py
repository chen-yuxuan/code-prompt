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


def redundancy_score(response: str, pred: str | int) -> float:
    """Calculate the redundancy score of the prediction in the response.
    The redundancy score is defined as the length of the prediction divided by the length of the response.
    """
    if pred is None:
        return 0.0
    # clean response
    response = (
        response.lower().strip().replace("\n", " ").replace('"', " ").replace("'", "")
    )
    # special tokens from codegemma
    response = response.replace("<eos>", "").replace("<|file_separator|>", "")
    # markdown code blocks
    response = response.replace("```", "")

    pred_str = str(pred).lower().strip()
    if len(response) == 0:
        return 0.0
    return len(pred_str) / len(response)


def evaluate_result(
    result: dict, target_classes: list[str | int], label_column_name: str = "label"
) -> dict:
    """Evaluate a single result dictionary containing 'response' and 'label' keys.
    Adds 'pred', 'correct', and 'redundancy' keys to the dictionary.
    """
    response = result.get("response", "")
    label = result.get(label_column_name, None)

    pred = response_to_label(response, target_classes)
    correct = 1 if pred == label else 0
    redundancy = redundancy_score(response, pred)

    result["pred"] = pred
    result["correct"] = correct
    result["redundancy"] = redundancy
    return result

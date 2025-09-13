NL_PROMPT = (
    "You are a linguist to annotate textual entailment/contradiction."
    "Your task is to read a premise and a hypothesis, and then determine "
    "whether the relation between them is entailment, contradiction, or neutral.\n"
    "Entailment means the hypothesis logically follows from the premise. "
    "Contradiction means the hypothesis contradicts the premise. "
    "Neutral means neither entailment nor contradiction.\n"
    'Please answer with one word only: "entailment", "neutral" or "contradiction".'
    "\n{few_shot_examples}"
    "\nPremise: {premise}"
    "\nHypothesis: {hypothesis}"
    "\nRelation: "
)
NL_SHOT_PROMPT = "\nPremise: {premise}\nHypothesis: {hypothesis}\nRelation: {label}\n"

CODE_PROMPT = (
    "def natural_language_inference(premise: str, hypothesis: str){typing}:\n"
    '   """Classify the relationship between a premise and a hypothesis as\n'
    '   "entailment", "neutral", or "contradiction".\n'
    '   "entailment" means the hypothesis logically follows from the premise.\n'
    '   "contradiction" means the hypothesis contradicts the premise.\n'
    '   "neutral" means neither entailment nor contradiction.\n\n'
    "   Args:\n"
    "       - premise (str): The premise text.\n"
    "       - hypothesis (str): The hypothesis text.\n"
    "   Returns:\n"
    '       Literal["entailment", "neutral", "contradiction"]: The relationship label.\n'
    '   """\n'
    "   pass\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'premise = "{premise}"\n'
    'hypothesis = "{hypothesis}"\n'
    'assert (natural_language_inference(premise, hypothesis) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\npremise = "{premise}"'
    '\nhypothesis = "{hypothesis}"'
    '\nassert (natural_language_inference(premise, hypothesis) == "{label}")\n'
)


def map_label_id_to_str(label_id: int) -> str:
    """Map label id to string."""
    if label_id == 0:
        return "entailment"
    elif label_id == 1:
        return "neutral"
    else:
        return "contradiction"


def get_nl_prompt(example: dict, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            label = map_label_id_to_str(example["label"])
            few_shot_str += NL_SHOT_PROMPT.format(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
                label=label,
            )
    else:
        few_shot_str = ""
    return NL_PROMPT.format(
        few_shot_examples=few_shot_str,
        premise=example["premise"],
        hypothesis=example["hypothesis"],
    )


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
) -> str:
    """Get the code prompt."""
    _TYPING = (
        ' -> Literal["entailment", "neutral", "contradiction"]' if type_hint else ""
    )
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            label = map_label_id_to_str(example["label"])
            few_shot_str += CODE_SHOT_PROMPT.format(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
                label=label,
            )
    else:
        few_shot_str = ""
    prompt = CODE_PROMPT.format(
        typing=_TYPING,
        few_shot_examples=few_shot_str,
        premise=example["premise"],
        hypothesis=example["hypothesis"],
    )

    # add special tokens if necessary for different code LLMs
    if "gemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + "<|fim_suffix|>)\n<|fim_middle|>"
    if "deepseek-coder" in model.lower():
        return "<|fim_begin|>" + prompt + "<|fim_hole|>)\n<|fim_end|>"
    return prompt

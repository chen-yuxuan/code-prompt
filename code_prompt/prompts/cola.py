NL_PROMPT = (
    "You are a linguist to annotate grammatical acceptance. "
    "Your task is to read the following text and tell if it is grammatically "
    "acceptable or unacceptable in standard English.\n"
    'Please answer with "acceptable" or "unacceptable" only.'
    "\n{few_shot_examples}"
    "\nHere is the text to classify:"
    "\nInput: {text}"
    "\nOutput: "
)
NL_SHOT_PROMPT = "\nInput: {text}" "\nOutput: {{{label}}}\n"

CODE_PROMPT = (
    "def classify_grammatical_acceptance(text: str){typing}:\n"
    '   """Classify the given text as either grammatically "acceptable" or "unacceptable".\n\n'
    "   Args:\n"
    "       - text (str): The text to classify.\n"
    "   Returns:\n"
    '       Literal["acceptable", "unacceptable"]: The grammatical acceptance of the text.\n'
    "   pass\n"
    '   """\n\n\n'
    "{few_shot_examples}"
    "# Test case for inference\n"
    "text = {text}\n"
    'assert (classify_grammatical_acceptance(text) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    "\ntext = {text}"
    '\nassert (classify_grammatical_acceptance(text) == "{label}")\n'
)


def get_nl_prompt(text: str, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                text=example["sentence"],
                label="acceptable" if example["label"] == 1 else "unacceptable",
            )
    else:
        few_shot_str = ""
    return NL_PROMPT.format(text=text, few_shot_examples=few_shot_str)


def get_code_prompt(
    text: str,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
) -> str:
    """Get the code prompt."""
    _TYPING = ' -> Literal["acceptable", "unacceptable"]' if type_hint else ""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += CODE_SHOT_PROMPT.format(
                text=repr(example["sentence"]),
                label="acceptable" if example["label"] == 1 else "unacceptable",
            )
    else:
        few_shot_str = ""
    prompt = CODE_PROMPT.format(
        text=repr(text),
        few_shot_examples=few_shot_str,
        typing=_TYPING if type_hint else "",
    )

    # add special tokens if necessary for different code LLMs
    if "gemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + "<|fim_suffix|>)\n<|fim_middle|>"
    if "deepseek" in model.lower():
        return "<|fim_begin|>" + prompt + "<|fim_hole|>)\n<|fim_end|>"
    return prompt

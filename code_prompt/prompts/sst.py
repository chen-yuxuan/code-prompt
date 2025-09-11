NL_PROMPT = (
    "You are a data annotator for sentiment classification. "
    "Your task is to read the following text and classify the text into either "
    "positive or negative based on its sentiment.\n"
    'Please answer directly with "positive" or "negative" only.'
    "\n{few_shot_examples}"
    "\nHere is the text to classify:"
    "\nInput: {text}"
    "\nOutput: "
)
NL_SHOT_PROMPT = "\nInput: {text}" "\nOutput: {{{label}}}\n"

CODE_PROMPT = (
    "def classify_sentiment(text: str){typing}:\n"
    '   """Classify the sentiment of the given text as either positive or negative.\n'
    "   Args:\n"
    "       - text (str): The text to classify.\n"
    "   Returns:\n"
    '       Literal["positive", "negative"]: The sentiment label of the text.\n'
    '   """\npass\n\n'
    "{few_shot_examples}"
    "# Test case for inference\n"
    "text = {text}\n"
    'assert (classify_sentiment(text) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    "\ntext = {text}"
    '\nassert (classify_sentiment(text) == "{label}")\n'
)


def get_nl_prompt(text: str, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                text=example["sentence"],
                label="positive" if example["label"] == 1 else "negative",
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
    _TYPING = ' -> Literal["positive", "negative"]'
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += CODE_SHOT_PROMPT.format(
                text=repr(example["sentence"]),
                label="positive" if example["label"] == 1 else "negative",
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
    if "deepseek" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + "<|fim_hole|>)\n<|fim_end|>"
    # for openai and deepseek-v3
    return prompt

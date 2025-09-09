NL_PROMPT = (
    "You are a data annotator for topic classification. "
    "Your task is to read the following text and classify the text into one "
    "of the topics: World, Sport, Business or Sci/Tech.\n"
    'Please answer with "World" or "Sport" or "Business" or "Sci/Tech" only.'
    "\n{few_shot_examples}"
    "\nHere is the text to classify:"
    "\nInput: {text}"
    "\nOutput: "
)
NL_SHOT_PROMPT = "\nInput: {text}" "\nOutput: {{{label}}}\n"

CODE_PROMPT = (
    "def classify_topic(text: str){typing}:\n"
    '   """Classify the topic of the given text into one of the following:.\n'
    "       - World\n"
    "       - Sport\n"
    "       - Business\n"
    "       - Sci/Tech\n"
    "   Args:\n"
    "       - text (str): The text to classify.\n"
    "   Returns:\n"
    '       Literal["World", "Sport", "Business", "Sci/Tech"]: The topic label of the text.\n'
    "   pass\n"
    '   """\n\n\n'
    "{few_shot_examples}"
    "# Test case for inference\n"
    "text = {text}\n"
    'assert (classify_topic(text) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    "\ntext = {text}"
    '\nassert (classify_topic(text) == "{label}")\n'
)


CLASS_ID_TO_NAME = {1: "World", 2: "Sport", 3: "Business", 4: "Sci/Tech"}

def get_nl_prompt(text: str, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                text=example["sentence"],
                label=CLASS_ID_TO_NAME[example["label"]],
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
    _TYPING = ' -> Literal["World", "Sport", "Business", "Sci/Tech"]' if type_hint else ""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += CODE_SHOT_PROMPT.format(
                text=repr(example["sentence"]),
                label=CLASS_ID_TO_NAME[example["label"]],
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
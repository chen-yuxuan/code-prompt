NL_PROMPT = (
    "You are an annotator to stage a patient according to Milan-criteria "
    "given their tumor size(s). Your task is to read the following text "
    "about an HCC patient containing a list of tumor sizes and classify "
    "the patient into either True or False based on the rule of Milan criteria.\n"
    "Milan criteria: one tumor <= 5 cm or up to 3 tumors each <= 3 cm.\n"
    "If within the criteria, label True, else False. "
    'Please answer directly with True or False only.'
    "\n{few_shot_examples}"
    "\nHere is the text to classify:"
    "\nInput: The tumor sizes of the HCC patient: {text}"
    "\nOutput: "
)
NL_SHOT_PROMPT = "\nInput: The tumor sizes of the HCC patient: {text}\nOutput: {label}\n"

CODE_PROMPT = (
    "def HCC_staging(text: str){typing}:\n"
    '   """Classify the patient according to Milan criteria based on tumor sizes in the text.\n'
    "   Milan criteria: one tumor <= 5 cm or up to 3 tumors each ≤ 3 cm.\n"
    "   Args:\n"
    "       - text (str): The input text of the HCC patient containing tumor sizes.\n\n"
    "   Returns:\n"
    '       bool: True if the patient meets Milan criteria, else False.\n\n'
    '   """\n'
    '   pass\n\n'
    "{few_shot_examples}"
    "# Test case for inference\n"
    'text = "The tumor sizes of the HCC patient: {text}"\n'
    'assert (HCC_staging(text) == '
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\ntext = "The tumor sizes of the HCC patient: {text}"'
    '\nassert (classify_sentiment(text) == "{label}")\n'
)


def get_nl_prompt(example: str, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                text=example["TSIZE"].replace("\n", ", "),
                label=str(example["label"])
            )
    else:
        few_shot_str = ""
    return NL_PROMPT.format(
        text=example["TSIZE"].replace("\n", ", "),
        few_shot_examples=few_shot_str
    )


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
) -> str:
    """Get the code prompt."""
    _TYPING = ' -> bool'
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += CODE_SHOT_PROMPT.format(
                text=example["TSIZE"].replace("\n", ", "),
                label=str(example["label"])
            )
    else:
        few_shot_str = ""
    prompt = CODE_PROMPT.format(
        text=example["TSIZE"].replace("\n", ", "),
        few_shot_examples=few_shot_str + "\n",
        typing=_TYPING if type_hint else "",
    )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    # for openai and deepseek-v3
    return prompt

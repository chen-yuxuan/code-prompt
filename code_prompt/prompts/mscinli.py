NL_PROMPT = (
    "You are a linguist to annotate textual entailment/contrasting/reasoning/neutral"
    " over a pair of sentences extracted from publications in the field of computer science."
    "Your task is to read sentence1 and a sentence2, and then determine whether the "
    "relation between them is entailment, contrasting, reasoning or neutral.\n"
    "Entailment means sentence2 and can be appended to sentence1 using words like 'Specifically', "
    "'Precisely', 'In particular', 'Particularly', 'That is', 'In other words'."
    "Contrasting means sentence2 and can be appended to sentence1 using words like "
    "'However', 'On the other hand', 'In contrast', 'On the contrary'.\n"
    "Reasoning means sentence2 and can be appended to sentence1 using words like "
    "'Therefore', 'Thus', 'Consequently', 'As a result', 'As a consequence', 'From here we can infer'.\n"
    "Neutral means none of the above.\n"
    'Please answer with one word only: "entailment", "contrasting", "reasoning" or "neutral".\n'
    "\n{few_shot_examples}"
    "\nsentence1: {sentence1}"
    "\nsentence2: {sentence2}"
    "\nrelation: "
)
NL_SHOT_PROMPT = "\nsentence1: {sentence1}\nsentence2: {sentence2}\nrelation: {label}\n"

CODE_PROMPT = (
    "def natural_language_inference(sentence1: str, sentence2: str){typing}:\n"
    '   """Classify the relation between sentence1 and sentence2 from publications\n'
    "   in the field of computer science into one of 4 categories:\n"
    '   "entailment" means sentence2 can be appended to sentence1 using words like\n'
    "   'Specifically', 'Precisely', 'In particular', 'Particularly', 'That is', 'In other words'.\n"
    '   "contrasting" means sentence2 can be appended to sentence1 using words like\n'
    "   'However', 'On the other hand', 'In contrast', 'On the contrary'.\n"
    '   "reasoning" means sentence2 can be appended to sentence1 using words like\n'
    "   'Therefore', 'Thus', 'Consequently', 'As a result', 'As a consequence', 'From here we can infer'.\n"
    '   "neutral" means none of the above.\n'
    "   Args:\n"
    "       - sentence1 (str): The first sentence.\n"
    "       - sentence2 (str): The second sentence.\n"
    "   Returns:\n"
    '       Literal["entailment", "contrasting", "reasoning", "neutral"]: The relation between\n'
    "       sentence1 and sentence2.\n"
    '   """\n'
    "   pass\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'sentence1 = "{sentence1}"\n'
    'sentence2 = "{sentence2}"\n'
    'assert (natural_language_inference(sentence1, sentence2) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\nsentence1 = "{sentence1}"'
    '\nsentence2 = "{sentence2}"'
    '\nassert (natural_language_inference(sentence1, sentence2) == "{label}")\n'
)


def get_nl_prompt(example: dict, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                sentence1=example["sentence1"],
                sentence2=example["sentence2"],
                label=example["label"],
            )
    else:
        few_shot_str = ""
    return NL_PROMPT.format(
        few_shot_examples=few_shot_str,
        sentence1=example["sentence1"],
        sentence2=example["sentence2"],
    )


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
) -> str:
    """Get the code prompt."""
    _TYPING = (
        ' -> Literal["entailment", "contrasting", "reasoning", "neutral"]'
        if type_hint
        else ""
    )
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += CODE_SHOT_PROMPT.format(
                sentence1=example["sentence1"],
                sentence2=example["sentence2"],
                label=example["label"],
            )
    else:
        few_shot_str = ""
    prompt = CODE_PROMPT.format(
        typing=_TYPING,
        few_shot_examples=few_shot_str,
        sentence1=example["sentence1"],
        sentence2=example["sentence2"],
    )

    # add special tokens if necessary for different code LLMs
    if "gemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + "<|fim_suffix|>)\n<|fim_middle|>"
    if "deepseek-coder" in model.lower():
        return "<|fim_begin|>" + prompt + "<|fim_hole|>)\n<|fim_end|>"
    return prompt

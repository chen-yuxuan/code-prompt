NL_PROMPT = (
    "You are a linguist to annotate sentence pairs for semantic equivalence.\n"
    "Your task is to read two sentences and determine whether they are paraphrases "
    "of each other or not.\n"
    "Paraphrase (equivalent) means the two sentences express the same meaning. "
    "Not paraphrase (not_equivalent) means they do not express the same meaning.\n"
    'Please answer with one word only: "equivalent" or "not_equivalent".'
    "\n{few_shot_examples}"
    "\nSentence 1: {sentence1}"
    "\nSentence 2: {sentence2}"
    "\nRelation: "
)
NL_SHOT_PROMPT = (
    "\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nRelation: {label}\n"
)

CODE_PROMPT = (
    "def detect_paraphrase(sentence1: str, sentence2: str){typing}:\n"
    '   """Detect if two sentences are paraphrases of each other.\n'
    "   Paraphrase means the two sentences express the same meaning.\n"
    "   Args:\n"
    "       - sentence1 (str): The first sentence.\n"
    "       - sentence2 (str): The second sentence.\n"
    "   Returns:\n"
    '       Literal["equivalent", "not_equivalent"]: "equivalent" if the sentences are paraphrases, '
    '"not_equivalent" otherwise.\n'
    '   """\n'
    "   pass\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'sentence1 = "{sentence1}"\n'
    'sentence2 = "{sentence2}"\n'
    'assert (detect_paraphrase(sentence1, sentence2) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\nsentence1 = "{sentence1}"'
    '\nsentence2 = "{sentence2}"'
    '\nassert (detect_paraphrase(sentence1, sentence2) == "{label}")\n'
)


def map_label_id_to_str(label_id: int) -> str:
    """Map label id to string."""
    if label_id == 0:
        return "equivalent"
    elif label_id == 1:
        return "not_equivalent"
    else:
        raise ValueError(f"Invalid label id: {label_id}")


def get_nl_prompt(example: dict, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            label = map_label_id_to_str(e["label"])
            few_shot_str += NL_SHOT_PROMPT.format(
                sentence1=e["sentence1"],
                sentence2=e["sentence2"],
                label=label,
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
    _TYPING = ' -> Literal["equivalent", "not_equivalent"]' if type_hint else ""
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            label = map_label_id_to_str(e["label"])
            few_shot_str += CODE_SHOT_PROMPT.format(
                sentence1=e["sentence1"],
                sentence2=e["sentence2"],
                label=label,
            )
    else:
        few_shot_str = ""
    prompt = CODE_PROMPT.format(
        typing=_TYPING,
        few_shot_examples=few_shot_str + "\n",
        sentence1=example["sentence1"],
        sentence2=example["sentence2"],
    )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    return prompt

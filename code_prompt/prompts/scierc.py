import re


NL_PROMPT = (
    "You are a data annotator for relation classification of two entities in a text. "
    "Your task is to read the following text in which the subject entity is surrounded "
    "by '[[' and ']]' , and object entity is surrounded by '<<' and '>>', and then "
    "classify the relation between the two entities into one of the following 7 labels:\n"
    '"compare", "conjunction", "evaluate-for", "feature-of", '
    '"hyponym-of", "part-of", "used-for".\n'
    "\n{few_shot_examples}"
    "\nHere is the text for relation classification:"
    "\nInput: {text}"
    "\nOutput: The relation between {subj} and {obj} in the input is "
)
NL_SHOT_PROMPT = "\nInput:\n{text}\nOutput: The relation between {subj} and {obj} in the input is {label}.\n"

CODE_PROMPT = (
    "def classify_relation(text: str, subj: str, obj: str){typing}:\n"
    '   """Classify the relation of the subject entity and object entity from the given text\n'
    "   into one of the following 7 labels:\n"
    '   "compare", "conjunction", "evaluate-for", "feature-of",\n'
    '   "hyponym-of", "part-of", "used-for".\n'
    "   Args:\n"
    "       - text (str): A text with a subject entity surrounded by '[[' and ']]' ,\n"
    "   and an object entity surrounded by '<<' and '>>'.\n"
    "       - subj (str): The subject entity in the text.\n"
    "       - obj (str): The object entity in the text.\n"
    "   Returns:\n"
    '       Literal["compare", "conjunction", "evaluate-for", "feature-of", '
    '       "hyponym-of", "part-of", "used-for"]: The relation label between the subject entity '
    "       and object entity.\n"
    '   """\n'
    "   pass\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'text = "{text}"\n'
    'subj, obj = "{subj}", "{obj}"\n'
    'assert (classify_relation(text, subj, obj) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\ntext = "{text}"'
    '\nsubj, obj = "{subj}", "{obj}"'
    '\nassert (classify_relation(text, subj, obj) == "{label}")\n'
)


def get_subj_text(text: str) -> str:
    """Get the substring surrounded by '[[' and ']]'."""
    try:
        start = text.index("[[") + 2
        end = text.index("]]", start)
        return text[start:end].strip()
    except ValueError:
        raise ValueError("Subject entity markers '[[' and ']]' not found in the text.")


def get_obj_text(text: str) -> str:
    """Get the substring surrounded by '<<' and '>>'."""
    try:
        start = text.index("<<") + 2
        end = text.index(">>", start)
        return text[start:end].strip()
    except ValueError:
        raise ValueError("Object entity markers '<<' and '>>' not found in the text.")


def remove_special_tokens(text: str) -> str:
    """
    Replace Penn Treebank bracket tokens:
      -LRB-/ -RRB- → ( )
      -LSB-/ -RSB- → [ ]
    Remove only extra spaces around these tokens, without affecting
    the rest of the text.
    """
    # Replace tokens with brackets
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")

    # Fix spaces around brackets
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)

    return (
        text.replace("<< ", "<<")
        .replace(" >>", ">>")
        .replace("[[ ", "[[")
        .replace(" ]]", "]]")
    )


def get_nl_prompt(example: dict, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            cleaned_text = remove_special_tokens(e["text"])
            few_shot_str += NL_SHOT_PROMPT.format(
                text=cleaned_text,
                subj=get_subj_text(cleaned_text),
                obj=get_obj_text(cleaned_text),
                label=e["label"],
            )
    else:
        few_shot_str = ""

    cleaned_text = remove_special_tokens(example["text"])
    return NL_PROMPT.format(
        few_shot_examples=few_shot_str,
        text=cleaned_text,
        subj=get_subj_text(cleaned_text),
        obj=get_obj_text(cleaned_text),
    )


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
) -> str:
    """Get the code prompt."""
    _TYPING = (
        ' -> Literal["compare", "conjunction", "evaluate-for", "feature-of", '
        '"hyponym-of", "part-of", "used-for"]'
        if type_hint
        else ""
    )
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            cleaned_text = remove_special_tokens(e["text"])
            few_shot_str += CODE_SHOT_PROMPT.format(
                text=cleaned_text,
                subj=get_subj_text(cleaned_text),
                obj=get_obj_text(cleaned_text),
                label=e["label"],
            )
    else:
        few_shot_str = ""

    cleaned_text = remove_special_tokens(example["text"])
    prompt = CODE_PROMPT.format(
        few_shot_examples=few_shot_str + "\n",
        text=cleaned_text,
        subj=get_subj_text(cleaned_text),
        obj=get_obj_text(cleaned_text),
        typing=_TYPING if type_hint else "",
    )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    return prompt

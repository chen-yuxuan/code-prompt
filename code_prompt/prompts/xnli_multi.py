CODE_PROMPT = (
    "def natural_language_inference(premise: str, hypothesis: str){typing}:\n"
    '   """Classify the relationship between a premise and a hypothesis (both written in {lang}) as\n'
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


def map_lang_abbr_to_str(lang: str) -> str:
    """Map language abbreviation to string."""
    if lang == "en":
        return "English"
    elif lang == "ar":
        return "Arabic"
    elif lang == "bg":
        return "Bulgarian"
    elif lang == "de":
        return "German"
    elif lang == "el":
        return "Greek"
    elif lang == "es":
        return "Spanish"
    elif lang == "fr":
        return "French"
    elif lang == "hi":
        return "Hindi"
    elif lang == "ru":
        return "Russian"
    elif lang == "sw":
        return "Swahili"
    elif lang == "th":
        return "Thai"
    elif lang == "tr":
        return "Turkish"
    elif lang == "ur":
        return "Urdu"
    elif lang == "vi":
        return "Vietnamese"
    elif lang == "zh":
        return "Chinese"
    else:
        return "English"


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
    lang: str = "de",
) -> str:
    """Get the code prompt."""
    _TYPING = (
        ' -> Literal["entailment", "neutral", "contradiction"]' if type_hint else ""
    )
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            label = map_label_id_to_str(e["label"])
            few_shot_str += CODE_SHOT_PROMPT.format(
                premise=e["premise"],
                hypothesis=e["hypothesis"],
                label=label,
            )
    else:
        few_shot_str = ""
    prompt = CODE_PROMPT.format(
        typing=_TYPING,
        few_shot_examples=few_shot_str + "\n",
        premise=example["premise"],
        hypothesis=example["hypothesis"],
        lang=map_lang_abbr_to_str(lang),
    )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    return prompt

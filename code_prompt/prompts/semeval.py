NL_PROMPT = (
    "You are a data annotator for relation classification of two entities in a text. "
    "Your task is to read the following text in which two entities are marked by <e1> "
    "and </e1> for the subject entity, and <e2> and </e2> for the object entity, and "
    "classify the relation between the two entities into one of the following 10 labels:\n"
    '"Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination", '
    '"Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic", '
    '"Product-Producer", "Other",\n'
    'where "Other" means there is no relation between the two entities.\n'
    "\n{few_shot_examples}"
    "\nHere is the text for relation classification:"
    "\nInput: {text}"
    "\nOutput: The relation between {subj} and {obj} in the input is "
)
NL_SHOT_PROMPT = "\nInput:\n{text}\nOutput: The relation between {subj} and {obj} in the input is {label}.\n"

CODE_PROMPT = (
    "def classify_relation(text: str, subj: str, obj: str){typing}:\n"
    '   """Classify the relation of the subject entity and object entity from the given text\n'
    "   into one of the following 10 labels:\n"
    '   "Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination",\n'
    '   "Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic",\n'
    '   "Product-Producer", "Other", where "Other" means there is no relation between the two entities.\n'
    "   Args:\n"
    "       - text (str): A text with a subject entity marked by <e1> and </e1>, and an object entity "
    "       marked by <e2> and </e2>.\n"
    "       - subj (str): The subject entity in the text.\n"
    "       - obj (str): The object entity in the text.\n"
    "   Returns:\n"
    '       Literal["Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination", '
    '       "Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic", '
    '       "Product-Producer", "Other"]: The relation label between the subject entity and object entity.\n'
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

CODE_PROMPT_JS = (
    "const assert = require('assert');\n\n"
    "function classifyRelation(text, subj, obj) {{\n"
    "    /*\n"
    "    Classify the relation of the subject entity and object entity from the given text\n"
    "    into one of the following 10 labels:\n"
    '    "Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination",\n'
    '    "Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic",\n'
    '    "Product-Producer", "Other", where "Other" means there is no relation between the two entities.\n'
    "    Args:\n"
    "        - text (string): A text with a subject entity marked by <e1> and </e1>, and an object entity "
    "          marked by <e2> and </e2>.\n"
    "        - subj (string): The subject entity in the text.\n"
    "        - obj (string): The object entity in the text.\n"
    "    Returns:\n"
    '        "Cause-Effect" | "Component-Whole" | "Content-Container" | "Entity-Destination" |\n'
    '        "Entity-Origin" | "Instrument-Agency" | "Member-Collection" | "Message-Topic" |\n'
    '        "Product-Producer" | "Other": The relation label between the subject entity and object entity.\n'
    "    */\n"
    "    // TODO: Implement classification logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'const text = "{text}";\n'
    'const subj = "{subj}", obj = "{obj}";\n'
    'assert.strictEqual(classifyRelation(text, subj, obj), "'
)
CODE_SHOT_PROMPT_JS = (
    "\n// Example test case"
    '\nconst text = "{text}";'
    '\nconst subj = "{subj}", obj = "{obj}";'
    '\nassert.strictEqual(classifyRelation(text, subj, obj), "{label}");\n'
)

CODE_PROMPT_CPP = (
    "#include <cassert>\n"
    "#include <string>\n\n"
    "std::string classify_relation(const std::string& text, const std::string& subj, const std::string& obj) {{\n"
    "    /*\n"
    "    Classify the relation of the subject entity and object entity from the given text\n"
    "    into one of the following 10 labels:\n"
    '    "Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination",\n'
    '    "Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic",\n'
    '    "Product-Producer", "Other", where "Other" means there is no relation between the two entities.\n'
    "    Args:\n"
    "        - text (std::string): A text with a subject entity marked by <e1> and </e1>, and an object entity "
    "          marked by <e2> and </e2>.\n"
    "        - subj (std::string): The subject entity in the text.\n"
    "        - obj (std::string): The object entity in the text.\n"
    "    Returns:\n"
    '        "Cause-Effect" or "Component-Whole" or "Content-Container" or "Entity-Destination" or\n'
    '        "Entity-Origin" or "Instrument-Agency" or "Member-Collection" or "Message-Topic" or\n'
    '        "Product-Producer" or "Other": The relation label between the subject entity and object entity.\n'
    "    */\n"
    "    // TODO: Implement classification logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'std::string text = "{text}";\n'
    'std::string subj = "{subj}", obj = "{obj}";\n'
    'assert(classify_relation(text, subj, obj) == "'
)
CODE_SHOT_PROMPT_CPP = (
    "\n// Example test case"
    '\nstd::string text = "{text}";'
    '\nstd::string subj = "{subj}", obj = "{obj}";'
    '\nassert(classify_relation(text, subj, obj) == "{label}");\n'
)


def tokens_to_text(example: dict) -> str:
    tokens = example["token"]
    punctuations = {".", ",", "?", "!", ";", ":"}
    no_space_before = punctuations | {"</e1>", "</e2>"}
    no_space_after = {"<e1>", "<e2>"}

    text = ""
    for i, tok in enumerate(tokens):
        if i > 0:
            if tok not in no_space_before and tokens[i - 1] not in no_space_after:
                text += " "
        text += tok
    return text


def get_subj_text(example):
    tokens = example["token"]
    subj_tokens = tokens[example["subj_start"] + 1 : example["subj_end"]]
    return " ".join(subj_tokens)


def get_obj_text(example):
    tokens = example["token"]
    obj_tokens = tokens[example["obj_start"] + 1 : example["obj_end"]]
    return " ".join(obj_tokens)


def get_relation_without_order(example) -> str:
    # to remove "(e1,e2)" or "(e2,e1)" in the relation string
    # chunk at the first "(" and take the first part
    relation = example["relation"].split("(")[0].strip()
    return relation


def get_nl_prompt(example: dict, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                text=tokens_to_text(e),
                subj=get_subj_text(e),
                obj=get_obj_text(e),
                label=get_relation_without_order(e),
            )
    else:
        few_shot_str = ""
    return NL_PROMPT.format(
        few_shot_examples=few_shot_str + "\n",
        text=tokens_to_text(example),
        subj=get_subj_text(example),
        obj=get_obj_text(example),
    )


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
    language: str = "python",
) -> str:
    """Get the code prompt."""
    _TYPING = (
        ' -> Literal["Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination", '
        '"Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic", '
        '"Product-Producer", "Other"]'
    )
    code_prompt, code_shot_prompt = CODE_PROMPT, CODE_SHOT_PROMPT
    if language.lower() in "javascript" or language.lower() in "js":
        code_prompt, code_shot_prompt = CODE_PROMPT_JS, CODE_SHOT_PROMPT_JS
    elif language.lower() == "cpp":
        code_prompt, code_shot_prompt = CODE_PROMPT_CPP, CODE_SHOT_PROMPT_CPP
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            few_shot_str += code_shot_prompt.format(
                text=tokens_to_text(e),
                subj=get_subj_text(e),
                obj=get_obj_text(e),
                label=get_relation_without_order(e),
            )
    else:
        few_shot_str = ""
    if language.lower() in ["python", "py"]:
        prompt = code_prompt.format(
            few_shot_examples=few_shot_str + "\n",
            text=tokens_to_text(example),
            subj=get_subj_text(example),
            obj=get_obj_text(example),
            typing=_TYPING if type_hint else "",
        )
    else:
        prompt = code_prompt.format(
            few_shot_examples=few_shot_str,
            text=tokens_to_text(example),
            subj=get_subj_text(example),
            obj=get_obj_text(example),
        )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    return prompt

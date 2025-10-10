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

CODE_PROMPT_JS = (
    "const assert = require('assert');\n\n"
    "function detectParaphrase(sentence1, sentence2) {{\n"
    "    /*\n"
    "    Detect if two sentences are paraphrases of each other.\n"
    "    Paraphrase means the two sentences express the same meaning.\n"
    "    Args:\n"
    "        - sentence1 (string): The first sentence.\n"
    "        - sentence2 (string): The second sentence.\n"
    "    Returns:\n"
    '        "equivalent" | "not_equivalent": "equivalent" if the sentences are paraphrases, '
    '"not_equivalent" otherwise.\n'
    "    */\n"
    "    // TODO: Implement paraphrase detection logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'const sentence1 = "{sentence1}";\n'
    'const sentence2 = "{sentence2}";\n'
    'assert.strictEqual(detectParaphrase(sentence1, sentence2), "'
)
CODE_SHOT_PROMPT_JS = (
    "\n// Example test case"
    '\nconst sentence1 = "{sentence1}";'
    '\nconst sentence2 = "{sentence2}";'
    '\nassert.strictEqual(detectParaphrase(sentence1, sentence2), "{label}");\n'
)

CODE_PROMPT_CPP = (
    "#include <cassert>\n"
    "#include <string>\n\n"
    "std::string detect_paraphrase(const std::string& sentence1, const std::string& sentence2) {{\n"
    "    /*\n"
    "    Detect if two sentences are paraphrases of each other.\n"
    "    Paraphrase means the two sentences express the same meaning.\n"
    "    Args:\n"
    "        - sentence1 (std::string): The first sentence.\n"
    "        - sentence2 (std::string): The second sentence.\n"
    "    Returns:\n"
    '        "equivalent" or "not_equivalent": "equivalent" if the sentences are paraphrases, '
    '"not_equivalent" otherwise.\n'
    "    */\n"
    "    // TODO: Implement paraphrase detection logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'std::string sentence1 = "{sentence1}";\n'
    'std::string sentence2 = "{sentence2}";\n'
    'assert(detect_paraphrase(sentence1, sentence2) == "'
)
CODE_SHOT_PROMPT_CPP = (
    "\n// Example test case"
    '\nstd::string sentence1 = "{sentence1}";'
    '\nstd::string sentence2 = "{sentence2}";'
    '\nassert(detect_paraphrase(sentence1, sentence2) == "{label}");\n'
)

CODE_PROMPT_RUBY = (
    "def detect_paraphrase(sentence1, sentence2)\n"
    "    # Detect if two sentences are paraphrases of each other.\n"
    "    # Paraphrase means the two sentences express the same meaning.\n"
    "    # Args:\n"
    "    #     - sentence1 (String): The first sentence.\n"
    "    #     - sentence2 (String): The second sentence.\n"
    "    # Returns:\n"
    '    #     "equivalent" or "not_equivalent": "equivalent" if the sentences are paraphrases, '
    '"not_equivalent" otherwise.\n'
    "    # TODO: Implement paraphrase detection logic\n"
    "end\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'sentence1 = "{sentence1}"\n'
    'sentence2 = "{sentence2}"\n'
    'raise "Assertion failed" unless (detect_paraphrase(sentence1, sentence2) == "'
)
CODE_PROMPT_SHOT_PROMPT_RUBY = (
    "\n# Example test case"
    '\nsentence1 = "{sentence1}"'
    '\nsentence2 = "{sentence2}"'
    '\nraise "Assertion failed" unless (detect_paraphrase(sentence1, sentence2) == "{label}")\n'
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
    language: str = "python",
) -> str:
    """Get the code prompt."""
    _TYPING = ' -> Literal["equivalent", "not_equivalent"]'
    code_prompt, code_shot_prompt = CODE_PROMPT, CODE_SHOT_PROMPT
    if language.lower() in "javascript" or language.lower() in "js":
        code_prompt, code_shot_prompt = CODE_PROMPT_JS, CODE_SHOT_PROMPT_JS
    elif language.lower() == "cpp":
        code_prompt, code_shot_prompt = CODE_PROMPT_CPP, CODE_SHOT_PROMPT_CPP
    elif language.lower() == "ruby":
        code_prompt, code_shot_prompt = CODE_PROMPT_RUBY, CODE_PROMPT_SHOT_PROMPT_RUBY
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            label = map_label_id_to_str(e["label"])
            few_shot_str += code_shot_prompt.format(
                sentence1=e["sentence1"],
                sentence2=e["sentence2"],
                label=label,
            )
    else:
        few_shot_str = ""
    if language.lower() in ["python", "py"]:
        prompt = code_prompt.format(
            sentence1=example["sentence1"],
            sentence2=example["sentence2"],
            few_shot_examples=few_shot_str + "\n",
            typing=_TYPING if type_hint else "",
        )
    else:
        prompt = code_prompt.format(
            sentence1=example["sentence1"],
            sentence2=example["sentence2"],
            few_shot_examples=few_shot_str + "\n",
        )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    return prompt

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

CODE_PROMPT_JS = (
    "const assert = require('assert');\n\n"
    "function naturalLanguageInference(sentence1, sentence2) {\n"
    '  /**\n'
    "   * Classify the relation between sentence1 and sentence2 from publications\n"
    "   * in the field of computer science into one of 4 categories:\n"
    '   * "entailment" means sentence2 can be appended to sentence1 using words like\n'
    "   * 'Specifically', 'Precisely', 'In particular', 'Particularly', 'That is', 'In other words'.\n"
    '   * "contrasting" means sentence2 can be appended to sentence1 using words like\n'
    "   * 'However', 'On the other hand', 'In contrast', 'On the contrary'.\n"
    '   * "reasoning" means sentence2 can be appended to sentence1 using words like\n'
    "   * 'Therefore', 'Thus', 'Consequently', 'As a result', 'As a consequence', 'From here we can infer'.\n"
    '   * "neutral" means none of the above.\n'
    "   * @param {string} sentence1 - The first sentence.\n"
    "   * @param {string} sentence2 - The second sentence.\n"
    '   * @returns {"entailment" | "contrasting" | "reasoning" | "neutral"} The relation between\n'
    "   *          sentence1 and sentence2.\n"
    "   */\n"
    "  // TODO: Implement logic here\n"
    "}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'const sentence1 = "{sentence1}";\n'
    'const sentence2 = "{sentence2}";\n'
    'assert.strictEqual(naturalLanguageInference(sentence1, sentence2), "'
)
CODE_SHOT_PROMPT_JS = (
    "\n// Example test case"
    '\nconst sentence1 = "{sentence1}";'
    '\nconst sentence2 = "{sentence2}";'
    '\nassert.strictEqual(naturalLanguageInference(sentence1, sentence2), "{label}");\n'
)

CODE_PROMPT_CPP = (
    "#include <cassert>\n"
    "#include <string>\n"
    "std::string natural_language_inference(const std::string& sentence1, const std::string& sentence2) {{\n"
    "    /**\n"
    "    Classify the relation between sentence1 and sentence2 from publications\n"
    "    in the field of computer science into one of 4 categories:\n"
    '    "entailment" means sentence2 can be appended to sentence1 using words like\n'
    "    'Specifically', 'Precisely', 'In particular', 'Particularly', 'That is', 'In other words'.\n"
    '    "contrasting" means sentence2 can be appended to sentence1 using words like\n'
    "    'However', 'On the other hand', 'In contrast', 'On the contrary'.\n"
    '    "reasoning" means sentence2 can be appended to sentence1 using words like\n'
    "    'Therefore', 'Thus', 'Consequently', 'As a result', 'As a consequence', 'From here we can infer'.\n"
    '    "neutral" means none of the above.\n'
    "     Args:\n"
    "         - sentence1 (const std::string&): The first sentence.\n"
    "         - sentence2 (const std::string&): The second sentence.\n"
    '     Returns:\n'
    '         std::string: The relation between sentence1 and sentence2. One of "entailment", "contrasting", "reasoning", "neutral".\n'
    "    */\n"
    "    // TODO: Implement logic here\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'std::string sentence1 = "{sentence1}";\n'
    'std::string sentence2 = "{sentence2}";\n'
    'assert(natural_language_inference(sentence1, sentence2) == "'
)
CODE_SHOT_PROMPT_CPP = (
    "\n// Example test case"
    '\nstd::string sentence1 = "{sentence1}";'
    '\nstd::string sentence2 = "{sentence2}";'
    '\nassert(natural_language_inference(sentence1, sentence2) == "{label}");\n'
)

CODE_PROMPT_RUBY = (
    "def natural_language_inference(sentence1, sentence2)\n"
    '    # Classify the relation between sentence1 and sentence2 from publications\n'
    "    # in the field of computer science into one of 4 categories:\n"
    '    # "entailment" means sentence2 can be appended to sentence1 using words like\n'
    "    # 'Specifically', 'Precisely', 'In particular', 'Particularly', 'That is', 'In other words'.\n"
    '    # "contrasting" means sentence2 can be appended to sentence1 using words like\n'
    "    # 'However', 'On the other hand', 'In contrast', 'On the contrary'.\n"
    '    # "reasoning" means sentence2 can be appended to sentence1 using words like\n'
    "    # 'Therefore', 'Thus', 'Consequently', 'As a result', 'As a consequence', 'From here we can infer'.\n"
    '    # "neutral" means none of the above.\n'
    '    # Args:\n'
    '    #   - sentence1 (String): The first sentence.\n'
    '    #   - sentence2 (String): The second sentence.\n'
    '    # Returns:\n'
    '    #   String: The relation between sentence1 and sentence2. One of "entailment", "contrasting", "reasoning", "neutral".\n'
    "    # TODO: Implement logic here\n"
    "end\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'sentence1 = "{sentence1}"\n'
    'sentence2 = "{sentence2}"\n'
    'raise "Assertion failed" unless (natural_language_inference(sentence1, sentence2) == "'
)
CODE_SHOT_PROMPT_RUBY = (
    "\n# Example test case"
    '\nsentence1 = "{sentence1}"'
    '\nsentence2 = "{sentence2}"'
    '\nraise "Assertion failed" unless (natural_language_inference(sentence1, sentence2) == "{label}")\n'
)


def get_nl_prompt(example: dict, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                sentence1=e["sentence1"],
                sentence2=e["sentence2"],
                label=e["label"],
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
    _TYPING = ' -> Literal["entailment", "contrasting", "reasoning", "neutral"]'
    code_prompt, code_shot_prompt = CODE_PROMPT, CODE_SHOT_PROMPT
    if language.lower() == "javascript" or language.lower() == "js":
        code_prompt, code_shot_prompt = CODE_PROMPT_JS, CODE_SHOT_PROMPT_JS
    elif language.lower() == "cpp":
        code_prompt, code_shot_prompt = CODE_PROMPT_CPP, CODE_SHOT_PROMPT_CPP
    elif language.lower() == "ruby":
        code_prompt, code_shot_prompt = CODE_PROMPT_RUBY, CODE_SHOT_PROMPT_RUBY
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            few_shot_str += code_shot_prompt.format(
                sentence1=e["sentence1"],
                sentence2=e["sentence2"],
                label=e["label"],
            )
    else:
        few_shot_str = ""
    
    if language.lower() in ["python", "py"]:
        prompt = code_prompt.format(
            typing=_TYPING if type_hint else "",
            few_shot_examples=few_shot_str + "\n",
            sentence1=example["sentence1"],
            sentence2=example["sentence2"],
        )
    else:
        prompt = code_prompt.format(
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

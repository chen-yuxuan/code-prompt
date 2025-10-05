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
NL_SHOT_PROMPT = "\nInput: {text}" "\nOutput: {label}\n"

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
    'text = "{text}"\n'
    'assert (classify_sentiment(text) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\ntext = "{text}"'
    '\nassert (classify_sentiment(text) == "{label}")\n'
)

CODE_PROMPT_JS = (
    "const assert = require('assert');\n\n"
    "function classifySentiment(text) {{\n"
    "    /*\n"
    "    Classify the sentiment of the given text as either positive or negative.\n"
    "    Args:\n"
    "        - text (string): The text to classify.\n"
    "    Returns:\n"
    '        "positive" | "negative": The sentiment label of the text.\n'
    "    */\n"
    "    // TODO: Implement classification logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'const text = "{text}";\n'
    'assert.strictEqual(classifySentiment(text), "'
)
CODE_SHOT_PROMPT_JS = (
    "\n// Example test case"
    '\nconst text = "{text}";'
    '\nassert.strictEqual(classifySentiment(text), "{label}");\n'
)

CODE_PROMPT_CPP = (
    "#include <cassert>\n"
    "#include <string>\n\n"
    "std::string classify_sentiment(const std::string& text) {{\n"
    "    /*\n"
    "    Classify the sentiment of the given text as either positive or negative.\n"
    "    Args:\n"
    "        - text (std::string): The text to classify.\n"
    "    Returns:\n"
    '        "positive" or "negative": The sentiment label of the text.\n'
    "    */\n"
    "    // TODO: Implement classification logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'std::string text = "{text}";\n'
    'assert(classify_sentiment(text) == "'
)
CODE_SHOT_PROMPT_CPP = (
    "\n// Example test case"
    '\nstd::string text = "{text}";'
    '\nassert(classifySentiment(text) == "{label}");\n'
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
    language: str = "python",
) -> str:
    """Get the code prompt."""
    _TYPING = ' -> Literal["positive", "negative"]'
    code_prompt, code_shot_prompt = CODE_PROMPT, CODE_SHOT_PROMPT
    if language.lower() in "javascript" or language.lower() in "js":
        code_prompt, code_shot_prompt = CODE_PROMPT_JS, CODE_SHOT_PROMPT_JS
    elif language.lower() == "cpp":
        code_prompt, code_shot_prompt = CODE_PROMPT_CPP, CODE_SHOT_PROMPT_CPP

    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += code_shot_prompt.format(
                text=example["sentence"],
                label="positive" if example["label"] == 1 else "negative",
            )
    else:
        few_shot_str = ""
    if language.lower() in ["python", "py"]:
        prompt = code_prompt.format(
            text=text,
            few_shot_examples=few_shot_str + "\n",
            typing=_TYPING if type_hint else "",
        )
    else:
        prompt = code_prompt.format(
            text=text,
            few_shot_examples=few_shot_str + "\n",
        )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    # for openai and deepseek-v3
    return prompt

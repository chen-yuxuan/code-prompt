NL_PROMPT = (
    "You are a clinical annotator to determine whether a patient with hepatocellular carcinoma (HCC) "
    "meets the Milan criteria based on reported tumor size(s).\n"
    "Milan criteria:\n"
    "  - One single tumor with its diameter <= 5 cm,\n"
    "  - Alternatively, up to 3 tumors, each with its diameter <= 3 cm.\n\n"
    "Important notes on tumor size interpretation:\n"
    "  - Tumor size (or diameter) may be reported in millimeters (mm) or centimeters (cm).\n"
    "  - It may appear as:\n"
    '      * A single value (e.g., "25 mm"),\n'
    '      * Two dimensions (e.g., "25 x 20 mm"), or\n'
    '      * Three dimensions (e.g., "2.5 x 2.0 x 1.8 cm").\n'
    "  - In all cases, the diameter is defined as the largest single dimension."
    "  - If a patient has multiple tumors, their sizes will be listed together as a comma-separated text, "
    "    e.g., '5 mm, 30 mm, 3x4x5 mm'.\n\n"
    "Your task: Read the provided text describing the patient's tumor size(s), "
    "and classify whether the patient meets the Milan criteria.\n"
    "If the patient is within the criteria, label 'True'. If not, label 'False'.\n"
    "Please answer directly with 'True' or 'False' only."
    "\n{few_shot_examples}"
    "\nHere is the text to classify:"
    "\nInput: The tumor sizes of the HCC patient: {text}"
    "\nOutput: "
)
NL_SHOT_PROMPT = (
    "\nInput: The tumor size(s) of the HCC patient: {text}\nOutput: {label}\n"
)

CODE_PROMPT = (
    "def HCC_staging(text: str){typing}:\n"
    '   """Classify the patient according to Milan criteria based on tumor size(s) in the text.\n\n'
    "   Milan criteria:\n"
    "       - One single tumor with its diameter <= 5 cm,\n"
    "       - Alternatively, up to 3 tumors, each with its diameter <= 3 cm.\n\n"
    "   Important notes on tumor size interpretation:\n"
    "       - Tumor size (or diameter) may be reported in millimeters (mm) or centimeters (cm).\n"
    "       - It may appear as:\n"
    '           * A single value (e.g., "25 mm"),\n'
    '           * Two dimensions (e.g., "25 x 20 mm"), or\n'
    '           * Three dimensions (e.g., "2.5 x 2.0 x 1.8 cm").\n'
    "       - In all cases, the diameter is defined as the largest single dimension."
    "       - If a patient has multiple tumors, their sizes will be listed together as a comma-separated text, "
    "    e.g., '5 mm, 30 mm, 3x4x5 mm'.\n\n"
    "   This function reads the provided `text` describing the patient's tumor size(s),\n"
    "   and classify whether the patient meets the Milan criteria.\n"
    "   If the patient is within the criteria, return 'True'. If not, return 'False'.\n"
    "   Args:\n"
    "       - text (str): The input text of the HCC patient reporting tumor size(s).\n"
    "   Returns:\n"
    '       - str: "True" if the patient meets the Milan criteria, "False" otherwise.\n'
    '   """\n'
    "   {body}\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'text = "The tumor size(s) of the HCC patient: {text}"\n'
    'assert (HCC_staging(text) == "'
)
BODY = (
    "   # Extract tumor sizes from the text\n"
    "   sizes: list[str] = extract_tumor_sizes(text)\n"
    "   # Convert sizes to millimeters\n"
    "   sizes: list[float] = [convert_to_mm(size) for size in sizes]\n"
    "   # Get the largest diameter for each tumor size\n"
    "   diameters: list[float] = [get_largest_diameter(size) for size in sizes]\n"
    "   # Determine if the patient meets the Milan criteria\n"
    "   if len(diameters) == 1 and diameters[0] <= 50:\n"
    '       return "True"\n'
    "   elif len(diameters) <= 3 and all(d <= 30 for d in diameters):\n"
    '       return "True"\n'
    "   else:\n"
    '       return "False"'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\ntext = "The tumor size(s) of the HCC patient: {text}"'
    '\nassert (HCC_staging(text) == "{label}")\n'
)

CODE_PROMPT_JS = (
    "const assert = require('assert');\n\n"
    "function HCCStaging(text) {{\n"
    "    /*\n"
    "    Classify the patient according to Milan criteria based on tumor size(s) in the text.\n\n"
    "    Milan criteria:\n"
    "        - One single tumor with its diameter <= 5 cm,\n"
    "        - Alternatively, up to 3 tumors, each with its diameter <= 3 cm.\n\n"
    "    Important notes on tumor size interpretation:\n"
    "        - Tumor size (or diameter) may be reported in millimeters (mm) or centimeters (cm).\n"
    "        - It may appear as:\n"
    '            * A single value (e.g., "25 mm"),\n'
    '            * Two dimensions (e.g., "25 x 20 mm"), or\n'
    '            * Three dimensions (e.g., "2.5 x 2.0 x 1.8 cm").\n'
    "        - In all cases, the diameter is defined as the largest single dimension."
    "        - If a patient has multiple tumors, their sizes will be listed together as a comma-separated text, "
    "    e.g., '5 mm, 30 mm, 3x4x5 mm'.\n\n"
    "    This function reads the provided `text` describing the patient's tumor size(s),\n"
    "    and classify whether the patient meets the Milan criteria.\n"
    '    If the patient is within the criteria, return "True". If not, return "False".\n'
    "    Args:\n"
    "        - text (string): The input text of the HCC patient reporting tumor size(s).\n"
    "    Returns:\n"
    '        - string: "True" if the patient meets the Milan criteria, "False" otherwise.\n'
    "    */\n"
    "    // TODO: Implement Milan staging logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'const text = "The tumor size(s) of the HCC patient: {text}";\n'
    'assert.strictEqual(HCCStaging(text), "'
)
CODE_SHOT_PROMPT_JS = (
    "\n// Example test case"
    '\nconst text = "The tumor size(s) of the HCC patient: {text}";'
    '\nassert.strictEqual(HCCStaging(text), "{label}");\n'
)

CODE_PROMPT_CPP = (
    "#include <cassert>\n"
    "#include <string>\n\n"
    "std::string HCC_staging(const std::string& text) {{\n"
    "    /*\n"
    "    Classify the patient according to Milan criteria based on tumor size(s) in the text.\n\n"
    "    Milan criteria:\n"
    "        - One single tumor with its diameter <= 5 cm,\n"
    "        - Alternatively, up to 3 tumors, each with its diameter <= 3 cm.\n\n"
    "    Important notes on tumor size interpretation:\n"
    "        - Tumor size (or diameter) may be reported in millimeters (mm) or centimeters (cm).\n"
    "        - It may appear as:\n"
    '            * A single value (e.g., "25 mm"),\n'
    '            * Two dimensions (e.g., "25 x 20 mm"), or\n'
    '            * Three dimensions (e.g., "2.5 x 2.0 x 1.8 cm").\n'
    "        - In all cases, the diameter is defined as the largest single dimension."
    "        - If a patient has multiple tumors, their sizes will be listed together as a comma-separated text, "
    "    e.g., '5 mm, 30 mm, 3x4x5 mm'.\n\n"
    "    This function reads the provided `text` describing the patient's tumor size(s),\n"
    "    and classify whether the patient meets the Milan criteria.\n"
    '    If the patient is within the criteria, return "True". If not, return "False".\n'
    "    Args:\n"
    "        - text (std::string): The input text of the HCC patient reporting tumor size(s).\n"
    "    Returns:\n"
    '        - std::string: "True" if the patient meets the Milan criteria, "False" otherwise.\n'
    "    */\n"
    "    // TODO: Implement Milan staging logic\n"
    "}}\n\n"
    "{few_shot_examples}"
    "// Test case for inference\n"
    'std::string text = "The tumor size(s) of the HCC patient: {text}";\n'
    'assert(HCC_staging(text) == "'
)
CODE_SHOT_PROMPT_CPP = (
    "\n// Example test case"
    '\nstd::string text = "The tumor size(s) of the HCC patient: {text}";'
    '\nassert(HCC_staging(text) == "{label}");\n'
)

CODE_PROMPT_RUBY = (
    "def HCC_staging(text)\n"
    "    # Classify the patient according to Milan criteria based on tumor size(s) in the text.\n\n"
    "    # Milan criteria:\n"
    "        - One single tumor with its diameter <= 5 cm,\n"
    "        - Alternatively, up to 3 tumors, each with its diameter <= 3 cm.\n\n"
    "    # Important notes on tumor size interpretation:\n"
    "        - Tumor size (or diameter) may be reported in millimeters (mm) or centimeters (cm).\n"
    "        - It may appear as:\n"
    '            * A single value (e.g., "25 mm"),\n'
    '            * Two dimensions (e.g., "25 x 20 mm"), or\n'
    '            * Three dimensions (e.g., "2.5 x 2.0 x 1.8 cm").\n'
    "        - In all cases, the diameter is defined as the largest single dimension."
    "        - If a patient has multiple tumors, their sizes will be listed together as a comma-separated text, "
    "    e.g., '5 mm, 30 mm, 3x4x5 mm'.\n\n"
    "    # This function reads the provided `text` describing the patient's tumor size(s),\n"
    "    # and classify whether the patient meets the Milan criteria.\n"
    '    # If the patient is within the criteria, return "True". If not, return "False".\n'
    "    # Args:\n"
    "        - text (string): The input text of the HCC patient reporting tumor size(s).\n"
    "    # Returns:\n"
    '        - string: "True" if the patient meets the Milan criteria, "False" otherwise.\n'
    "    # TODO: Implement Milan staging logic\n"
    "end\n\n"
    "{few_shot_examples}"
    "# Test case for inference\n"
    'text = "The tumor size(s) of the HCC patient: {text}"\n'
    'raise "Assertion failed" unless (HCC_staging(text) == "'
)
CODE_SHOT_PROMPT_RUBY = (
    "\n# Example test case"
    '\ntext = "The tumor size(s) of the HCC patient: {text}"'
    '\nraise "Assertion failed" unless (HCC_staging(text) == "{label}")\n'
)


def get_nl_prompt(example: str, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                text=e["TSIZE"].replace("\n", ", "), label=str(e["label"])
            )
    else:
        few_shot_str = ""
    return NL_PROMPT.format(
        text=example["TSIZE"].replace("\n", ", "), few_shot_examples=few_shot_str
    )


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    implement: bool = False,
    model: str = None,
    language: str = "python",
) -> str:
    """Get the code prompt."""
    _TYPING = ' -> Literal["True", "False"]'
    code_prompt, code_shot_prompt = CODE_PROMPT, CODE_SHOT_PROMPT
    if language.lower() in "javascript" or language.lower() in "js":
        code_prompt, code_shot_prompt = CODE_PROMPT_JS, CODE_SHOT_PROMPT_JS
    elif language.lower() == "cpp":
        code_prompt, code_shot_prompt = CODE_PROMPT_CPP, CODE_SHOT_PROMPT_CPP
    elif language.lower() == "ruby":
        code_prompt, code_shot_prompt = CODE_PROMPT_RUBY, CODE_SHOT_PROMPT_RUBY
    if few_shot_examples:
        few_shot_str = ""
        for e in few_shot_examples:
            few_shot_str += code_shot_prompt.format(
                text=e["TSIZE"].replace("\n", ", "), label=str(e["label"])
            )
    else:
        few_shot_str = ""
    if language.lower() in ["python", "py"]:
        prompt = code_prompt.format(
            text=example["TSIZE"].replace("\n", ", "),
            few_shot_examples=few_shot_str + "\n",
            typing=_TYPING if type_hint else "",
            body=BODY if implement else "pass",
        )
    else:
        prompt = code_prompt.format(
            text=example["TSIZE"].replace("\n", ", "),
            few_shot_examples=few_shot_str + "\n",
        )

    # add special tokens if necessary for different code LLMs
    if "codegemma" in model.lower() or "qwen" in model.lower():
        return "<|fim_prefix|>" + prompt + '<|fim_suffix|>")\n<|fim_middle|>'
    if "deepseek-coder" in model.lower() and "v2" in model.lower():
        return "<|fim_begin|>" + prompt + '<|fim_hole|>")\n<|fim_end|>'
    # for openai and deepseek-v3
    return prompt

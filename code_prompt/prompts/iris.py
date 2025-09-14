NL_PROMPT = (
    "You are a data annotator for species classification of iris flowers. "
    "Your task is to read the following rule for classification and the numerical "
    "features of the iris flower, and classify the species into either "
    "setosa, versicolor, or virginica based on the rule.\n"
    'Please answer directly with only one of: "setosa", "versicolor", or "virginica" as output.'
    "Rule: \n"
    'If Petal Width < 0.8 cm, then the species is "setosa".\n'
    'If Petal Width >= 0.8 cm and < 1.8 cm, then the species is "versicolor".\n'
    'If Petal Width >= 1.8 cm, then the species is "virginica".\n\n'
    "\n{few_shot_examples}"
    "\nHere is the flower to classify:"
    "\nInput: sepal length: {sepal_length} cm, sepal width: {sepal_width} cm, "
    "petal length: {petal_length} cm, petal width: {petal_width} cm"
    "\nOutput: "
)
NL_SHOT_PROMPT = (
    "\nInput: sepal length: {sepal_length} cm, sepal width: {sepal_width} cm, "
    "petal length: {petal_length} cm, petal width: {petal_width} cm"
    "\nOutput: {label}\n"
)

CODE_PROMPT = (
    "def classify_iris_species(text: str){typing}:\n"
    '   """Classify the species of the iris flower based on its features and the following rule.\n'
    '   Rule: If Petal Width < 0.8 cm, then the species is "setosa".\n'
    '         If Petal Width >= 0.8 cm and < 1.8 cm, then the species is "versicolor".\n'
    '         If Petal Width >= 1.8 cm, then the species is "virginica".\n'
    "   Args:\n"
    "       - text (str): A string containing the features of the iris flower in the format:\n"
    '       "sepal length: X cm, sepal width: Y cm, petal length: Z cm, petal width: W cm".\n'
    "   Returns:\n"
    '       Literal["setosa", "versicolor", "virginica"]: The species label of the iris flower.\n'
    '   """\npass\n\n'
    "{few_shot_examples}"
    "# Test case for inference\n"
    'text = "sepal length: {sepal_length} cm, sepal width: {sepal_width} cm, '
    'petal length: {petal_length} cm, petal width: {petal_width} cm"\n'
    'assert (classify_iris_species(text) == "'
)
CODE_SHOT_PROMPT = (
    "\n# Example test case"
    '\ntext = "sepal length: {sepal_length} cm, sepal width: {sepal_width} cm, '
    'petal length: {petal_length} cm, petal width: {petal_width} cm"'
    '\nassert (classify_iris_species(text) == "{label}")\n'
)


def relabel_iris(example: dict) -> str:
    """Relabel the iris species based on the rule."""
    petal_width = example["PetalWidthCm"]
    if petal_width < 0.8:
        label = "setosa"
    elif petal_width < 1.8:
        label = "versicolor"
    else:
        label = "virginica"
    return label


def get_nl_prompt(example: dict, few_shot_examples: list[dict] = None) -> str:
    """Get the natural language prompt."""
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += NL_SHOT_PROMPT.format(
                sepal_length=example["SepalLengthCm"],
                sepal_width=example["SepalWidthCm"],
                petal_length=example["PetalLengthCm"],
                petal_width=example["PetalWidthCm"],
                label=relabel_iris(example),
            )
    else:
        few_shot_str = ""
    return NL_PROMPT.format(
        few_shot_examples=few_shot_str,
        sepal_length=example["SepalLengthCm"],
        sepal_width=example["SepalWidthCm"],
        petal_length=example["PetalLengthCm"],
        petal_width=example["PetalWidthCm"],
    )


def get_code_prompt(
    example: dict,
    few_shot_examples: list[dict] = None,
    type_hint: bool = True,
    model: str = None,
) -> str:
    """Get the code prompt."""
    _TYPING = ' -> Literal["setosa", "versicolor", "virginica"]'
    if few_shot_examples:
        few_shot_str = ""
        for example in few_shot_examples:
            few_shot_str += CODE_SHOT_PROMPT.format(
                sepal_length=example["SepalLengthCm"],
                sepal_width=example["SepalWidthCm"],
                petal_length=example["PetalLengthCm"],
                petal_width=example["PetalWidthCm"],
                label=relabel_iris(example),
            )
    else:
        few_shot_str = ""
    prompt = CODE_PROMPT.format(
        sepal_length=example["SepalLengthCm"],
        sepal_width=example["SepalWidthCm"],
        petal_length=example["PetalLengthCm"],
        petal_width=example["PetalWidthCm"],
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



NL_PROMPT = """You are a data annotator for text classification.
Your task is to read the following text and classify the text into either positive or negative based on its sentiment.
Please provide your classification in the format: "label: <label>" where <label> is either "POSITIVE" or "NEGATIVE".

Here is the text to classify:
{text}
"""

CODE_PROMPT = '''
def classify_sentiment(text: str) -> Literal["POSITIVE", "NEGATIVE"]:
    """
    Classify the sentiment of the given text as either POSITIVE or NEGATIVE.
    
    Args:
        text (str): The text to classify.
    
    Returns:
        Literal["POSITIVE", "NEGATIVE"]: The sentiment label of the text.
    """
    pass
    

# last test case for inference
assert (classify_sentiment({text}) ==
'''
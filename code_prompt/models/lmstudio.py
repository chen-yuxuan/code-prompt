import ast

from openai import OpenAI


MODEL = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"


def get_response(
    message: str, model: str = MODEL, completion: bool = False
) -> dict | str:
    """Get the response from the LLM model."""
    url = (
        "http://localhost:1234/v1/completions"
        if completion
        else "http://localhost:1234/v1"
    )
    client = OpenAI(base_url=url, api_key="lm-studio")
    response = (
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message},
            ],
            temperature=0,
        )
        .choices[0]
        .message.content
    )

    try:
        response = ast.literal_eval(response)
    except Exception:
        response = str(response)
    return response

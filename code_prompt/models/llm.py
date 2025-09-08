import logging
import os
from typing import Optional

from openai import OpenAI
from google import genai
from google.genai import types
import vllm

CLAUDE_API_KEY = "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"


def get_client(
    model: str,
    api_key: Optional[str] = None,
):
    """Provide an LLM client for prompting.
    Supports models hosted on OpenAI, Gemini and vLLM.
    """
    if model.lower() == "chatgpt" or model.lower().startswith("gpt-"):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI(api_key=api_key)
    if model.lower().startswith("gemini"):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        return genai.Client(api_key=api_key)
    return vllm.LLM(model=model)


def get_response(
    prompt: str,
    client: str | OpenAI | genai.Client | vllm.LLM,
    model: str,
    api_key: Optional[str] = None,
    enforce_code_prompt: bool = False,
    max_tokens: int = None,
) -> str:
    """Get response from the specified LLM model."""
    if isinstance(client, str):
        client = get_client(client, api_key=api_key)
    if isinstance(client, OpenAI):
        if enforce_code_prompt:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                suffix='"',
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].text.strip()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
    elif isinstance(client, genai.Client):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text.strip()
    elif isinstance(client, vllm.LLM):
        sampling_params = vllm.SamplingParams(temperature=0.0)
        response = client.generate(prompt, sampling_params)[0].outputs[0]
        return response.text.strip()
    else:
        logging.error("Unsupported model.")

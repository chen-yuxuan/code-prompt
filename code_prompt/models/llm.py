import logging
import os
from typing import Optional

from openai import OpenAI
from google import genai
from google.genai import types
import vllm


def get_client(
    model: str,
    api_key: Optional[str] = None,
    beta: bool = False,
):
    """Provide an LLM client for prompting.
    Supports models hosted on OpenAI, Gemini and vLLM.
    """
    if model.lower() == "chatgpt" or model.lower().startswith("gpt-"):
        if not api_key:
            api_key = "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"
        return OpenAI(api_key=api_key)
    if model.lower().startswith("gemini"):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        return genai.Client(api_key=api_key)
    if "deepseek-v3" in model.lower():
        if not api_key:
            api_key = "sk-79006f408908450b987eccd55c4e91e4"
        base_url = (
            "https://api.deepseek.com" if not beta else "https://api.deepseek.com/beta"
        )
        return OpenAI(api_key=api_key, base_url=base_url)
    return vllm.LLM(model=model)


def get_response(
    prompt: str,
    client: str | OpenAI | genai.Client | vllm.LLM,
    model: str,
    api_key: Optional[str] = None,
    enforce_code_prompt: bool = False,
    max_tokens: int = 16,
) -> str:
    """Get response from the specified LLM model."""
    if isinstance(client, str):
        client = get_client(client, api_key=api_key)
    if isinstance(client, OpenAI):  # OpenAI or DeepSeek
        if enforce_code_prompt:
            response = client.completions.create(
                model=model if "deepseek" not in model.lower() else "deepseek-chat",
                prompt=prompt,
                suffix='"',
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].text.strip()
        else:
            if "gpt-3.5-turbo-instruct" in model:
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return response.choices[0].text.strip()
            else:
                response = client.chat.completions.create(
                    model=(
                        model if "deepseek" not in model.lower() else "deepseek-chat"
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
    elif isinstance(client, genai.Client):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=max_tokens
            ),
        )
        return response.text.strip()
    elif isinstance(client, vllm.LLM):
        sampling_params = vllm.SamplingParams(temperature=0.0)
        response = client.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[
            0
        ]
        return response.text.strip()
    else:
        logging.error("Unsupported model.")

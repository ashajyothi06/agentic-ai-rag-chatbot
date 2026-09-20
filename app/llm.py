import json
import re

from openai import OpenAI

from app.config import Settings


class LLMClient:
    """Small OpenAI-compatible client wrapper for Qwen or Grok."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.llm_provider
        self.model = settings.active_llm_model
        self.client = OpenAI(
            api_key=settings.active_llm_api_key,
            base_url=settings.active_llm_base_url,
            timeout=45.0,
            max_retries=2,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    def json_chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        raw = self.chat(system_prompt, user_prompt, temperature=0.0)
        return _extract_json_object(raw)


def _extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return a JSON object: {text[:300]}")

    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM JSON response was not an object.")
    return value

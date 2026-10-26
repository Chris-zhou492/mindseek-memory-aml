import json
import os
from typing import Any
from urllib.request import Request, urlopen

from memory_service.profile_prompt import USER_PROFILE_EXTRACTION_PROMPT

PROFILE_SECTIONS = (
    "preferences",
    "personal_facts",
    "goals",
    "health_wellness",
    "work_education",
    "other",
)


def _complete_profile_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    user_messages: list[str],
    system_prompt: str = USER_PROFILE_EXTRACTION_PROMPT,
) -> str:
    if not api_key:
        raise RuntimeError("Profile API key is required")
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Input:\n" + "\n".join(user_messages)},
        ],
    }
    if (
        os.environ.get("PROFILE_LLM_ENABLE_THINKING", "").lower() == "false"
        and not model.casefold().startswith("gpt-")
    ):
        payload["enable_thinking"] = False
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"]


def extract_profile(messages: list[dict[str, Any]]) -> dict[str, list[str]]:
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    if not user_messages:
        return {}

    api_key = (
        os.environ.get("PROFILE_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY, PROFILE_LLM_API_KEY, or DEEPSEEK_API_KEY is required for challenge Add")

    content = _complete_profile_json(
        api_key=api_key,
        base_url=(
            os.environ.get("PROFILE_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("DEEPSEEK_API_BASE")
            or "https://api.openai.com/v1"
        ),
        model=os.environ.get("CHALLENGE_PROFILE_MODEL") or os.environ.get("PROFILE_LLM_MODEL") or "gpt-4o-mini",
        user_messages=user_messages,
    )
    if not content:
        raise ValueError("Profile model returned an empty response")
    payload = json.loads(content)
    if not isinstance(payload, dict) or not isinstance(payload.get("profile"), dict):
        raise ValueError("Profile model returned an invalid profile")

    profile: dict[str, list[str]] = {}
    for section in PROFILE_SECTIONS:
        values = payload["profile"].get(section, [])
        if not isinstance(values, list):
            raise ValueError(f"Profile section {section} must be a list")
        seen: set[str] = set()
        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            item = value.strip()
            if item and item.casefold() not in seen:
                seen.add(item.casefold())
                cleaned.append(item)
        profile[section] = cleaned[:10]
    return profile

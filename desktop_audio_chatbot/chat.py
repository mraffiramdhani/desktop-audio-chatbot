"""OpenAI-compatible chat completions client."""

from __future__ import annotations

from datetime import datetime

import httpx

from .config import Settings

SYSTEM_PROMPT = """You are a desktop audio assistant. The user is listening to system/desktop audio (a meeting, video, stream, or other speech playing on their computer). You receive a rolling transcript of what was heard.

Use that transcript as the primary source of truth when answering.
- If they ask "what did they just say?" or similar, quote/paraphrase the latest transcript.
- If they ask to summarize the last N minutes, only use transcript in that window (timestamps are local time).
- If the transcript is empty or does not contain enough information, say so clearly instead of inventing content.
- Be concise and helpful. Prefer bullet points for summaries and action items.
"""


def completions_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("LLM base URL is empty")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def build_messages(
    user_message: str,
    history: list[dict],
    transcript: str,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    now = now or datetime.now().astimezone()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    if transcript.strip():
        context = (
            f"Current local time: {stamp}\n\n"
            "Recent desktop-audio transcript (oldest first):\n"
            f"{transcript.strip()}"
        )
    else:
        context = (
            f"Current local time: {stamp}\n\n"
            "Recent desktop-audio transcript: (empty — nothing has been transcribed yet)."
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context},
    ]
    for item in history[-20:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message.strip()})
    return messages


class ChatClient:
    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout

    def complete(self, settings: Settings, messages: list[dict[str, str]]) -> str:
        url = completions_url(settings.llm_base_url)
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key.strip():
            headers["Authorization"] = f"Bearer {settings.llm_api_key.strip()}"

        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": 0.3,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise ChatError(f"Could not reach the LLM API at {url}: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:800]
            raise ChatError(f"LLM API returned HTTP {response.status_code}: {detail}")

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatError("LLM API returned an unexpected response body") from exc

        text = (content or "").strip()
        if not text:
            raise ChatError("LLM API returned an empty message")
        return text


class ChatError(RuntimeError):
    pass

from datetime import datetime

from desktop_audio_chatbot.chat import ChatClient, ChatError, build_messages, completions_url


def test_completions_url_normalization():
    assert completions_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"
    assert completions_url("http://localhost:11434") == "http://localhost:11434/v1/chat/completions"
    assert (
        completions_url("http://localhost:1234/v1/chat/completions")
        == "http://localhost:1234/v1/chat/completions"
    )


def test_build_messages_includes_transcript_and_history():
    messages = build_messages(
        "What did they just say?",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "[12:01:00] The launch is delayed until Friday.",
        now=datetime(2026, 9, 21, 12, 2, 0),
    )
    assert messages[0]["role"] == "system"
    assert "delayed until Friday" in messages[1]["content"]
    assert messages[-1] == {"role": "user", "content": "What did they just say?"}
    assert messages[-3]["content"] == "hi"


def test_build_messages_empty_transcript():
    messages = build_messages("Summarize", [], "")
    assert "empty" in messages[1]["content"].lower()


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload)

    def json(self):
        return self._payload


def test_chat_client_parses_message(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _FakeResponse(
                payload={"choices": [{"message": {"content": "  They said hello.  "}}]}
            )

    monkeypatch.setattr("desktop_audio_chatbot.chat.httpx.Client", FakeClient)
    from desktop_audio_chatbot.config import Settings

    client = ChatClient()
    reply = client.complete(
        Settings(llm_base_url="http://example.local/v1", llm_api_key="sk-test", llm_model="m"),
        [{"role": "user", "content": "hi"}],
    )
    assert reply == "They said hello."
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_chat_client_http_error(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            return _FakeResponse(status_code=401, payload={}, text="nope")

    monkeypatch.setattr("desktop_audio_chatbot.chat.httpx.Client", FakeClient)
    from desktop_audio_chatbot.config import Settings

    try:
        ChatClient().complete(Settings(llm_base_url="http://x/v1"), [{"role": "user", "content": "hi"}])
        assert False, "expected ChatError"
    except ChatError as exc:
        assert "401" in str(exc)

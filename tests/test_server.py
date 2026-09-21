from fastapi.testclient import TestClient

from desktop_audio_chatbot.server import app, engine


def test_health_and_settings_hide_key():
    with TestClient(app) as client:
        assert client.get("/api/health").json()["ok"] is True
        settings = client.get("/api/settings").json()
        assert settings["llm_api_key"] == ""
        assert settings["llm_api_key_set"] is True
        assert settings["llm_model"] == "test-model"


def test_devices_endpoint_shape():
    with TestClient(app) as client:
        data = client.get("/api/devices").json()
        assert "devices" in data
        assert "loopback_available" in data
        assert "help" in data
        assert "steps" in data["help"]


def test_transcript_roundtrip_and_chat_uses_it(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            context = json["messages"][1]["content"]
            user = json["messages"][-1]["content"]
            assert "meeting moved to Thursday" in context
            assert "just say" in user.lower()

            class Resp:
                status_code = 200
                text = ""

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "They said the meeting moved to Thursday."
                                }
                            }
                        ]
                    }

            return Resp()

    monkeypatch.setattr("desktop_audio_chatbot.chat.httpx.Client", FakeClient)

    with TestClient(app) as client:
        added = client.post(
            "/api/transcript",
            json={"text": "The meeting moved to Thursday.", "duration": 3},
        ).json()
        assert added["ok"] is True
        listed = client.get("/api/transcript").json()["segments"]
        assert listed[-1]["text"] == "The meeting moved to Thursday."

        chat = client.post(
            "/api/chat",
            json={"message": "What did they just say?", "context_seconds": 120},
        ).json()
        assert chat["ok"] is True
        assert "Thursday" in chat["reply"]


def test_listen_unknown_device():
    with TestClient(app) as client:
        result = client.post("/api/listen/start", json={"device_id": "missing:device"}).json()
        assert result["ok"] is False
        assert "no longer available" in result["error"].lower() or "available" in result["error"].lower()


def test_save_settings_keeps_existing_key():
    with TestClient(app) as client:
        saved = client.post(
            "/api/settings",
            json={"llm_model": "local-model", "llm_api_key": ""},
        ).json()
        assert saved["llm_model"] == "local-model"
        assert saved["llm_api_key_set"] is True
        assert engine.settings.llm_api_key == "test-key"


def test_index_serves_ui():
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Live transcript" in page.text
        assert "Start listening" in page.text

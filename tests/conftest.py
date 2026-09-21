from __future__ import annotations

import pytest

from desktop_audio_chatbot.config import load_settings
from desktop_audio_chatbot.server import engine


@pytest.fixture(autouse=True)
def isolate_app(tmp_path, monkeypatch):
    monkeypatch.setattr("desktop_audio_chatbot.config.APP_DIR", tmp_path)
    monkeypatch.setattr("desktop_audio_chatbot.config.SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr("desktop_audio_chatbot.config.MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    engine.settings = load_settings()
    engine.store.clear()
    engine.clear_chat()
    engine.last_error = ""
    engine.last_rms = 0.0
    if engine.listening:
        engine.stop_listening()
    yield
    if engine.listening:
        engine.stop_listening()

from desktop_audio_chatbot.config import Settings, apply_updates, load_settings, save_settings


def test_public_dict_masks_key():
    settings = Settings(llm_api_key="sk-abcdefghijklmnop")
    public = settings.public_dict()
    assert public["llm_api_key"] == ""
    assert public["llm_api_key_set"] is True
    assert public["llm_api_key_hint"] == "sk-a…mnop"


def test_empty_key_keeps_previous():
    settings = Settings(llm_api_key="keep-me", llm_model="a")
    apply_updates(settings, {"llm_api_key": "", "llm_model": "b"})
    assert settings.llm_api_key == "keep-me"
    assert settings.llm_model == "b"


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("desktop_audio_chatbot.config.APP_DIR", tmp_path)
    monkeypatch.setattr("desktop_audio_chatbot.config.SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    settings = load_settings()
    settings.llm_model = "saved-model"
    save_settings(settings)
    loaded = load_settings()
    assert loaded.llm_model == "saved-model"
    assert loaded.llm_api_key == "from-env"

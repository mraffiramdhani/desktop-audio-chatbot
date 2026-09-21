"""Load settings from environment, optional .env, and a local JSON file."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path.home() / ".desktop-audio-chatbot"
SETTINGS_PATH = APP_DIR / "settings.json"
MODELS_DIR = APP_DIR / "models"

# Load a repo-local .env if present (does not override existing env vars).
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def _env(name: str, *aliases: str, default: str = "") -> str:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


@dataclass
class Settings:
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_language: str = ""
    host: str = "127.0.0.1"
    port: int = 8765

    def public_dict(self) -> dict:
        """JSON-safe view for the UI. Never send the raw API key."""
        data = asdict(self)
        data["llm_api_key"] = ""
        data["llm_api_key_set"] = bool(self.llm_api_key.strip())
        data["llm_api_key_hint"] = _mask_key(self.llm_api_key)
        return data


def _mask_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"


def default_settings() -> Settings:
    return Settings(
        llm_base_url=_env("LLM_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1"),
        llm_api_key=_env("LLM_API_KEY", "OPENAI_API_KEY"),
        llm_model=_env("LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini"),
        whisper_model=_env("WHISPER_MODEL", default="base"),
        whisper_device=_env("WHISPER_DEVICE", default="auto"),
        whisper_language=_env("WHISPER_LANGUAGE"),
        host=_env("HOST", default="127.0.0.1"),
        port=int(_env("PORT", default="8765") or "8765"),
    )


def _read_saved() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_settings() -> Settings:
    settings = default_settings()
    saved = _read_saved()
    valid = {f.name for f in fields(Settings)}
    for key, value in saved.items():
        if key not in valid or value is None:
            continue
        if key == "port":
            try:
                settings.port = int(value)
            except (TypeError, ValueError):
                continue
        elif key == "llm_api_key" and str(value).strip() == "":
            continue
        else:
            setattr(settings, key, value)
    return settings


def save_settings(settings: Settings) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(SETTINGS_PATH, 0o600)
    except OSError:
        pass


def apply_updates(settings: Settings, updates: dict) -> Settings:
    """Merge UI updates. Empty API key means 'keep existing key'."""
    if "llm_base_url" in updates and updates["llm_base_url"] is not None:
        settings.llm_base_url = str(updates["llm_base_url"]).strip() or settings.llm_base_url
    if "llm_model" in updates and updates["llm_model"] is not None:
        settings.llm_model = str(updates["llm_model"]).strip() or settings.llm_model
    if "whisper_model" in updates and updates["whisper_model"] is not None:
        settings.whisper_model = str(updates["whisper_model"]).strip() or settings.whisper_model
    if "whisper_device" in updates and updates["whisper_device"] is not None:
        settings.whisper_device = str(updates["whisper_device"]).strip() or settings.whisper_device
    if "whisper_language" in updates and updates["whisper_language"] is not None:
        settings.whisper_language = str(updates["whisper_language"]).strip()
    if "llm_api_key" in updates:
        key = updates["llm_api_key"]
        if key is not None and str(key).strip() != "":
            settings.llm_api_key = str(key).strip()
    return settings

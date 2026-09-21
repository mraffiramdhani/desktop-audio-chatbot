"""Glue: capture → transcribe → transcript store, plus chat history."""

from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np

from .audio import AudioCapture, CaptureDevice, CaptureError, find_device, list_capture_devices
from .chat import ChatClient, ChatError, build_messages
from .config import Settings
from .transcribe import Transcriber
from .transcript import TranscriptStore

EventCallback = Callable[[dict], None]


class AppEngine:
    def __init__(self, settings: Settings, emit: EventCallback) -> None:
        self.settings = settings
        self.emit = emit
        self.store = TranscriptStore()
        self.transcriber = Transcriber()
        self.chat = ChatClient()
        self.history: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._work: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=6)
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_error = ""
        self.last_rms = 0.0
        self._loading_model = False
        self.capture = AudioCapture(
            on_segment=self._enqueue_audio,
            on_level=self._on_level,
            on_error=self._on_capture_error,
        )

    @property
    def listening(self) -> bool:
        return bool(self.capture.listening)

    def status(self) -> dict:
        device = self.capture.device
        return {
            "listening": self.listening,
            "device": device.as_dict() if device else None,
            "rms": self.last_rms,
            "error": self.last_error,
            "model_loading": self._loading_model,
            "whisper_model": self.settings.whisper_model,
            "transcript_count": len(self.store.all()),
        }

    def start_listening(self, device_id: str) -> CaptureDevice:
        devices = list_capture_devices()
        device = find_device(device_id, devices)
        if device is None:
            raise CaptureError(
                "That audio device is no longer available. Refresh the device list and try again."
            )
        self.last_error = ""
        self._loading_model = True
        self.emit({"type": "status", **self.status()})
        try:
            self.transcriber.ensure_loaded(self.settings)
        except Exception as exc:
            self._loading_model = False
            raise CaptureError(
                "Could not load the Whisper model "
                f"“{self.settings.whisper_model}”: {exc}"
            ) from exc
        self._loading_model = False
        self._ensure_worker()
        self.capture.start(device)
        self.emit({"type": "status", **self.status()})
        return device

    def stop_listening(self) -> None:
        self.capture.stop()
        self.emit({"type": "status", **self.status()})

    def update_settings(self, settings: Settings) -> None:
        old_model = (self.settings.whisper_model, self.settings.whisper_device)
        self.settings = settings
        if old_model != (settings.whisper_model, settings.whisper_device):
            self.transcriber.unload()

    def clear_transcript(self) -> None:
        self.store.clear()
        self.emit({"type": "transcript_clear"})

    def clear_chat(self) -> None:
        with self._lock:
            self.history.clear()

    def chat_reply(self, user_message: str, context_seconds: float = 300.0) -> str:
        user_message = (user_message or "").strip()
        if not user_message:
            raise ChatError("Message is empty.")
        if not self.settings.llm_base_url.strip():
            raise ChatError("Set an LLM base URL in Settings.")
        looks_openai = "openai.com" in self.settings.llm_base_url.lower()
        if looks_openai and not self.settings.llm_api_key.strip():
            raise ChatError("Set your LLM API key in Settings (or in the .env file).")

        transcript = self.store.context_text(max_seconds=context_seconds)
        with self._lock:
            history = list(self.history)
        messages = build_messages(user_message, history, transcript)
        reply = self.chat.complete(self.settings, messages)
        with self._lock:
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": reply})
        return reply

    def shutdown(self) -> None:
        self.stop_listening()
        self._stop.set()
        try:
            self._work.put_nowait(None)
        except queue.Full:
            pass

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._transcribe_loop, name="stt-worker", daemon=True)
        self._worker.start()

    def _enqueue_audio(self, audio: np.ndarray) -> None:
        try:
            self._work.put_nowait(audio)
        except queue.Full:
            try:
                _ = self._work.get_nowait()
            except queue.Empty:
                pass
            try:
                self._work.put_nowait(audio)
            except queue.Full:
                pass

    def _transcribe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._work.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                text, language = self.transcriber.transcribe(item, self.settings)
            except Exception as exc:
                self.last_error = f"Transcription failed: {exc}"
                self.emit({"type": "error", "message": self.last_error})
                continue
            if not text:
                continue
            if sum(ch.isalpha() for ch in text) < 4:
                continue
            duration = float(item.size) / 16000.0
            segment = self.store.add(text, duration=duration, language=language)
            self.emit({"type": "transcript", "segment": segment.as_dict()})

    def _on_level(self, rms: float) -> None:
        self.last_rms = rms
        self.emit({"type": "level", "rms": rms})

    def _on_capture_error(self, message: str) -> None:
        self.last_error = message
        self.emit({"type": "error", "message": message})
        self.emit({"type": "status", **self.status()})

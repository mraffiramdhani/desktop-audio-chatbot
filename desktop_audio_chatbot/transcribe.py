"""Local speech-to-text via faster-whisper."""

from __future__ import annotations

import threading

import numpy as np

from .config import MODELS_DIR, Settings


class Transcriber:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._loaded_key: tuple | None = None

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._loaded_key = None

    def ensure_loaded(self, settings: Settings) -> None:
        device, compute = _resolve_device(settings.whisper_device)
        key = (settings.whisper_model, device, compute)
        with self._lock:
            if self._model is not None and self._loaded_key == key:
                return
            from faster_whisper import WhisperModel

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self._model = WhisperModel(
                settings.whisper_model,
                device=device,
                compute_type=compute,
                download_root=str(MODELS_DIR),
            )
            self._loaded_key = key

    def transcribe(self, audio: np.ndarray, settings: Settings) -> tuple[str, str]:
        self.ensure_loaded(settings)
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size < 400:  # ~25 ms at 16 kHz
            return "", ""
        peak = float(np.max(np.abs(samples)))
        if peak > 1.0:
            samples = samples / peak
        language = settings.whisper_language.strip() or None
        with self._lock:
            assert self._model is not None
            segments, info = self._model.transcribe(
                samples,
                language=language,
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            parts = []
            for seg in segments:
                piece = getattr(seg, "text", "").strip()
                if not piece:
                    continue
                if float(getattr(seg, "no_speech_prob", 0.0) or 0.0) > 0.65:
                    continue
                parts.append(piece)
            detected = getattr(info, "language", "") or ""
        text = " ".join(parts).strip()
        return text, detected


def _resolve_device(requested: str) -> tuple[str, str]:
    requested = (requested or "auto").strip().lower()
    if requested == "cuda":
        return "cuda", "float16"
    if requested == "cpu":
        return "cpu", "int8"
    # auto
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"

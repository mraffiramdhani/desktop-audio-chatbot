"""Timestamped transcript buffer used as LLM context."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass
class TranscriptSegment:
    id: str
    text: str
    t_start: float
    t_end: float
    language: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class TranscriptStore:
    def __init__(self, max_seconds: float = 30 * 60) -> None:
        self.max_seconds = max_seconds
        self._segments: list[TranscriptSegment] = []
        self._lock = threading.Lock()

    def add(
        self,
        text: str,
        duration: float,
        language: str = "",
        t_end: float | None = None,
    ) -> TranscriptSegment:
        text = " ".join(text.split()).strip()
        end = time.time() if t_end is None else t_end
        start = end - max(duration, 0.0)
        segment = TranscriptSegment(
            id=uuid.uuid4().hex[:12],
            text=text,
            t_start=start,
            t_end=end,
            language=language,
        )
        with self._lock:
            self._segments.append(segment)
            self._prune_locked(time.time())
        return segment

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()

    def all(self) -> list[TranscriptSegment]:
        with self._lock:
            return list(self._segments)

    def recent(self, max_seconds: float = 300.0, now: float | None = None) -> list[TranscriptSegment]:
        now = time.time() if now is None else now
        cutoff = now - max_seconds
        with self._lock:
            return [s for s in self._segments if s.t_end >= cutoff]

    def context_text(self, max_seconds: float = 300.0, now: float | None = None) -> str:
        segments = self.recent(max_seconds=max_seconds, now=now)
        if not segments:
            return ""
        lines = []
        for seg in segments:
            stamp = time.strftime("%H:%M:%S", time.localtime(seg.t_start))
            lines.append(f"[{stamp}] {seg.text}")
        return "\n".join(lines)

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.max_seconds
        self._segments = [s for s in self._segments if s.t_end >= cutoff]

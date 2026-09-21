import time

from desktop_audio_chatbot.transcript import TranscriptStore


def test_recent_context_window():
    store = TranscriptStore(max_seconds=600)
    now = 1_700_000_000.0
    store.add("old line", duration=1, t_end=now - 400)
    store.add("recent line", duration=1, t_end=now - 20)
    text = store.context_text(max_seconds=120, now=now)
    assert "recent line" in text
    assert "old line" not in text


def test_prune_old_segments():
    store = TranscriptStore(max_seconds=30)
    store.add("keep me", duration=1, t_end=time.time())
    store.add("drop me", duration=1, t_end=time.time() - 90)
    texts = [s.text for s in store.all()]
    assert texts == ["keep me"]

import numpy as np

from desktop_audio_chatbot.audio import (
    SpeechSegmenter,
    classify_kind,
    loopback_setup_help,
    resample_audio,
    to_mono,
    _parse_pactl_short,
)


def test_classify_loopback_hints():
    assert classify_kind("alsa_output.pci-0000_00_1f.3.analog-stereo.monitor") == "loopback"
    assert classify_kind("Stereo Mix (Realtek)") == "loopback"
    assert classify_kind("BlackHole 2ch") == "loopback"
    assert classify_kind("WASAPI Speakers", forced_loopback=True) == "loopback"
    assert classify_kind("Built-in Microphone") == "microphone"


def test_resample_and_mono():
    stereo = np.ones((10, 2), dtype=np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (10,)
    stretched = resample_audio(np.ones(16000, dtype=np.float32), 16000, 8000)
    assert abs(stretched.size - 8000) <= 1


def test_segmenter_emits_after_silence():
    seg = SpeechSegmenter(
        sample_rate=16000,
        rms_threshold=0.05,
        min_speech_sec=0.2,
        silence_sec=0.2,
        max_sec=5,
    )
    speech = np.ones(int(0.3 * 16000), dtype=np.float32) * 0.2
    silence = np.zeros(int(0.25 * 16000), dtype=np.float32)
    assert seg.add(speech) == []
    out = seg.add(silence)
    assert len(out) == 1
    assert out[0].size > 1000


def test_segmenter_ignores_quiet_noise():
    seg = SpeechSegmenter(sample_rate=16000, rms_threshold=0.05)
    quiet = np.ones(16000, dtype=np.float32) * 0.001
    assert seg.add(quiet) == []
    assert seg.flush() is None


def test_pactl_short_parser():
    text = "43\talsa_output.usb.analog-stereo.monitor\tPipeWire\ts32le 2ch 48000Hz\tIDLE\n"
    devices = _parse_pactl_short(text)
    assert len(devices) == 1
    assert devices[0].kind == "loopback"
    assert devices[0].id.endswith("alsa_output.usb.analog-stereo.monitor")


def test_loopback_help_mentions_platform_tools():
    win = loopback_setup_help("Windows")
    assert any("WASAPI" in step for step in win["steps"])
    linux = loopback_setup_help("Linux")
    assert any("monitor" in step for step in linux["steps"])
    mac = loopback_setup_help("Darwin")
    assert any("BlackHole" in step for step in mac["steps"])

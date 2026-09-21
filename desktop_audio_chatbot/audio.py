"""Desktop/loopback and microphone capture, plus OS-specific setup help."""

from __future__ import annotations

import json
import platform
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

try:
    import sounddevice as sd

    SD_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on PortAudio
    sd = None  # type: ignore[assignment]
    SD_ERROR = str(exc)

LOOPBACK_HINTS = (
    "monitor",
    "loopback",
    "stereo mix",
    "what u hear",
    "wave out mix",
    "blackhole",
    "soundflower",
    "vb-audio",
    "vb cable",
    "cable output",
    "cable-output",
    "sound siphon",
)

LevelCallback = Callable[[float], None]
SegmentCallback = Callable[[np.ndarray], None]
ErrorCallback = Callable[[str], None]


@dataclass
class CaptureDevice:
    id: str
    name: str
    kind: str  # loopback | microphone
    backend: str  # sounddevice | wasapi_loopback | parec | pwrecord
    hostapi: str = ""
    sd_index: int | None = None
    pulse_name: str | None = None
    channels: int = 1
    samplerate: int = 16000
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("extra", None)
        return data


def classify_kind(name: str, *, forced_loopback: bool = False) -> str:
    if forced_loopback:
        return "loopback"
    lower = name.lower()
    if any(hint in lower for hint in LOOPBACK_HINTS):
        return "loopback"
    return "microphone"


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if orig_sr <= 0 or target_sr <= 0 or orig_sr == target_sr or audio.size == 0:
        return audio
    n_out = int(round(audio.size * target_sr / orig_sr))
    if n_out < 1:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, audio).astype(np.float32)


def to_mono(block: np.ndarray) -> np.ndarray:
    data = np.asarray(block, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.reshape(-1)


class SpeechSegmenter:
    """Energy-based endpointer: emit a clip after speech followed by silence."""

    def __init__(
        self,
        sample_rate: int = 16000,
        rms_threshold: float = 0.008,
        min_speech_sec: float = 0.45,
        silence_sec: float = 0.75,
        max_sec: float = 12.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.min_speech = int(min_speech_sec * sample_rate)
        self.silence_limit = int(silence_sec * sample_rate)
        self.max_samples = int(max_sec * sample_rate)
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech = 0
        self._silence = 0
        self._in_speech = False

    def add(self, samples: np.ndarray) -> list[np.ndarray]:
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        voiced = rms >= self.rms_threshold
        finished: list[np.ndarray] = []

        if voiced:
            self._in_speech = True
            self._silence = 0
            self._speech += chunk.size
            self._buf = np.concatenate([self._buf, chunk])
        elif self._in_speech:
            self._silence += chunk.size
            self._buf = np.concatenate([self._buf, chunk])
            if self._silence >= self.silence_limit and self._speech >= self.min_speech:
                finished.append(self._buf.copy())
                self.reset()
                return finished
        else:
            return []

        if self._buf.size >= self.max_samples:
            finished.append(self._buf.copy())
            self.reset()
        return finished

    def flush(self) -> np.ndarray | None:
        if self._in_speech and self._speech >= self.min_speech and self._buf.size:
            out = self._buf.copy()
            self.reset()
            return out
        self.reset()
        return None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech = 0
        self._silence = 0
        self._in_speech = False


def _hostapi_name(index: int) -> str:
    if sd is None:
        return ""
    try:
        return str(sd.query_hostapis()[index]["name"])
    except Exception:
        return ""


def list_sounddevice_devices() -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []
    if sd is None:
        return devices
    try:
        raw = sd.query_devices()
    except Exception:
        return devices

    for index, info in enumerate(raw):
        name = str(info.get("name") or f"Device {index}")
        host = _hostapi_name(int(info.get("hostapi") or 0))
        max_in = int(info.get("max_input_channels") or 0)
        max_out = int(info.get("max_output_channels") or 0)
        rate = int(info.get("default_samplerate") or 16000)
        if max_in > 0:
            kind = classify_kind(name)
            devices.append(
                CaptureDevice(
                    id=f"sd:{index}",
                    name=name,
                    kind=kind,
                    backend="sounddevice",
                    hostapi=host,
                    sd_index=index,
                    channels=min(max_in, 2),
                    samplerate=rate,
                )
            )
        if "wasapi" in host.lower() and max_out > 0:
            devices.append(
                CaptureDevice(
                    id=f"sdloop:{index}",
                    name=f"{name} (WASAPI loopback)",
                    kind="loopback",
                    backend="wasapi_loopback",
                    hostapi=host,
                    sd_index=index,
                    channels=min(max_out, 2),
                    samplerate=rate,
                    extra={"loopback": True},
                )
            )
    return devices


def _parse_pactl_short(text: str) -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pulse_name = parts[1].strip()
        if not pulse_name:
            continue
        kind = classify_kind(pulse_name)
        label = pulse_name
        if kind == "loopback" and not pulse_name.lower().endswith("monitor"):
            label = f"{pulse_name} (monitor)"
        elif kind == "loopback":
            label = f"{pulse_name} (desktop audio)"
        devices.append(
            CaptureDevice(
                id=f"pulse:{pulse_name}",
                name=label,
                kind=kind,
                backend="parec" if shutil.which("parec") else "pwrecord",
                hostapi="PulseAudio/PipeWire",
                pulse_name=pulse_name,
                channels=1,
                samplerate=16000,
            )
        )
    return devices


def list_pulse_sources() -> list[CaptureDevice]:
    if not shutil.which("pactl"):
        return []
    try:
        out = subprocess.check_output(
            ["pactl", "-f", "json", "list", "sources"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        if isinstance(data, dict):
            data = data.get("sources") or data.get("info") or []
        devices: list[CaptureDevice] = []
        if isinstance(data, list):
            for src in data:
                pulse_name = str(src.get("name") or "")
                desc = str(src.get("description") or pulse_name)
                if not pulse_name:
                    continue
                monitor = src.get("monitor_of_sink") not in (None, "", "n/a")
                kind = "loopback" if monitor or classify_kind(pulse_name) == "loopback" else "microphone"
                label = desc if desc else pulse_name
                if kind == "loopback" and "monitor" not in label.lower() and "desktop" not in label.lower():
                    label = f"{label} (desktop audio)"
                devices.append(
                    CaptureDevice(
                        id=f"pulse:{pulse_name}",
                        name=label,
                        kind=kind,
                        backend="parec" if shutil.which("parec") else "pwrecord",
                        hostapi="PulseAudio/PipeWire",
                        pulse_name=pulse_name,
                        channels=1,
                        samplerate=16000,
                    )
                )
        if devices:
            return devices
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError, ValueError):
        pass

    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return _parse_pactl_short(out)
    except (subprocess.SubprocessError, OSError):
        return []


def list_capture_devices() -> list[CaptureDevice]:
    devices = list_sounddevice_devices()
    seen_pulse = {d.pulse_name for d in devices if d.pulse_name}
    seen_names = {d.name.lower() for d in devices}
    for pulse_dev in list_pulse_sources():
        if pulse_dev.pulse_name in seen_pulse:
            continue
        # Prefer Pulse monitor entries even if a similar ALSA name exists.
        if pulse_dev.kind == "loopback" or pulse_dev.name.lower() not in seen_names:
            devices.append(pulse_dev)
            seen_pulse.add(pulse_dev.pulse_name)
    return devices


def preferred_device(devices: list[CaptureDevice] | None = None) -> CaptureDevice | None:
    devices = list_capture_devices() if devices is None else devices
    for item in devices:
        if item.kind == "loopback":
            return item
    return devices[0] if devices else None


def find_device(device_id: str, devices: list[CaptureDevice] | None = None) -> CaptureDevice | None:
    devices = list_capture_devices() if devices is None else devices
    for item in devices:
        if item.id == device_id:
            return item
    return None


def loopback_setup_help(system: str | None = None) -> dict:
    system = system or platform.system()
    if system == "Windows":
        steps = [
            "This app can capture desktop audio with WASAPI loopback. In the device list, pick a device whose name ends with “(WASAPI loopback)”. That is usually your Speakers or Headphones.",
            "No Stereo Mix is required on Windows 10/11 when a WASAPI loopback device is listed.",
            "If loopback devices are missing, open Sound settings → Recording → right-click empty area → Show Disabled Devices, then enable “Stereo Mix” (some drivers still use this).",
            "Also check that the app playing audio is using the same output device you selected for loopback (e.g. not a different USB headset).",
        ]
        title = "Windows desktop audio (WASAPI loopback)"
    elif system == "Darwin":
        steps = [
            "macOS does not expose a system loopback device by default. Install a virtual device such as BlackHole (2ch), Loopback, or Soundflower.",
            "Create a Multi-Output Device in Audio MIDI Setup that includes both your speakers and BlackHole, then set that Multi-Output as the system output so you can hear audio while capturing it.",
            "In this app, select the BlackHole (or Loopback) input as the desktop-audio device.",
        ]
        title = "macOS desktop audio (virtual loopback device)"
    else:
        steps = [
            "On Linux, capture the PulseAudio/PipeWire monitor source for your output sink (it is usually named like “analog-stereo.monitor”).",
            "PipeWire users: install pipewire-pulse. PulseAudio users: install pulseaudio and pulseaudio-utils (provides pactl and parec).",
            "List sources with: pactl list sources short   — pick the one ending in .monitor.",
            "If no monitor appears, make sure an output sink exists (play some audio, or create a null sink: pactl load-module module-null-sink sink_name=loopback_test).",
            "In this app, select the source labeled “desktop audio” / “monitor”.",
        ]
        title = "Linux desktop audio (PulseAudio / PipeWire monitor)"
    return {
        "os": system,
        "title": title,
        "steps": steps,
        "sounddevice_error": SD_ERROR,
    }


class CaptureError(RuntimeError):
    pass


class AudioCapture:
    def __init__(
        self,
        on_segment: SegmentCallback,
        on_level: LevelCallback | None = None,
        on_error: ErrorCallback | None = None,
        rms_threshold: float = 0.008,
    ) -> None:
        self.on_segment = on_segment
        self.on_level = on_level
        self.on_error = on_error
        self.rms_threshold = rms_threshold
        self._stop = threading.Event()
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._stream = None
        self._device: CaptureDevice | None = None
        self._capture_rate = 16000
        self.listening = False

    @property
    def device(self) -> CaptureDevice | None:
        return self._device

    def start(self, device: CaptureDevice) -> None:
        self.stop()
        self._stop.clear()
        self._queue = queue.Queue(maxsize=64)
        self._device = device
        self._capture_rate = int(device.samplerate or 16000)
        self._open(device)
        self._thread = threading.Thread(target=self._process_loop, name="audio-process", daemon=True)
        self._thread.start()
        self.listening = True

    def stop(self) -> None:
        self.listening = False
        self._stop.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        self._device = None

    def _emit_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)

    def _open(self, device: CaptureDevice) -> None:
        if device.backend in {"sounddevice", "wasapi_loopback"}:
            self._open_sounddevice(device)
            return
        if device.backend in {"parec", "pwrecord"}:
            self._open_pulse(device)
            return
        raise CaptureError(f"Unsupported audio backend: {device.backend}")

    def _open_sounddevice(self, device: CaptureDevice) -> None:
        if sd is None:
            raise CaptureError(
                "PortAudio/sounddevice is not available. Install PortAudio "
                f"(Linux: libportaudio2) and reinstall sounddevice. Detail: {SD_ERROR}"
            )
        extra_settings = None
        if device.backend == "wasapi_loopback":
            if not hasattr(sd, "WasapiSettings"):
                raise CaptureError("This sounddevice build has no WASAPI loopback support. Upgrade sounddevice.")
            extra_settings = sd.WasapiSettings(loopback=True)
        channels = max(1, int(device.channels or 1))
        samplerate = int(device.samplerate or 16000)
        self._capture_rate = samplerate

        def callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
            if status and self.on_error:
                # Non-fatal overflow/underflow notices.
                pass
            try:
                self._queue.put_nowait(np.array(indata, dtype=np.float32, copy=True))
            except queue.Full:
                pass

        try:
            kwargs = dict(
                device=device.sd_index,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                blocksize=0,
                callback=callback,
            )
            if extra_settings is not None:
                kwargs["extra_settings"] = extra_settings
            self._stream = sd.InputStream(**kwargs)
            self._stream.start()
        except Exception as exc:
            raise CaptureError(
                f"Could not open audio device “{device.name}”: {exc}"
            ) from exc

    def _open_pulse(self, device: CaptureDevice) -> None:
        source = device.pulse_name
        if not source:
            raise CaptureError("Pulse/PipeWire source name is missing.")
        self._capture_rate = 16000
        cmd = None
        if shutil.which("parec"):
            cmd = [
                "parec",
                "--format=float32le",
                "--rate=16000",
                "--channels=1",
                f"--device={source}",
            ]
        elif shutil.which("pw-record"):
            cmd = [
                "pw-record",
                "--rate=16000",
                "--channels=1",
                "--format=f32",
                f"--target={source}",
                "-",
            ]
        elif shutil.which("pw-cat"):
            cmd = [
                "pw-cat",
                "--record",
                "--format=f32",
                "--rate=16000",
                "--channels=1",
                "-t",
                "raw",
                f"--target={source}",
                "-",
            ]
        if cmd is None:
            raise CaptureError(
                "Found a Pulse/PipeWire monitor source, but neither parec nor pw-record is installed. "
                "Install pulseaudio-utils or pipewire-audio-client-libraries."
            )
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise CaptureError(f"Failed to start {cmd[0]}: {exc}") from exc
        threading.Thread(target=self._read_proc_bytes, name="pulse-read", daemon=True).start()
        threading.Thread(target=self._watch_proc, name="pulse-watch", daemon=True).start()

    def _read_proc_bytes(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        buf = b""
        chunk_bytes = int(16000 * 0.1 * 4)  # 100 ms float32 mono
        while not self._stop.is_set():
            try:
                data = proc.stdout.read(4096)
            except Exception:
                break
            if not data:
                break
            buf += data
            while len(buf) >= chunk_bytes:
                piece, buf = buf[:chunk_bytes], buf[chunk_bytes:]
                # Align to 4 bytes (float32)
                usable = len(piece) - (len(piece) % 4)
                if usable <= 0:
                    continue
                audio = np.frombuffer(piece[:usable], dtype="<f4").copy()
                try:
                    self._queue.put_nowait(audio)
                except queue.Full:
                    pass

    def _watch_proc(self) -> None:
        proc = self._proc
        if proc is None:
            return
        while not self._stop.is_set() and proc.poll() is None:
            time.sleep(0.2)
        if self._stop.is_set():
            return
        err = ""
        if proc.stderr is not None:
            try:
                err = proc.stderr.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                err = ""
        self._emit_error(
            f"Desktop audio capture process exited (code {proc.returncode}). {err}".strip()
        )
        self.listening = False

    def _process_loop(self) -> None:
        segmenter = SpeechSegmenter(sample_rate=16000, rms_threshold=self.rms_threshold)
        try:
            while not self._stop.is_set():
                try:
                    block = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                mono = to_mono(block)
                audio = resample_audio(mono, self._capture_rate, 16000)
                if audio.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
                if self.on_level:
                    try:
                        self.on_level(rms)
                    except Exception:
                        pass
                for segment in segmenter.add(audio):
                    try:
                        self.on_segment(segment)
                    except Exception as exc:
                        self._emit_error(f"Transcription input error: {exc}")
        finally:
            leftover = segmenter.flush()
            if leftover is not None and leftover.size:
                try:
                    self.on_segment(leftover)
                except Exception:
                    pass

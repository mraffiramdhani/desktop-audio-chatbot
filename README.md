# Desktop Audio Chatbot

A local desktop app that listens to **system/desktop audio** (the sound your computer is playing), transcribes it on-device with Whisper, and lets you chat with an OpenAI-compatible LLM about what was just heard.

Typical uses: ask “what did they just say?” during a video, summarize the last two minutes of a meeting or stream, or pull action items out of spoken audio.

Desktop/loopback capture is the main feature. A microphone is also available in the device list if you want it.

## Features

- Start/stop listening from a desktop GUI (native window when available, otherwise your browser)
- Live transcript pane with timestamps and an input-level meter
- Chat pane that always receives the recent transcript as context
- Local speech-to-text via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (model size is configurable; default `base`)
- LLM calls go to any OpenAI-compatible HTTP API (OpenAI, Ollama, LM Studio, vLLM, llama.cpp server, …)
- If no loopback device is found, the UI explains how to enable one on your OS

## Prerequisites

- Python 3.10 or newer
- FFmpeg (recommended; `faster-whisper` uses it)
- An OpenAI-compatible API endpoint and key (or a local server such as Ollama)
- OS audio pieces:

### Windows

No extra virtual cable is required on Windows 10/11. The app lists each WASAPI output as **`(WASAPI loopback)`**.

Optional fallback: in Sound settings → Recording, enable **Stereo Mix** if your driver provides it.

### Linux (PulseAudio or PipeWire)

You need a **monitor source** for your output sink (often named `*.monitor`).

```bash
# Debian/Ubuntu
sudo apt install python3-venv python3-pip ffmpeg libportaudio2 portaudio19-dev pulseaudio-utils

# Fedora
sudo dnf install python3-pip ffmpeg portaudio-devel pulseaudio-utils
```

PipeWire users should also have `pipewire-pulse` (and ideally `pipewire-audio-client-libraries` / `pw-record`).

List monitor sources:

```bash
pactl list sources short
```

Look for a name ending in `.monitor`. If none exist, play some audio or create a null sink for testing:

```bash
pactl load-module module-null-sink sink_name=loopback_test
pactl list sources short   # should include loopback_test.monitor
```

### macOS (nice-to-have)

macOS has no system loopback device by default. Install [BlackHole](https://existential.audio/blackhole/) (2ch), then:

1. Open **Audio MIDI Setup** and create a Multi-Output Device containing your speakers **and** BlackHole.
2. Set that Multi-Output as the system output so you can hear audio while capturing it.
3. In this app, select **BlackHole 2ch** as the audio source.

Also: `brew install portaudio ffmpeg`

## Install

```bash
git clone <this-repo>
cd desktop-audio-chatbot
python3 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least:

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
WHISPER_MODEL=base
```

Local examples:

```
# Ollama
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1

# LM Studio
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=your-local-model
```

Do not commit `.env`. Settings you save in the UI are stored in `~/.desktop-audio-chatbot/settings.json` with mode `600`.

Optional native window:

```bash
pip install pywebview
```

On Linux that usually also needs a GTK or Qt webview backend. If it is missing, the app opens your browser instead.

## Run

```bash
python -m desktop_audio_chatbot
```

Useful flags:

```bash
python -m desktop_audio_chatbot --list-devices
python -m desktop_audio_chatbot --no-window          # print the URL and keep serving
python -m desktop_audio_chatbot --browser            # force the system browser
python -m desktop_audio_chatbot --host 127.0.0.1 --port 8765
```

The UI is served at `http://127.0.0.1:8765/` by default.

The first listen downloads the Whisper model into `~/.desktop-audio-chatbot/models/` (`base` is ~150 MB). Later starts reuse it.

## How to pick the desktop audio device

1. Open the app and look at **Audio source**.
2. Prefer a device in the **Desktop / system audio** group:
   - Windows: `Speakers (WASAPI loopback)` or `Headphones (WASAPI loopback)` — pick the same output the video/meeting is using.
   - Linux: the source labeled **desktop audio** / **monitor** (`*.monitor`).
   - macOS: BlackHole / Loopback / Soundflower.
3. Click **Start listening**. Speak-free silence is ignored; speech appears in **Live transcript**.
4. Watch the **Level** meter. If it stays flat while audio is playing, you have the wrong device (or the player is routed to a different output).
5. Ask in chat: “what did they just say?”, “summarize the last 2 minutes”, etc.

You can switch to a **microphone** instead. Desktop capture is the intended path.

### No loopback device in the list?

The yellow banner in the UI repeats these OS steps. In short:

| OS | What to enable |
| --- | --- |
| Windows | Select a `(WASAPI loopback)` output. Or enable Stereo Mix. |
| Linux | Install PulseAudio/PipeWire utils and select a `*.monitor` source. |
| macOS | Install BlackHole (or similar) and a Multi-Output Device. |

Confirm devices from the terminal with `python -m desktop_audio_chatbot --list-devices` (`*` marks loopback/desktop devices).

## Chat and transcript context

Every chat request includes the last few minutes of transcript (default 5 minutes, timestamps in local time). The model is instructed not to invent speech that is not in the transcript.

Suggested prompts in the UI:

- What did they just say?
- Summarize last 2 minutes
- Action items

## Configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `LLM_BASE_URL` | OpenAI-compatible API root | `https://api.openai.com/v1` |
| `LLM_API_KEY` | Bearer token (`OPENAI_API_KEY` also accepted) | empty |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `WHISPER_MODEL` | `tiny`, `base`, `small`, `medium`, `large-v3`, plus `.en` variants | `base` |
| `WHISPER_DEVICE` | `auto`, `cpu`, or `cuda` | `auto` |
| `WHISPER_LANGUAGE` | Empty = auto-detect, or `en` / `es` / … | empty |
| `HOST` / `PORT` | Local UI bind address | `127.0.0.1` / `8765` |

The Settings panel can override these without editing `.env`.

Whisper size vs. speed (CPU): `tiny` is fastest and least accurate; `base` is the default balance; `small`/`medium` are better for noisy meetings if you have CPU/GPU to spare.

## Tests

```bash
python -m pytest
```

## Troubleshooting

- **Level meter moves, no transcript:** audio may be music/noise, or too quiet. Try speech, raise output volume, or switch Whisper to `tiny` while debugging.
- **Loopback captures the wrong app:** the player is using another output (HDMI vs speakers, headset vs default). Match the WASAPI/monitor device to that output.
- **Linux PortAudio errors:** install `libportaudio2`. The app can still capture Pulse monitors via `parec` if `pulseaudio-utils` is installed.
- **LLM HTTP 401:** check the API key and base URL. Local servers often accept any placeholder key.
- **First start is slow:** Whisper model download. Watch the status chip (“Loading Whisper…”).
- **CUDA errors:** set `WHISPER_DEVICE=cpu` in Settings.

## Privacy

Audio is transcribed locally. Only the text you already see in the transcript pane (plus your chat messages) is sent to the LLM API you configure. API keys are not committed to git.

"""Local FastAPI app: UI, devices, listening, transcript, chat."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio import CaptureError, list_capture_devices, loopback_setup_help, preferred_device
from .chat import ChatError
from .config import Settings, apply_updates, load_settings, save_settings
from .engine import AppEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def broadcast(self, event: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def emit(self, event: dict) -> None:
        loop = self.loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(event), loop)


hub = Hub()
engine = AppEngine(load_settings(), hub.emit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    engine.settings = load_settings()
    yield


app = FastAPI(title="Desktop Audio Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ListenRequest(BaseModel):
    device_id: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    context_seconds: float = 300.0


class SettingsUpdate(BaseModel):
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    whisper_model: str | None = None
    whisper_device: str | None = None
    whisper_language: str | None = None


class TranscriptAdd(BaseModel):
    text: str = Field(min_length=1)
    duration: float = 2.0
    language: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/status")
def status() -> dict:
    return engine.status()


@app.get("/api/devices")
def devices() -> dict:
    items = list_capture_devices()
    loopbacks = [d for d in items if d.kind == "loopback"]
    help_info = loopback_setup_help()
    preferred = preferred_device(items)
    return {
        "devices": [d.as_dict() for d in items],
        "loopback_available": bool(loopbacks),
        "preferred_id": preferred.id if preferred else None,
        "help": help_info,
    }


@app.get("/api/settings")
def get_settings() -> dict:
    return engine.settings.public_dict()


@app.post("/api/settings")
def post_settings(body: SettingsUpdate) -> dict:
    updated = apply_updates(engine.settings, body.model_dump())
    save_settings(updated)
    engine.update_settings(updated)
    return updated.public_dict()


@app.post("/api/listen/start")
def listen_start(body: ListenRequest) -> dict:
    try:
        device = engine.start_listening(body.device_id)
    except CaptureError as exc:
        return {"ok": False, "error": str(exc), "status": engine.status()}
    return {"ok": True, "device": device.as_dict(), "status": engine.status()}


@app.post("/api/listen/stop")
def listen_stop() -> dict:
    engine.stop_listening()
    return {"ok": True, "status": engine.status()}


@app.get("/api/transcript")
def transcript() -> dict:
    return {"segments": [s.as_dict() for s in engine.store.all()]}


@app.post("/api/transcript")
def transcript_add(body: TranscriptAdd) -> dict:
    segment = engine.store.add(body.text, duration=body.duration, language=body.language)
    hub.emit({"type": "transcript", "segment": segment.as_dict()})
    return {"ok": True, "segment": segment.as_dict()}


@app.delete("/api/transcript")
def transcript_clear() -> dict:
    engine.clear_transcript()
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict:
    try:
        reply = engine.chat_reply(body.message, context_seconds=body.context_seconds)
    except ChatError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "reply": reply}


@app.delete("/api/chat")
def chat_clear() -> dict:
    engine.clear_chat()
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    await ws.send_json({"type": "status", **engine.status()})
    try:
        while True:
            # Keep the socket open; client messages are ignored (ping/pong via receive).
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is not None:
        engine.update_settings(settings)
    return app

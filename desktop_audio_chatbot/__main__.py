"""Launch the local UI server and optional desktop window."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from .audio import list_capture_devices, loopback_setup_help, preferred_device
from .config import load_settings
from .server import app, engine


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _print_devices() -> int:
    devices = list_capture_devices()
    help_info = loopback_setup_help()
    if not devices:
        print("No capture devices found.")
        print(help_info["title"])
        for step in help_info["steps"]:
            print(f"  - {step}")
        if help_info.get("sounddevice_error"):
            print(f"sounddevice error: {help_info['sounddevice_error']}")
        return 1
    print("Capture devices:")
    for dev in devices:
        mark = "*" if dev.kind == "loopback" else " "
        print(f" {mark} [{dev.kind:10}] {dev.id:40} {dev.name}")
    preferred = preferred_device(devices)
    if preferred:
        print(f"\nPreferred (first desktop/loopback): {preferred.name} ({preferred.id})")
    if not any(d.kind == "loopback" for d in devices):
        print("\nNo desktop/loopback device was detected.")
        print(help_info["title"])
        for step in help_info["steps"]:
            print(f"  - {step}")
    return 0


def open_desktop_window(url: str) -> bool:
    try:
        import webview

        webview.create_window("Desktop Audio Chatbot", url, width=1280, height=840)
        webview.start()
        return True
    except Exception as exc:
        print(f"Native window unavailable ({exc}). Opening a browser instead.")
        webbrowser.open(url)
        return False


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Desktop audio chatbot")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--list-devices", action="store_true", help="Print capture devices and exit")
    parser.add_argument("--no-window", action="store_true", help="Run the local web UI only (no desktop window)")
    parser.add_argument("--browser", action="store_true", help="Open the system browser instead of a native window")
    args = parser.parse_args(argv)

    if args.list_devices:
        return _print_devices()

    host = args.host
    port = args.port
    url = f"http://{host}:{port}/"

    config = uvicorn.Config(app, host=host, port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()

    deadline = time.time() + 15
    while time.time() < deadline and not _port_open(host if host != "0.0.0.0" else "127.0.0.1", port):
        if not thread.is_alive():
            print("Server failed to start.", file=sys.stderr)
            return 1
        time.sleep(0.1)

    print(f"Desktop Audio Chatbot is running at {url}")
    print("Press Ctrl+C to stop.")

    try:
        if args.no_window:
            while thread.is_alive():
                time.sleep(0.4)
        elif args.browser:
            webbrowser.open(url)
            while thread.is_alive():
                time.sleep(0.4)
        else:
            open_desktop_window(url)
            server.should_exit = True
    except KeyboardInterrupt:
        server.should_exit = True
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

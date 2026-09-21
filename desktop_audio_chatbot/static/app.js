const $ = (id) => document.getElementById(id);

const state = {
  listening: false,
  devices: [],
};

function fmtTime(epoch) {
  const d = new Date((epoch || 0) * 1000);
  return d.toLocaleTimeString([], { hour12: false });
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  if (res.status === 204) return null;
  return res.json();
}

function setError(message) {
  const el = $("errorBanner");
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.classList.remove("hidden");
  el.textContent = message;
}

function setStatus({ listening, model_loading, error }) {
  state.listening = Boolean(listening);
  const label = $("statusLabel");
  const dot = $("statusDot");
  const btn = $("listenBtn");
  dot.classList.toggle("live", Boolean(listening));
  if (model_loading) {
    label.textContent = "Loading Whisper…";
    btn.textContent = "Starting…";
    btn.disabled = true;
  } else if (listening) {
    label.textContent = "Listening";
    btn.textContent = "Stop listening";
    btn.classList.add("danger");
    btn.classList.remove("primary");
    btn.disabled = false;
  } else {
    label.textContent = "Idle";
    btn.textContent = "Start listening";
    btn.classList.remove("danger");
    btn.classList.add("primary");
    btn.disabled = false;
  }
  if (error) setError(error);
}

function renderDevices(data) {
  state.devices = data.devices || [];
  const select = $("deviceSelect");
  const current = select.value;
  select.innerHTML = "";
  const groups = {
    loopback: document.createElement("optgroup"),
    microphone: document.createElement("optgroup"),
  };
  groups.loopback.label = "Desktop / system audio";
  groups.microphone.label = "Microphones";
  for (const device of state.devices) {
    const opt = document.createElement("option");
    opt.value = device.id;
    opt.textContent = device.name;
    (groups[device.kind] || groups.microphone).appendChild(opt);
  }
  if (groups.loopback.children.length) select.appendChild(groups.loopback);
  if (groups.microphone.children.length) select.appendChild(groups.microphone);
  if (!state.devices.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No audio devices found";
    select.appendChild(opt);
  }
  const preferred = data.preferred_id || current;
  if (preferred) select.value = preferred;

  const banner = $("loopbackBanner");
  if (!data.loopback_available && data.help) {
    const steps = (data.help.steps || []).map((s) => `<li>${s}</li>`).join("");
    banner.classList.remove("hidden");
    banner.innerHTML = `<strong>${data.help.title}</strong><div>No desktop/loopback device was found. You can still pick a microphone, or enable system audio:</div><ol>${steps}</ol>`;
  } else {
    banner.classList.add("hidden");
  }
}

function addTranscript(segment) {
  const empty = $("transcriptEmpty");
  if (empty) empty.remove();
  const wrap = $("transcript");
  const line = document.createElement("div");
  line.className = "line";
  line.innerHTML = `<time>${fmtTime(segment.t_start)}</time><div>${escapeHtml(segment.text)}</div>`;
  wrap.appendChild(line);
  wrap.scrollTop = wrap.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function addChat(role, text) {
  const log = $("chatLog");
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const who = role === "user" ? "You" : role === "error" ? "Error" : "Assistant";
  wrap.innerHTML = `<div class="who">${who}</div><div class="bubble">${escapeHtml(text)}</div>`;
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}

function setLevel(rms) {
  const pct = Math.max(0, Math.min(100, Math.sqrt(Math.max(rms, 0)) * 220));
  $("levelFill").style.width = `${pct}%`;
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "status") setStatus(msg);
    if (msg.type === "level") setLevel(msg.rms || 0);
    if (msg.type === "transcript" && msg.segment) addTranscript(msg.segment);
    if (msg.type === "transcript_clear") {
      $("transcript").innerHTML = `<p class="empty" id="transcriptEmpty">Start listening to desktop audio. Speech will show up here.</p>`;
    }
    if (msg.type === "error") setError(msg.message);
  };
  ws.onclose = () => setTimeout(connectWs, 1200);
}

async function loadAll() {
  const [deviceData, settings, transcript, status] = await Promise.all([
    api("/api/devices"),
    api("/api/settings"),
    api("/api/transcript"),
    api("/api/status"),
  ]);
  renderDevices(deviceData);
  $("llmBaseUrl").value = settings.llm_base_url || "";
  $("llmModel").value = settings.llm_model || "";
  $("whisperModel").value = settings.whisper_model || "base";
  $("whisperDevice").value = settings.whisper_device || "auto";
  $("whisperLanguage").value = settings.whisper_language || "";
  $("apiKeyHint").textContent = settings.llm_api_key_set
    ? `Saved key: ${settings.llm_api_key_hint}`
    : "No API key saved yet.";
  for (const seg of transcript.segments || []) addTranscript(seg);
  setStatus(status);
}

$("listenBtn").addEventListener("click", async () => {
  setError("");
  if (state.listening) {
    await api("/api/listen/stop", { method: "POST", body: "{}" });
    return;
  }
  const deviceId = $("deviceSelect").value;
  if (!deviceId) {
    setError("Select an audio device first.");
    return;
  }
  $("listenBtn").disabled = true;
  const result = await api("/api/listen/start", {
    method: "POST",
    body: JSON.stringify({ device_id: deviceId }),
  });
  if (!result.ok) setError(result.error || "Could not start listening.");
  if (result.status) setStatus(result.status);
});

$("clearTranscriptBtn").addEventListener("click", async () => {
  await api("/api/transcript", { method: "DELETE" });
  $("transcript").innerHTML = `<p class="empty" id="transcriptEmpty">Start listening to desktop audio. Speech will show up here.</p>`;
});

$("clearChatBtn").addEventListener("click", async () => {
  await api("/api/chat", { method: "DELETE" });
  $("chatLog").innerHTML = "";
});

async function sendChat(text) {
  const message = (text || "").trim();
  if (!message) return;
  $("chatInput").value = "";
  addChat("user", message);
  $("sendBtn").disabled = true;
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, context_seconds: 300 }),
    });
    if (result.ok) addChat("assistant", result.reply);
    else addChat("error", result.error || "Chat failed.");
  } catch (err) {
    addChat("error", String(err.message || err));
  } finally {
    $("sendBtn").disabled = false;
    $("chatInput").focus();
  }
}

$("chatForm").addEventListener("submit", (ev) => {
  ev.preventDefault();
  sendChat($("chatInput").value);
});

$("chatInput").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    sendChat($("chatInput").value);
  }
});

$("suggestions").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-prompt]");
  if (btn) sendChat(btn.dataset.prompt);
});

$("settingsBtn").addEventListener("click", () => $("settingsDrawer").classList.remove("hidden"));
$("closeSettingsBtn").addEventListener("click", () => $("settingsDrawer").classList.add("hidden"));

$("settingsForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = {
    llm_base_url: $("llmBaseUrl").value,
    llm_api_key: $("llmApiKey").value,
    llm_model: $("llmModel").value,
    whisper_model: $("whisperModel").value,
    whisper_device: $("whisperDevice").value,
    whisper_language: $("whisperLanguage").value,
  };
  const saved = await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
  $("llmApiKey").value = "";
  $("apiKeyHint").textContent = saved.llm_api_key_set
    ? `Saved key: ${saved.llm_api_key_hint}`
    : "No API key saved yet.";
  $("settingsDrawer").classList.add("hidden");
});

connectWs();
loadAll().catch((err) => setError(`Failed to load app state: ${err.message}`));

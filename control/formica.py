#!/usr/bin/env python3
"""Formica: the control panel of a self-hosted AI workplace.

Formica runs Open WebUI with a document knowledge base (PostgreSQL, Qdrant,
Docling, an embedding model) behind its own reverse proxy (Caddy), on top of
any OpenAI-compatible backend, first of all a MyrmidLM cluster.

Everything is configured in the web interface. Formica keeps its settings in
config.json, writes compose.yaml and the Caddyfile from them and runs
`docker compose`. The generated files are plain and readable: what you see in
stack/ is exactly what runs.

  formica.py serve     the control panel (port 8095)
  formica.py render    only write stack/ from config.json
  formica.py token     show the setup token

Standard library only, like MyrmidLM.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path(os.environ.get("FORMICA_HOME", "/opt/formica"))
CONFIG = HOME / "config.json"
STACK = HOME / "stack"
TOKEN_FILE = HOME / "setup-token"
PROJECT = "formica"
PORT = int(os.environ.get("FORMICA_PORT", "8095"))


def version() -> str:
    for p in (HERE / "VERSION", HERE.parent / "VERSION"):
        if p.exists():
            return p.read_text().strip()
    return "dev"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------
# Versions: tested together. Formica only moves them with a new release.
# --------------------------------------------------------------------------

IMAGES = {
    "caddy": "caddy:2",
    "open-webui": "ghcr.io/open-webui/open-webui:v0.12.4",
    "postgres": "postgres:17",
    "qdrant": "qdrant/qdrant:latest",
    "docling": "ghcr.io/docling-project/docling-serve-cpu:latest",
    "embeddings": "ghcr.io/ggml-org/llama.cpp:server",
}

MODULES = [
    # name, label, what it does, can be switched off
    ("caddy", "Reverse Proxy (Caddy)", "Zugriff im Browser, HTTPS und Zertifikate", False),
    ("open-webui", "Open WebUI", "Chat, Benutzer, Wissensdatenbanken", False),
    ("postgres", "PostgreSQL", "Benutzer, Chats und Einstellungen von Open WebUI", False),
    ("qdrant", "Qdrant", "Vektordatenbank für die Dokumente", False),
    ("embeddings", "Embeddings (bge-m3)", "Wandelt Texte und Fragen in Vektoren um", False),
    ("docling", "Docling", "Liest PDFs, Office-Dateien, Tabellen und Scans", True),
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULTS = {
    "general": {
        "name": "Formica",
        "host": "",              # DNS name or IP the users type in the browser
        "access": "http",        # http | internal | acme | proxy
        "http_port": 80,
        "https_port": 443,
        "acme_email": "",
        "trusted_proxies": "",   # access=proxy: IPs of the company proxy
    },
    "backend": {
        "url": "",               # e.g. http://192.168.2.243:8080/v1 (MyrmidLM master)
        "key": "",
    },
    "rag": {
        "embedding_model": "gpustack/bge-m3-GGUF:Q8_0",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 8,
        "hybrid": False,
        "docling": True,
        "docling_workers": 2,
        "ocr": "auto",           # auto | force | off
    },
    "users": {
        "signup": True,           # users may register, the admin unlocks them
        "default_role": "pending",
    },
}

RAG_TEMPLATE = """### Task:
Respond to the user query using only the provided context, incorporating inline citations in the format [id] **only when the <source> tag includes an explicit id attribute** (e.g., <source id="1">).

### Guidelines:
- If the answer isn't present in the context, clearly state that the documents don't contain this information. Do not answer from your own general knowledge.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- When information comes from a table, reproduce the relevant values exactly (numbers, units, dates).
- If sources contradict each other, point this out and name both sources.
- **Only include inline citations using [id] (e.g., [1], [2]) when the <source> tag includes an id attribute.**
- Do not cite if the <source> tag does not contain an id attribute.
- Do not use XML tags in your response.
- Ensure citations are concise and directly related to the information provided.

### Output:
Provide a clear and direct response to the user's query, including inline citations in the format [id] only when the <source> tag with id attribute is present in the context.

<context>
{{CONTEXT}}
</context>
"""

_lock = threading.RLock()


def _merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    with _lock:
        raw = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
        cfg = _merge(DEFAULTS, raw)
        sec = cfg.setdefault("secrets", {})
        changed = False
        for k in ("pg_password", "qdrant_key", "webui_secret"):
            if not sec.get(k):
                sec[k] = secrets.token_hex(24)  # hex: safe inside URLs
                changed = True
        if changed or not CONFIG.exists():
            save_config(cfg)
        return cfg


def save_config(cfg: dict) -> None:
    with _lock:
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        os.chmod(tmp, 0o600)
        tmp.replace(CONFIG)


def public_config(cfg: dict) -> dict:
    """What the browser may see: no secrets, the backend key masked."""
    out = {k: v for k, v in cfg.items() if k not in ("secrets", "admin")}
    out["backend"] = dict(cfg["backend"])
    key = out["backend"].get("key", "")
    out["backend"]["key"] = ""
    out["backend"]["key_set"] = bool(key)
    return out


def normalize_backend_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if not re.match(r"^https?://", url):
        url = "http://" + url
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def validate(cfg: dict) -> list[str]:
    errs = []
    g = cfg["general"]
    if g["access"] not in ("http", "internal", "acme", "proxy"):
        errs.append("Unbekannte Zugriffsart.")
    if not g["host"]:
        errs.append("Bitte Hostname oder IP-Adresse eintragen, unter der Formica erreichbar ist.")
    if g["access"] == "acme" and re.match(r"^\d+\.\d+\.\d+\.\d+$", g["host"] or ""):
        errs.append("Let's Encrypt braucht einen öffentlichen DNS-Namen, keine IP-Adresse.")
    if g["access"] == "proxy" and not g["trusted_proxies"].strip():
        errs.append("Bitte die IP-Adresse des vorgeschalteten Reverse Proxys angeben.")
    for k in ("http_port", "https_port"):
        if not (1 <= int(g[k]) <= 65535):
            errs.append(f"Ungültiger Port: {g[k]}")
    if not cfg["backend"]["url"]:
        errs.append("Bitte die Adresse des Modell-Backends (MyrmidLM-Master) angeben.")
    r = cfg["rag"]
    if not (200 <= int(r["chunk_size"]) <= 8000):
        errs.append("Chunk-Größe zwischen 200 und 8000.")
    if not (0 <= int(r["chunk_overlap"]) < int(r["chunk_size"])):
        errs.append("Overlap muss kleiner als die Chunk-Größe sein.")
    if r["ocr"] not in ("auto", "force", "off"):
        errs.append("Unbekannte OCR-Einstellung.")
    if not (1 <= int(r["top_k"]) <= 50):
        errs.append("Top K zwischen 1 und 50.")
    return errs


# --------------------------------------------------------------------------
# Rendering: config.json -> stack/compose.yaml + stack/caddy/Caddyfile
# --------------------------------------------------------------------------

def _yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    # JSON strings are valid YAML; "$" is escaped for docker compose
    return json.dumps(str(v).replace("$", "$$"), ensure_ascii=False)


def to_yaml(obj, indent: int = 0) -> str:
    pad = "  " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = k if re.match(r"^[A-Za-z0-9_.\-]+$", k) else json.dumps(k)
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{key}:")
                lines.append(to_yaml(v, indent + 1))
            elif isinstance(v, dict):
                lines.append(f"{pad}{key}: {{}}")
            elif isinstance(v, list):
                lines.append(f"{pad}{key}: []")
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(v)}")
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict):
                inner = to_yaml(v, indent + 1).lstrip()
                lines.append(f"{pad}- {inner}")
            else:
                lines.append(f"{pad}- {_yaml_scalar(v)}")
    return "\n".join(lines)


def webui_url(cfg: dict) -> str:
    g = cfg["general"]
    host = g["host"] or "localhost"
    if g["access"] in ("internal", "acme"):
        port = int(g["https_port"])
        return f"https://{host}" + ("" if port == 443 else f":{port}")
    if g["access"] == "proxy":
        return f"https://{host}"  # TLS is done by the proxy in front
    port = int(g["http_port"])
    return f"http://{host}" + ("" if port == 80 else f":{port}")


def render_caddyfile(cfg: dict) -> str:
    g = cfg["general"]
    upstream = "reverse_proxy open-webui:8080"
    glob = ["admin off"]
    if g["access"] in ("http", "proxy"):
        glob.append("auto_https off")
    if g["access"] == "proxy":
        ips = " ".join(re.split(r"[\s,;]+", g["trusted_proxies"].strip()))
        glob.append(f"servers {{\n\t\ttrusted_proxies static {ips}\n\t\tclient_ip_headers X-Forwarded-For\n\t}}")
    if g["access"] == "acme" and g["acme_email"]:
        glob.append(f"email {g['acme_email']}")
    head = "{\n" + "".join(f"\t{x}\n" for x in glob) + "}\n\n"
    if g["access"] in ("http", "proxy"):
        site = f":80 {{\n\t{upstream}\n}}\n"
    elif g["access"] == "internal":
        site = f"{g['host']} {{\n\ttls internal\n\t{upstream}\n}}\n"
    else:
        site = f"{g['host']} {{\n\t{upstream}\n}}\n"
    return "# Erzeugt von Formica. Änderungen bitte in der Formica-Oberfläche.\n" + head + site


def render_compose(cfg: dict) -> dict:
    g, b, r, u, s = cfg["general"], cfg["backend"], cfg["rag"], cfg["users"], cfg["secrets"]
    https = g["access"] in ("internal", "acme")

    caddy_ports = [f"{int(g['http_port'])}:80"]
    if https:
        caddy_ports.append(f"{int(g['https_port'])}:443")

    webui_env = {
        "WEBUI_NAME": g["name"],
        "WEBUI_URL": webui_url(cfg),
        "WEBUI_SECRET_KEY": s["webui_secret"],
        "DATABASE_URL": f"postgresql://openwebui:{s['pg_password']}@postgres:5432/openwebui",
        # Formica owns these settings: they come from here on every start.
        # Models, knowledge bases, users and groups stay in Open WebUI.
        "ENABLE_PERSISTENT_CONFIG": "false",
        "ENABLE_SIGNUP": "true" if u["signup"] else "false",
        "DEFAULT_USER_ROLE": u["default_role"],
        "ENABLE_OLLAMA_API": "false",
        "ENABLE_OPENAI_API": "true",
        "OPENAI_API_BASE_URL": b["url"],
        "OPENAI_API_KEY": b["key"] or "none",
        "VECTOR_DB": "qdrant",
        "QDRANT_URI": "http://qdrant:6333",
        "QDRANT_API_KEY": s["qdrant_key"],
        "ENABLE_QDRANT_MULTITENANCY_MODE": "true",
        "QDRANT_ON_DISK": "true",
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_OPENAI_API_BASE_URL": "http://embeddings:8080/v1",
        "RAG_OPENAI_API_KEY": "none",
        "RAG_EMBEDDING_MODEL": "bge-m3",
        "RAG_EMBEDDING_BATCH_SIZE": "16",
        "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": "true",
        "CHUNK_SIZE": str(int(r["chunk_size"])),
        "CHUNK_OVERLAP": str(int(r["chunk_overlap"])),
        "RAG_TOP_K": str(int(r["top_k"])),
        "RAG_RELEVANCE_THRESHOLD": "0",
        "ENABLE_RAG_HYBRID_SEARCH": "true" if r["hybrid"] else "false",
        "RAG_TEMPLATE": RAG_TEMPLATE,
        "ANONYMIZED_TELEMETRY": "false",
        # Updates come with Formica (tested versions), not from the banner in Open WebUI
        "ENABLE_VERSION_UPDATE_CHECK": "false",
    }
    if r["docling"]:
        webui_env["CONTENT_EXTRACTION_ENGINE"] = "docling"
        webui_env["DOCLING_SERVER_URL"] = "http://docling:5001"
        params = {"do_ocr": r["ocr"] != "off", "table_mode": "accurate"}
        if r["ocr"] != "off":
            params["force_ocr"] = r["ocr"] == "force"
            params["ocr_lang"] = ["de", "en"]
        webui_env["DOCLING_PARAMS"] = json.dumps(params)

    depends = {"postgres": {"condition": "service_healthy"},
               "qdrant": {"condition": "service_started"},
               "embeddings": {"condition": "service_started"}}
    if r["docling"]:
        depends["docling"] = {"condition": "service_started"}

    services = {
        "caddy": {
            "image": IMAGES["caddy"],
            "restart": "unless-stopped",
            "ports": caddy_ports,
            "volumes": ["./caddy/Caddyfile:/etc/caddy/Caddyfile:ro", "caddy_data:/data", "caddy_config:/config"],
            "depends_on": ["open-webui"],
            "networks": ["formica"],
        },
        "open-webui": {
            "image": IMAGES["open-webui"],
            "restart": "unless-stopped",
            "depends_on": depends,
            "environment": webui_env,
            "volumes": ["openwebui_data:/app/backend/data"],
            "networks": ["formica"],
        },
        "postgres": {
            "image": IMAGES["postgres"],
            "restart": "unless-stopped",
            "environment": {"POSTGRES_USER": "openwebui", "POSTGRES_PASSWORD": s["pg_password"],
                            "POSTGRES_DB": "openwebui"},
            "volumes": ["postgres_data:/var/lib/postgresql/data"],
            "healthcheck": {"test": ["CMD-SHELL", "pg_isready -U openwebui -d openwebui"],
                            "interval": "10s", "timeout": "5s", "retries": 10},
            "networks": ["formica"],
        },
        "qdrant": {
            "image": IMAGES["qdrant"],
            "restart": "unless-stopped",
            "environment": {"QDRANT__SERVICE__API_KEY": s["qdrant_key"],
                            "QDRANT__TELEMETRY_DISABLED": "true"},
            "volumes": ["qdrant_data:/qdrant/storage"],
            "networks": ["formica"],
        },
        "embeddings": {
            # llama.cpp server, the same engine as MyrmidLM, with an OpenAI
            # compatible /v1/embeddings. The model is loaded from Hugging Face once.
            "image": IMAGES["embeddings"],
            "restart": "unless-stopped",
            "environment": {"LLAMA_CACHE": "/models"},
            "command": ["-hf", r["embedding_model"], "--embeddings", "--host", "0.0.0.0", "--port", "8080",
                        "-c", "8192", "-b", "8192", "-ub", "8192"],
            "volumes": ["embeddings_models:/models"],
            "networks": ["formica"],
        },
    }
    if r["docling"]:
        services["docling"] = {
            "image": IMAGES["docling"],
            "restart": "unless-stopped",
            "environment": {"DOCLING_SERVE_MAX_SYNC_WAIT": "600",
                            "DOCLING_SERVE_ENG_LOC_NUM_WORKERS": str(int(r["docling_workers"]))},
            "networks": ["formica"],
        }

    return {
        "name": PROJECT,
        "services": services,
        "volumes": {v: {} for v in ("caddy_data", "caddy_config", "openwebui_data", "postgres_data",
                                    "qdrant_data", "embeddings_models")},
        "networks": {"formica": {}},
    }


def render(cfg: dict) -> None:
    (STACK / "caddy").mkdir(parents=True, exist_ok=True)
    head = ("# Erzeugt von Formica " + version() + ". Nicht von Hand ändern:\n"
            "# Formica schreibt die Datei bei jedem Übernehmen neu.\n")
    compose = STACK / "compose.yaml"
    compose.write_text(head + to_yaml(render_compose(cfg)) + "\n")
    os.chmod(compose, 0o600)  # holds the database passwords
    (STACK / "caddy" / "Caddyfile").write_text(render_caddyfile(cfg))


# --------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------

def compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-p", PROJECT, "-f", str(STACK / "compose.yaml"), *args]


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except FileNotFoundError:
        return 127, "docker ist nicht installiert oder nicht erreichbar."
    except subprocess.TimeoutExpired:
        return 124, "Zeitüberschreitung."


class Job:
    """One long-running action at a time (apply, pull), its output for the UI."""

    def __init__(self):
        self.name = ""
        self.running = False
        self.ok = None
        self.lines: list[str] = []
        self.started = 0.0
        self.on_ok = None

    def start(self, name: str, steps: list[list[str]], on_ok=None) -> bool:
        with _lock:
            if self.running:
                return False
            self.name, self.running, self.ok, self.lines, self.started = name, True, None, [], time.time()
            self.on_ok = on_ok
        threading.Thread(target=self._run, args=(steps,), daemon=True).start()
        return True

    def _run(self, steps):
        ok = True
        for cmd in steps:
            shown = ["docker", "compose", *cmd[6:]] if cmd[:2] == ["docker", "compose"] else cmd
            self.lines.append("$ " + " ".join(shown))
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    self.lines.append(line.rstrip())
                    if len(self.lines) > 2000:
                        del self.lines[:500]
                if p.wait() != 0:
                    ok = False
                    self.lines.append(f"Fehler (Exit-Code {p.returncode})")
                    break
            except FileNotFoundError:
                ok = False
                self.lines.append("docker ist nicht erreichbar.")
                break
        if ok and self.on_ok:
            try:
                self.on_ok()
            except Exception as e:  # noqa: BLE001
                log(f"Job {self.name}: {e}")
        self.ok, self.running = ok, False
        log(f"Job {self.name}: {'ok' if ok else 'Fehler'}")

    def state(self) -> dict:
        return {"name": self.name, "running": self.running, "ok": self.ok,
                "lines": self.lines[-400:], "started": self.started}


JOB = Job()


def apply_stack(pull: bool = False) -> bool:
    cfg = load_config()
    caddyfile = STACK / "caddy" / "Caddyfile"
    before = caddyfile.read_text() if caddyfile.exists() else None
    render(cfg)
    steps = []
    if pull:
        steps.append(compose_cmd("pull"))
    steps.append(compose_cmd("up", "-d", "--remove-orphans"))
    if before is not None and before != caddyfile.read_text():
        # a bind-mounted file: compose does not notice the change by itself
        steps.append(compose_cmd("restart", "caddy"))
    def done():
        c = load_config()
        c["applied"] = time.strftime("%Y-%m-%d %H:%M")
        save_config(c)

    return JOB.start("Aktualisieren" if pull else "Übernehmen", steps, done)


def services_status() -> list[dict]:
    state = {}
    if (STACK / "compose.yaml").exists():
        code, out = run(compose_cmd("ps", "-a", "--format", "json"), timeout=20)
        if code == 0:
            out = out.strip()
            items = []
            if out.startswith("["):
                items = json.loads(out)
            else:
                items = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
            for it in items:
                state[it.get("Service")] = {
                    "state": it.get("State", ""),
                    "status": it.get("Status", ""),
                    "health": it.get("Health", ""),
                    "image": it.get("Image", ""),
                }
    cfg = load_config()
    out = []
    for name, label, desc, optional in MODULES:
        enabled = not (name == "docling" and not cfg["rag"]["docling"])
        st = state.get(name, {})
        out.append({"name": name, "label": label, "desc": desc, "optional": optional,
                    "enabled": enabled, "state": st.get("state", "fehlt" if enabled else "aus"),
                    "status": st.get("status", ""), "health": st.get("health", ""),
                    "image": IMAGES[name]})
    return out


def caddy_root_cert() -> str | None:
    code, out = run(compose_cmd("exec", "-T", "caddy", "cat", "/data/caddy/pki/authorities/local/root.crt"))
    return out if code == 0 and "BEGIN CERTIFICATE" in out else None


# --------------------------------------------------------------------------
# Backend test
# --------------------------------------------------------------------------

def test_backend(url: str, key: str) -> dict:
    url = normalize_backend_url(url)
    if not url:
        return {"ok": False, "error": "Keine Adresse angegeben."}
    req = urllib.request.Request(url + "/models", headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return {"ok": True, "url": url, "models": models}
    except urllib.error.HTTPError as e:
        msg = "API-Key falsch oder fehlt." if e.code in (401, 403) else f"HTTP {e.code}"
        return {"ok": False, "url": url, "error": msg}
    except Exception as e:  # noqa: BLE001 - shown to the admin
        return {"ok": False, "url": url, "error": f"Nicht erreichbar: {e}"}


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------

def system_info() -> dict:
    info = {"version": version(), "cpus": os.cpu_count()}
    try:
        info["load"] = [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        pass
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0]) * 1024
        info["mem_total"] = mem.get("MemTotal")
        info["mem_avail"] = mem.get("MemAvailable")
    except OSError:
        pass
    try:
        du = shutil.disk_usage(HOME if HOME.exists() else "/")
        info["disk_total"], info["disk_free"] = du.total, du.free
    except OSError:
        pass
    code, out = run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=10)
    info["docker"] = out.strip() if code == 0 else None
    return info


# --------------------------------------------------------------------------
# Admin login
# --------------------------------------------------------------------------

def hash_password(pw: str, salt: bytes | None = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 240_000)
    return {"salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(dk).decode()}


def check_password(cfg: dict, pw: str) -> bool:
    a = cfg.get("admin")
    if not a:
        return False
    test = hash_password(pw, base64.b64decode(a["salt"]))
    return hmac.compare_digest(test["hash"], a["hash"])


def setup_token() -> str:
    if not TOKEN_FILE.exists():
        HOME.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(secrets.token_urlsafe(12))
        os.chmod(TOKEN_FILE, 0o600)
    return TOKEN_FILE.read_text().strip()


SESSIONS: dict[str, float] = {}
SESSION_TTL = 12 * 3600
FAILED: dict[str, list[float]] = {}


def too_many_attempts(ip: str) -> bool:
    now = time.time()
    FAILED[ip] = [t for t in FAILED.get(ip, []) if now - t < 300]
    return len(FAILED[ip]) >= 8


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def update_from_form(cfg: dict, data: dict) -> dict:
    """Take only known fields, with the right types."""
    for section, fields in DEFAULTS.items():
        incoming = data.get(section)
        if not isinstance(incoming, dict):
            continue
        for k, default in fields.items():
            if k not in incoming:
                continue
            v = incoming[k]
            if isinstance(default, bool):
                v = bool(v)
            elif isinstance(default, int):
                v = int(v)
            else:
                v = str(v).strip()
            if section == "backend" and k == "key" and v == "":
                continue  # empty field: keep the stored key
            cfg[section][k] = v
    cfg["backend"]["url"] = normalize_backend_url(cfg["backend"]["url"])
    return cfg


class Handler(BaseHTTPRequestHandler):
    server_version = "Formica"

    def log_message(self, fmt, *args):  # quieter than the default
        if not self.path.startswith("/api/") or self.command != "GET":
            log(f"{self.client_address[0]} {self.command} {self.path.split('?')[0]}")

    # --- helpers ---
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj, extra: dict | None = None):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise ValueError("zu groß")
        if n and "application/json" not in (self.headers.get("Content-Type") or ""):
            raise ValueError("JSON erwartet")
        return json.loads(self.rfile.read(n) or b"{}")

    def _session(self) -> str | None:
        c = SimpleCookie(self.headers.get("Cookie") or "")
        tok = c["formica_session"].value if "formica_session" in c else None
        if tok and SESSIONS.get(tok, 0) > time.time():
            SESSIONS[tok] = time.time() + SESSION_TTL
            return tok
        return None

    def _new_session(self) -> dict:
        tok = secrets.token_urlsafe(32)
        SESSIONS[tok] = time.time() + SESSION_TTL
        return {"Set-Cookie": f"formica_session={tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_TTL}"}

    # --- routes ---
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = (HERE / "dashboard.html").read_bytes()
            return self._send(200, body, "text/html; charset=utf-8")
        if path == "/api/state":
            cfg = load_config()
            return self._json(200, {"setup": not cfg.get("admin"), "auth": bool(self._session()),
                                    "version": version()})
        if not self._session():
            return self._json(401, {"error": "Bitte anmelden."})
        if path == "/api/config":
            cfg = load_config()
            return self._json(200, {"config": public_config(cfg), "webui_url": webui_url(cfg),
                                    "errors": validate(cfg), "applied": cfg.get("applied")})
        if path == "/api/services":
            return self._json(200, {"services": services_status(), "job": JOB.state()})
        if path == "/api/job":
            return self._json(200, JOB.state())
        if path == "/api/system":
            return self._json(200, system_info())
        if path.startswith("/api/logs/"):
            name = path.rsplit("/", 1)[-1]
            if name not in IMAGES:
                return self._json(404, {"error": "Unbekannter Dienst"})
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            lines = max(10, min(int(q.get("lines", ["300"])[0]), 5000))
            code, out = run(compose_cmd("logs", "--no-color", "--tail", str(lines), name), timeout=20)
            return self._json(200, {"ok": code == 0, "log": out})
        if path == "/api/files":
            comp = (STACK / "compose.yaml")
            caddy = (STACK / "caddy" / "Caddyfile")
            txt = comp.read_text() if comp.exists() else ""
            cfg = load_config()
            for v in cfg["secrets"].values():  # never show passwords in the browser
                txt = txt.replace(v, "•••")
            if cfg["backend"]["key"]:
                txt = txt.replace(cfg["backend"]["key"], "•••")
            return self._json(200, {"compose": txt, "caddyfile": caddy.read_text() if caddy.exists() else ""})
        if path == "/api/root-cert":
            pem = caddy_root_cert()
            if not pem:
                return self._json(404, {"error": "Kein internes Zertifikat vorhanden (Zugriff „Interne CA“ und Caddy gestartet?)."})
            return self._send(200, pem.encode(), "application/x-pem-file",
                              {"Content-Disposition": 'attachment; filename="formica-root-ca.crt"'})
        return self._json(404, {"error": "Nicht gefunden"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        ip = self.client_address[0]
        try:
            data = self._body()
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": f"Ungültige Anfrage: {e}"})

        if path == "/api/setup":
            cfg = load_config()
            if cfg.get("admin"):
                return self._json(409, {"error": "Formica ist bereits eingerichtet."})
            if too_many_attempts(ip):
                return self._json(429, {"error": "Zu viele Versuche. Bitte 5 Minuten warten."})
            if not hmac.compare_digest(str(data.get("token", "")).strip(), setup_token()):
                FAILED.setdefault(ip, []).append(time.time())
                return self._json(403, {"error": "Einrichtungscode falsch. Anzeigen auf dem Server: sudo formica token"})
            pw = str(data.get("password", ""))
            if len(pw) < 10:
                return self._json(400, {"error": "Das Passwort braucht mindestens 10 Zeichen."})
            cfg["admin"] = hash_password(pw)
            save_config(cfg)
            TOKEN_FILE.unlink(missing_ok=True)
            log("Einrichtung: Admin-Passwort gesetzt")
            return self._json(200, {"ok": True}, self._new_session())

        if path == "/api/login":
            if too_many_attempts(ip):
                return self._json(429, {"error": "Zu viele Versuche. Bitte 5 Minuten warten."})
            if check_password(load_config(), str(data.get("password", ""))):
                FAILED.pop(ip, None)
                return self._json(200, {"ok": True}, self._new_session())
            FAILED.setdefault(ip, []).append(time.time())
            return self._json(403, {"error": "Passwort falsch."})

        tok = self._session()
        if not tok:
            return self._json(401, {"error": "Bitte anmelden."})

        if path == "/api/logout":
            SESSIONS.pop(tok, None)
            return self._json(200, {"ok": True}, {"Set-Cookie": "formica_session=; Path=/; Max-Age=0"})
        if path == "/api/password":
            cfg = load_config()
            if not check_password(cfg, str(data.get("old", ""))):
                return self._json(403, {"error": "Altes Passwort falsch."})
            if len(str(data.get("new", ""))) < 10:
                return self._json(400, {"error": "Das Passwort braucht mindestens 10 Zeichen."})
            cfg["admin"] = hash_password(str(data["new"]))
            save_config(cfg)
            SESSIONS.clear()
            return self._json(200, {"ok": True}, self._new_session())
        if path == "/api/config":
            cfg = load_config()
            try:
                cfg = update_from_form(cfg, data)
            except (TypeError, ValueError) as e:
                return self._json(400, {"error": f"Ungültiger Wert: {e}"})
            errs = validate(cfg)
            save_config(cfg)
            return self._json(200, {"ok": not errs, "errors": errs, "config": public_config(cfg),
                                    "webui_url": webui_url(cfg)})
        if path == "/api/test-backend":
            cfg = load_config()
            key = data.get("key") or cfg["backend"]["key"]
            return self._json(200, test_backend(data.get("url") or cfg["backend"]["url"], key))
        if path in ("/api/apply", "/api/update"):
            cfg = load_config()
            errs = validate(cfg)
            if errs:
                return self._json(400, {"error": "Bitte zuerst die Einstellungen korrigieren.", "errors": errs})
            if not apply_stack(pull=(path == "/api/update")):
                return self._json(409, {"error": "Es läuft bereits ein Vorgang."})
            return self._json(200, {"ok": True})
        m = re.match(r"^/api/service/([a-z\-]+)/(start|stop|restart)$", path)
        if m:
            name, action = m.groups()
            if name not in IMAGES:
                return self._json(404, {"error": "Unbekannter Dienst"})
            code, out = run(compose_cmd(action, name), timeout=120)
            return self._json(200 if code == 0 else 500, {"ok": code == 0, "output": out})
        return self._json(404, {"error": "Nicht gefunden"})


def serve() -> None:
    cfg = load_config()
    if (STACK / "compose.yaml").exists() or cfg.get("admin"):
        render(cfg)  # a new Formica version may write the files differently
    if not cfg.get("admin"):
        log(f"Einrichtung offen. Einrichtungscode: {setup_token()}")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    log(f"Formica {version()} auf Port {PORT}, Daten in {HOME}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "serve"
    if cmd == "serve":
        serve()
    elif cmd == "render":
        render(load_config())
        print(f"Geschrieben: {STACK}/compose.yaml und caddy/Caddyfile")
    elif cmd == "token":
        if load_config().get("admin"):
            print("Formica ist bereits eingerichtet. Passwort vergessen: formica.py reset-password")
        else:
            print(setup_token())
    elif cmd == "reset-password":
        cfg = load_config()
        cfg.pop("admin", None)
        save_config(cfg)
        TOKEN_FILE.unlink(missing_ok=True)
        print("Admin-Passwort entfernt. Neuer Einrichtungscode:", setup_token())
    elif cmd == "version":
        print(version())
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

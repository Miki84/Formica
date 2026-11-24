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


EVENTS_MAX = 2000


def event(text: str, level: str = "info", source: str = "formica") -> None:
    """A line for the event log (cockpit, Ereignisse): what happened, kept on disk."""
    log(text)
    try:
        f = HOME / "events.jsonl"
        with f.open("a") as fh:
            fh.write(json.dumps({"t": time.time(), "level": level, "source": source, "text": text},
                                ensure_ascii=False) + "\n")
        if f.stat().st_size > 600_000:  # keep the newest lines
            lines = f.read_text().splitlines()[-EVENTS_MAX:]
            f.write_text("\n".join(lines) + "\n")
    except OSError:
        pass


def read_events(limit: int = 200, q: str = "", level: str = "") -> list[dict]:
    f = HOME / "events.jsonl"
    if not f.exists():
        return []
    out = []
    for line in reversed(f.read_text().splitlines()):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if level and e.get("level") != level:
            continue
        if q and q.lower() not in (e.get("text", "") + " " + e.get("source", "")).lower():
            continue
        out.append(e)
        if len(out) >= limit:
            break
    return out


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
    "searxng": "searxng/searxng:latest",
    "whisper": "ghcr.io/speaches-ai/speaches:latest-cpu",
    "n8n": "docker.n8n.io/n8nio/n8n:latest",
    "files": "formica-files:local",          # built from app/services/files
    "images": "ghcr.io/leejet/stable-diffusion.cpp:master-vulkan",
    "images-model": "curlimages/curl:latest",
}
IMAGES_CUDA = "ghcr.io/leejet/stable-diffusion.cpp:master-cuda"

MODULES = [
    # name, label, what it does, can be switched off
    ("caddy", "Reverse Proxy (Caddy)", "Zugriff im Browser, HTTPS und Zertifikate", False),
    ("open-webui", "Open WebUI", "Chat, Benutzer, Wissensdatenbanken", False),
    ("postgres", "PostgreSQL", "Benutzer, Chats und Einstellungen von Open WebUI", False),
    ("qdrant", "Qdrant", "Vektordatenbank für die Dokumente", False),
    ("embeddings", "Embeddings (bge-m3)", "Wandelt Texte und Fragen in Vektoren um", False),
    ("docling", "Docling", "Liest PDFs, Office-Dateien, Tabellen und Scans", True),
    ("searxng", "SearXNG", "Metasuchmaschine für die Websuche in Open WebUI", True),
    ("whisper", "Whisper (speaches)", "Spracherkennung für die Spracheingabe im Chat", True),
    ("n8n", "n8n", "Workflows und Automatisierung", True),
    ("files", "Dateien erzeugen", "Excel, Word, PDF und CSV aus dem Chat", True),
    ("images", "Bilder erzeugen (sd-server)", "Stable Diffusion über stable-diffusion.cpp", True),
    ("images-model", "Bildmodell laden", "Lädt das Bildmodell einmalig herunter", True),
]

# Solutions (what the cockpit shows): a solution or a bundle of services.
CATALOG = [
    {"id": "chat", "label": "Chat & RAG", "icon": "chat", "core": True,
     "desc": "Open WebUI mit Wissensdatenbank: Chat, Benutzer, Dokumente",
     "services": ["open-webui", "postgres", "qdrant", "embeddings", "docling", "caddy"]},
    {"id": "websearch", "label": "Websuche", "icon": "search",
     "desc": "SearXNG: Open WebUI kann im Internet suchen", "services": ["searxng"]},
    {"id": "whisper", "label": "Whisper", "icon": "mic",
     "desc": "Spracheingabe im Chat (Speech-to-Text)", "services": ["whisper"]},
    {"id": "files", "label": "Dateien erzeugen", "icon": "file",
     "desc": "Excel, Word, PDF und CSV direkt aus dem Chat", "services": ["files"]},
    {"id": "images", "label": "Bilder erzeugen", "icon": "image",
     "desc": "Bilder aus Text (Stable Diffusion), im Chat über das Bild-Symbol", "services": ["images-model", "images"]},
    {"id": "n8n", "label": "n8n", "icon": "flow",
     "desc": "Workflows und Automatisierung", "services": ["n8n"], "web": ("n8n", 5678),
     "routing": {"type": "port", "port": 5678, "sub": "n8n"}},
]
CATALOG_BY_ID = {c["id"]: c for c in CATALOG}
IMAGE_MODELS = {
    "sd15": {"label": "Stable Diffusion 1.5 (klein, kommerzielle Nutzung erlaubt)",
             "url": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly-fp16.safetensors",
             "file": "v1-5-pruned-emaonly-fp16.safetensors", "steps": 20, "cfg": 7.0, "size": "512x512"},
    "sd-turbo": {"label": "SD Turbo (schnell, Lizenz vor kommerzieller Nutzung prüfen)",
                 "url": "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors",
                 "file": "sd_turbo.safetensors", "steps": 4, "cfg": 1.0, "size": "512x512"},
    "sdxl-turbo": {"label": "SDXL Turbo (bessere Bilder, GPU empfohlen, Lizenz prüfen)",
                   "url": "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors",
                   "file": "sd_xl_turbo_1.0_fp16.safetensors", "steps": 4, "cfg": 1.0, "size": "512x512"},
}
WHISPER_MODELS = ["Systran/faster-whisper-small", "Systran/faster-whisper-medium",
                  "deepdml/faster-whisper-large-v3-turbo-ct2", "Systran/faster-distil-whisper-large-v3"]


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
        "dashboard": "auto",     # MyrmidLM dashboard for the monitor: auto (port 8090), a URL, or off
    },
    "rag": {
        "embedding_source": "local",   # local (container here) | backend (MyrmidLM /v1/embeddings) | custom
        "embedding_url": "",           # custom: OpenAI-compatible base URL (…/v1)
        "embedding_key": "",
        "embedding_model": "gpustack/bge-m3-GGUF:Q8_0",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 8,
        "hybrid": False,
        "docling": True,
        "docling_workers": 2,
        "ocr": "auto",           # auto | force | off
    },
    "chat": {
        # Side tasks (search queries, titles, tags) go to the model as extra
        # requests. With one model they wait in line and push the chat out of
        # its prompt cache: a small model of their own keeps the chat fast.
        "task_model": "",         # empty: the chat model itself
        "query_generation": False,
        "title": True,
        "tags": False,
        "follow_ups": True,
        "autocomplete": False,
    },
    "solutions": {},          # {id: {"enabled", "routing": {"type", "port", "sub"}, "settings": {}}}
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
        for k in ("pg_password", "qdrant_key", "webui_secret", "searxng_secret", "files_key"):
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
    out["rag"] = dict(cfg["rag"])
    out["rag"]["embedding_key_set"] = bool(out["rag"].get("embedding_key"))
    out["rag"]["embedding_key"] = ""
    return out


def normalize_backend_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if not re.match(r"^https?://", url):
        url = "http://" + url
    # a full endpoint pasted in (…/v1/embeddings, …/v1/chat/completions): keep the base
    url = re.sub(r"/(embeddings|chat/completions|completions|models)$", "", url).rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def dashboard_url(cfg: dict) -> str:
    d = (cfg["backend"].get("dashboard") or "auto").strip()
    if d == "off":
        return ""
    if d != "auto":
        return d.rstrip("/") if re.match(r"^https?://", d) else "http://" + d.rstrip("/")
    u = urllib.parse.urlparse(cfg["backend"]["url"] or "")
    return f"{u.scheme}://{u.hostname}:8090" if u.hostname else ""


def embedding_endpoint(cfg: dict) -> tuple[str, str, str]:
    """(base URL, key, model name) Open WebUI uses for the embeddings."""
    r = cfg["rag"]
    if r["embedding_source"] == "backend":
        return cfg["backend"]["url"], cfg["backend"]["key"] or "none", "bge-m3"
    if r["embedding_source"] == "custom":
        return normalize_backend_url(r["embedding_url"]), r["embedding_key"] or "none", "bge-m3"
    return "http://embeddings:8080/v1", "none", "bge-m3"


def solution_conf(cfg: dict, sid: str) -> dict:
    """Settings of a solution, with the catalog's defaults."""
    cat = CATALOG_BY_ID.get(sid, {})
    raw = (cfg.get("solutions") or {}).get(sid) or {}
    routing = dict(cat.get("routing") or {"type": "none", "port": 0, "sub": ""})
    routing.update(raw.get("routing") or {})
    settings = {"whisper": {"model": WHISPER_MODELS[0]},
                "files": {"keep_days": 7},
                "images": {"mode": "cpu", "model": "sd15", "size": "", "url": "", "key": "", "ext_model": ""},
                }.get(sid, {})
    settings.update(raw.get("settings") or {})
    return {"enabled": bool(cat.get("core") or raw.get("enabled")), "routing": routing, "settings": settings}


def custom_solutions(cfg: dict) -> list[dict]:
    return [c for c in (cfg.get("custom") or []) if c.get("id")]


def enabled_services(cfg: dict) -> list[str]:
    out = []
    for cat in CATALOG:
        if not solution_conf(cfg, cat["id"])["enabled"]:
            continue
        for svc in cat["services"]:
            if svc == "docling" and not cfg["rag"]["docling"]:
                continue
            if svc == "embeddings" and cfg["rag"]["embedding_source"] != "local":
                continue
            if svc in ("images", "images-model") and solution_conf(cfg, "images")["settings"]["mode"] in ("external", "backend"):
                continue
            out.append(svc)
    out += ["custom-" + c["id"] for c in custom_solutions(cfg) if c.get("enabled", True)]
    return out


def all_service_names(cfg: dict) -> list[str]:
    return list(IMAGES) + ["custom-" + c["id"] for c in custom_solutions(cfg)]


def web_routes(cfg: dict) -> list[dict]:
    """Solutions with their own web interface, reached through Caddy."""
    routes = []
    for cat in CATALOG:
        sc = solution_conf(cfg, cat["id"])
        if cat.get("web") and sc["enabled"] and sc["routing"].get("type") in ("port", "subdomain"):
            svc, port = cat["web"]
            routes.append({"id": cat["id"], "label": cat["label"], "upstream": f"{svc}:{port}", **sc["routing"]})
    for c in custom_solutions(cfg):
        r = c.get("routing") or {}
        if c.get("enabled", True) and c.get("port") and r.get("type") in ("port", "subdomain"):
            routes.append({"id": c["id"], "label": c.get("label") or c["id"],
                           "upstream": f"custom-{c['id']}:{int(c['port'])}", **r})
    return routes


def route_url(cfg: dict, r: dict) -> str:
    g = cfg["general"]
    https = g["access"] in ("internal", "acme", "proxy")
    scheme = "https" if https else "http"
    if r.get("type") == "subdomain":
        return f"{scheme}://{r.get('sub')}.{g['host']}"
    return f"{'https' if g['access'] in ('internal', 'acme') else 'http'}://{g['host']}:{int(r.get('port') or 0)}"


def test_embeddings(cfg: dict) -> dict:
    url, key, model = embedding_endpoint(cfg)
    if cfg["rag"]["embedding_source"] == "local":
        return {"ok": None, "info": "läuft im Container auf diesem Server (Test über die Dienste-Übersicht)"}
    req = urllib.request.Request(url + "/embeddings", data=json.dumps({"input": ["Formica Test"], "model": model}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        dim = len(data["data"][0]["embedding"])
        return {"ok": True, "url": url, "dim": dim, "ms": round((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", {}).get("message") or f"HTTP {e.code}"
        except Exception:  # noqa: BLE001
            msg = f"HTTP {e.code}"
        return {"ok": False, "url": url, "error": msg}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "url": url, "error": f"Nicht erreichbar: {e}"}


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
    if r["embedding_source"] not in ("local", "backend", "custom"):
        errs.append("Unbekannte Quelle für die Embeddings.")
    if r["embedding_source"] == "custom" and not r["embedding_url"]:
        errs.append("Bitte die Adresse des Embedding-Dienstes eintragen.")
    used_ports = {int(g["http_port"]), 8095}
    if g["access"] in ("internal", "acme"):
        used_ports.add(int(g["https_port"]))
    for rt in web_routes(cfg):
        if rt["type"] == "port":
            p = int(rt.get("port") or 0)
            if not (1 <= p <= 65535):
                errs.append(f"{rt['label']}: ungültiger Port {p}")
            elif p in used_ports:
                errs.append(f"{rt['label']}: Port {p} ist schon belegt")
            used_ports.add(p)
        elif not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", str(rt.get("sub") or "")):
            errs.append(f"{rt['label']}: ungültige Subdomain „{rt.get('sub')}“")
        elif re.match(r"^\d+\.\d+\.\d+\.\d+$", g["host"] or ""):
            errs.append(f"{rt['label']}: Subdomains brauchen einen DNS-Namen als Hostname, keine IP-Adresse")
    for c in custom_solutions(cfg):
        if not re.match(r"^[a-z0-9][a-z0-9_.\-/:@]*$", c.get("image", "")):
            errs.append(f"Eigene Lösung {c['id']}: ungültiges Image")
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
    if "files" in enabled_services(cfg):  # downloads of "Dateien erzeugen"
        upstream = ("handle_path /formica-files/* {\n\t\treverse_proxy files:8000\n\t}\n"
                    "\thandle {\n\t\treverse_proxy open-webui:8080\n\t}")
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
    for rt in web_routes(cfg):
        up = f"reverse_proxy {rt['upstream']}"
        if rt["type"] == "port":
            p = int(rt["port"])
            addr = f":{p}" if g["access"] in ("http", "proxy") else f"{g['host']}:{p}"
        else:
            name = f"{rt['sub']}.{g['host']}"
            addr = f"http://{name}" if g["access"] in ("http", "proxy") else name
        tls = "\ttls internal\n" if g["access"] == "internal" else ""
        site += f"\n# {rt['label']}\n{addr} {{\n{tls}\t{up}\n}}\n"
    return "# Erzeugt von Formica. Änderungen bitte in der Formica-Oberfläche.\n" + head + site


def render_compose(cfg: dict) -> dict:
    g, b, r, u, s = cfg["general"], cfg["backend"], cfg["rag"], cfg["users"], cfg["secrets"]
    c = cfg["chat"]
    https = g["access"] in ("internal", "acme")

    caddy_ports = [f"{int(g['http_port'])}:80"]
    if https:
        caddy_ports.append(f"{int(g['https_port'])}:443")
    for rt in web_routes(cfg):
        if rt["type"] == "port":
            if f"{int(rt['port'])}:{int(rt['port'])}" not in caddy_ports:
                caddy_ports.append(f"{int(rt['port'])}:{int(rt['port'])}")
    active = enabled_services(cfg)

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
        "RAG_OPENAI_API_BASE_URL": embedding_endpoint(cfg)[0],
        "RAG_OPENAI_API_KEY": embedding_endpoint(cfg)[1],
        "RAG_EMBEDDING_MODEL": embedding_endpoint(cfg)[2],
        "RAG_EMBEDDING_BATCH_SIZE": "16",
        "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER": "true",
        "CHUNK_SIZE": str(int(r["chunk_size"])),
        "CHUNK_OVERLAP": str(int(r["chunk_overlap"])),
        "RAG_TOP_K": str(int(r["top_k"])),
        "RAG_RELEVANCE_THRESHOLD": "0",
        "ENABLE_RAG_HYBRID_SEARCH": "true" if r["hybrid"] else "false",
        "RAG_TEMPLATE": RAG_TEMPLATE,
        "ENABLE_RETRIEVAL_QUERY_GENERATION": "true" if c["query_generation"] else "false",
        "ENABLE_TITLE_GENERATION": "true" if c["title"] else "false",
        "ENABLE_TAGS_GENERATION": "true" if c["tags"] else "false",
        "ENABLE_FOLLOW_UP_GENERATION": "true" if c["follow_ups"] else "false",
        "ENABLE_AUTOCOMPLETE_GENERATION": "true" if c["autocomplete"] else "false",
        "ANONYMIZED_TELEMETRY": "false",
        # Updates come with Formica (tested versions), not from the banner in Open WebUI
        "ENABLE_VERSION_UPDATE_CHECK": "false",
    }
    if c["task_model"]:
        webui_env["TASK_MODEL_EXTERNAL"] = c["task_model"]
    if "searxng" in active:
        webui_env.update({"ENABLE_WEB_SEARCH": "true", "WEB_SEARCH_ENGINE": "searxng",
                          "SEARXNG_QUERY_URL": "http://searxng:8080/search?q=<query>",
                          "WEB_SEARCH_RESULT_COUNT": "5"})
    if "files" in active:
        webui_env["TOOL_SERVER_CONNECTIONS"] = json.dumps([{
            "type": "openapi", "url": "http://files:8000", "spec_type": "url", "spec": "", "path": "openapi.json",
            "auth_type": "bearer", "key": s["files_key"], "config": {"enable": True, "access_control": None},
            "info": {"id": "formica-files", "name": "Dateien erzeugen",
                     "description": "Excel, Word, PDF und CSV erstellen und als Download-Link zurückgeben"}}])
    im = solution_conf(cfg, "images")
    if im["enabled"]:
        ims = im["settings"]
        preset = IMAGE_MODELS.get(ims.get("model"), IMAGE_MODELS["sd15"])
        if ims.get("mode") == "backend":  # MyrmidLM: a host's image model behind the master's /v1/images/...
            base, key, model = cfg["backend"]["url"], cfg["backend"]["key"] or "none", ims.get("model") or "sd15"
        elif ims.get("mode") == "external":
            base, key, model = normalize_backend_url(ims.get("url", "")), ims.get("key") or "none", ims.get("ext_model") or "dall-e-3"
        else:
            base, key, model = "http://images:1234/v1", "none", ims.get("model") or "sd15"
        webui_env.update({"ENABLE_IMAGE_GENERATION": "true", "IMAGE_GENERATION_ENGINE": "openai",
                          "IMAGES_OPENAI_API_BASE_URL": base, "IMAGES_OPENAI_API_KEY": key,
                          "IMAGE_GENERATION_MODEL": model, "IMAGE_SIZE": ims.get("size") or preset["size"],
                          "ENABLE_IMAGE_PROMPT_GENERATION": "false"})
    if "whisper" in active:
        webui_env.update({"AUDIO_STT_ENGINE": "openai", "AUDIO_STT_OPENAI_API_BASE_URL": "http://whisper:8000/v1",
                          "AUDIO_STT_OPENAI_API_KEY": "none",
                          "AUDIO_STT_MODEL": solution_conf(cfg, "whisper")["settings"]["model"]})
    if r["docling"]:
        webui_env["CONTENT_EXTRACTION_ENGINE"] = "docling"
        webui_env["DOCLING_SERVER_URL"] = "http://docling:5001"
        params = {"do_ocr": r["ocr"] != "off", "table_mode": "accurate"}
        if r["ocr"] != "off":
            params["force_ocr"] = r["ocr"] == "force"
            params["ocr_lang"] = ["de", "en"]
        webui_env["DOCLING_PARAMS"] = json.dumps(params)

    local_embed = r["embedding_source"] == "local"
    depends = {"postgres": {"condition": "service_healthy"},
               "qdrant": {"condition": "service_started"}}
    if local_embed:
        depends["embeddings"] = {"condition": "service_started"}
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
    if not local_embed:
        services.pop("embeddings")
    if r["docling"]:
        services["docling"] = {
            "image": IMAGES["docling"],
            "restart": "unless-stopped",
            "environment": {"DOCLING_SERVE_MAX_SYNC_WAIT": "600",
                            "DOCLING_SERVE_ENG_LOC_NUM_WORKERS": str(int(r["docling_workers"]))},
            "networks": ["formica"],
        }

    if "searxng" in active:
        services["searxng"] = {
            "image": IMAGES["searxng"], "restart": "unless-stopped",
            "volumes": ["./searxng/settings.yml:/etc/searxng/settings.yml:ro"],
            "networks": ["formica"],
        }
    if "whisper" in active:
        model = solution_conf(cfg, "whisper")["settings"]["model"]
        services["whisper"] = {
            "image": IMAGES["whisper"], "restart": "unless-stopped",
            "environment": {"PRELOAD_MODELS": json.dumps([model]), "WHISPER__COMPUTE_TYPE": "int8",
                            "STT_MODEL_TTL": "-1"},
            "volumes": ["whisper_models:/home/ubuntu/.cache/huggingface/hub"],
            "networks": ["formica"],
        }
    if "n8n" in active:
        rt = next((x for x in web_routes(cfg) if x["id"] == "n8n"), None)
        url = route_url(cfg, rt) if rt else ""
        n8n_env = {"GENERIC_TIMEZONE": "Europe/Berlin", "TZ": "Europe/Berlin", "N8N_PORT": "5678",
                   "N8N_DIAGNOSTICS_ENABLED": "false", "N8N_PERSONALIZATION_ENABLED": "false",
                   "N8N_SECURE_COOKIE": "true" if url.startswith("https") else "false"}
        if url:
            u = urllib.parse.urlparse(url)
            n8n_env.update({"N8N_HOST": u.hostname or "", "N8N_PROTOCOL": u.scheme,
                            "WEBHOOK_URL": url.rstrip("/") + "/", "N8N_EDITOR_BASE_URL": url.rstrip("/") + "/"})
        services["n8n"] = {
            "image": IMAGES["n8n"], "restart": "unless-stopped", "environment": n8n_env,
            "volumes": ["n8n_data:/home/node/.n8n"], "networks": ["formica"],
        }
    if "files" in active:
        services["files"] = {
            "build": {"context": str(HOME / "app" / "services" / "files")},
            "image": IMAGES["files"], "restart": "unless-stopped",
            "environment": {"FILES_PUBLIC_URL": webui_url(cfg).rstrip("/") + "/formica-files",
                            "FILES_API_KEY": s["files_key"],
                            "FILES_KEEP_DAYS": str(solution_conf(cfg, "files")["settings"].get("keep_days") or 7)},
            "volumes": ["files_data:/data"], "networks": ["formica"],
        }
    if "images" in active:
        ims = solution_conf(cfg, "images")["settings"]
        p = IMAGE_MODELS.get(ims.get("model"), IMAGE_MODELS["sd15"])
        cuda = ims.get("mode") == "cuda"
        services["images-model"] = {
            "image": IMAGES["images-model"], "user": "0:0", "restart": "no",
            "entrypoint": ["sh", "-c"],
            "command": [f'test -s /models/{p["file"]} && echo "Modell vorhanden: {p["file"]}" || '
                        f'(echo "Lade {p["file"]} ..." && curl -fL --retry 3 -o /models/{p["file"]}.part "{p["url"]}" '
                        f'&& mv /models/{p["file"]}.part /models/{p["file"]} && echo fertig)'],
            "volumes": ["images_models:/models"], "networks": ["formica"],
        }
        run_sd = (f"exec /sd.cpp/bin/sd-server --listen-ip 0.0.0.0 --listen-port 1234 --model-dir /models "
                  f"-m /models/{p['file']} --steps {p['steps']} --cfg-scale {p['cfg']}")
        svc = {
            "image": IMAGES_CUDA if cuda else IMAGES["images"], "restart": "unless-stopped",
            "entrypoint": ["sh", "-c"],
            # without a GPU: drop the Vulkan backend, so the CPU backend does the work
            "command": [run_sd if cuda else "rm -f /sd.cpp/bin/libggml-vulkan*.so; " + run_sd],
            "depends_on": {"images-model": {"condition": "service_completed_successfully"}},
            "volumes": ["images_models:/models"], "networks": ["formica"],
        }
        if cuda:
            svc["deploy"] = {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]}}}
        services["images"] = svc
    custom_vols = []
    for cs in custom_solutions(cfg):
        if not cs.get("enabled", True):
            continue
        vols = []
        for v in cs.get("volumes") or []:
            vname = f"custom_{cs['id']}_{v['name']}"
            custom_vols.append(vname)
            vols.append(f"{vname}:{v['path']}")
        svc = {"image": cs["image"], "restart": "unless-stopped", "networks": ["formica"]}
        if cs.get("env"):
            svc["environment"] = dict(cs["env"])
        if vols:
            svc["volumes"] = vols
        if cs.get("gpu"):
            svc["deploy"] = {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]}}}
        services["custom-" + cs["id"]] = svc

    for name, extra in (cfg.get("extra_env") or {}).items():
        if name in services and extra:
            services[name].setdefault("environment", {}).update(extra)

    return {
        "name": PROJECT,
        "services": services,
        "volumes": {v: {} for v in ("caddy_data", "caddy_config", "openwebui_data", "postgres_data",
                                    "qdrant_data", "embeddings_models", "whisper_models", "n8n_data",
                                    "files_data", "images_models",
                                    *custom_vols)},
        "networks": {"formica": {}},
    }


# Own environment variables (Einstellungen > Erweitert): per service, on top of Formica's
EXTRA_SERVICES = ("open-webui", "docling", "qdrant", "embeddings", "postgres", "caddy")
ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# changing these breaks the wiring between the services (empty start, no documents)
CRITICAL_ENV = {"DATABASE_URL", "VECTOR_DB", "QDRANT_URI", "QDRANT_API_KEY", "WEBUI_SECRET_KEY",
                "POSTGRES_PASSWORD", "POSTGRES_USER", "POSTGRES_DB", "QDRANT__SERVICE__API_KEY",
                "ENABLE_PERSISTENT_CONFIG", "LLAMA_CACHE"}


def clean_extra_env(raw) -> tuple[dict, list[str]]:
    """{service: [{key, value}]} from the form -> {service: {KEY: value}}, errors."""
    out, errs = {}, []
    if not isinstance(raw, dict):
        return out, errs
    for svc, rows in raw.items():
        if svc not in EXTRA_SERVICES or not isinstance(rows, list):
            continue
        env = {}
        for row in rows:
            key = str((row or {}).get("key", "")).strip()
            val = str((row or {}).get("value", ""))
            if not key:
                continue
            if not ENV_KEY.match(key):
                errs.append(f"{svc}: ungültiger Name „{key}“ (nur Buchstaben, Ziffern, _)")
                continue
            if "\n" in val:
                errs.append(f"{svc}: {key} darf keinen Zeilenumbruch enthalten")
                continue
            env[key] = val
        if env:
            out[svc] = env
    return out, errs


SECRET_NAME = re.compile(r"(KEY|SECRET|PASSWORD|TOKEN)", re.I)


def managed_env(cfg: dict) -> dict:
    """The variables Formica itself sets, per service: name, value, secret or not."""
    base = dict(cfg, extra_env={})
    secrets_ = {v for v in cfg["secrets"].values() if v} | {v for v in (cfg["backend"]["key"],
                                                                         cfg["rag"].get("embedding_key")) if v}
    out = {}
    for name, svc in render_compose(base)["services"].items():
        rows = []
        for k, v in sorted((svc.get("environment") or {}).items()):
            v = str(v)
            secret = bool(SECRET_NAME.search(k)) and v not in ("none", "") or any(x in v for x in secrets_)
            rows.append({"key": k, "value": v, "secret": secret})
        out[name] = rows
    return out


def render(cfg: dict) -> None:
    (STACK / "caddy").mkdir(parents=True, exist_ok=True)
    head = ("# Erzeugt von Formica " + version() + ". Nicht von Hand ändern:\n"
            "# Formica schreibt die Datei bei jedem Übernehmen neu.\n")
    compose = STACK / "compose.yaml"
    compose.write_text(head + to_yaml(render_compose(cfg)) + "\n")
    os.chmod(compose, 0o600)  # holds the database passwords
    (STACK / "caddy" / "Caddyfile").write_text(render_caddyfile(cfg))
    (STACK / "searxng").mkdir(parents=True, exist_ok=True)
    (STACK / "searxng" / "settings.yml").write_text(
        "# Erzeugt von Formica\nuse_default_settings: true\n"
        f"server:\n  secret_key: \"{cfg['secrets']['searxng_secret']}\"\n  limiter: false\n  image_proxy: false\n"
        "search:\n  safe_search: 0\n  formats:\n    - html\n    - json\n"
        "ui:\n  default_locale: de\n")


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
        event(f"{name} gestartet")
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
        event(f"{self.name}: {'erfolgreich' if ok else 'fehlgeschlagen'}", "info" if ok else "error")
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
    steps.append(compose_cmd("up", "-d", "--remove-orphans", "--build"))
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
    active = set(enabled_services(cfg))
    rows = [(n, l, d, o, IMAGES[n]) for n, l, d, o in MODULES]
    rows += [("custom-" + c["id"], c.get("label") or c["id"], c.get("desc") or "Eigene Lösung", True, c["image"])
             for c in custom_solutions(cfg)]
    out = []
    for name, label, desc, optional, image in rows:
        enabled = name in active
        st = state.get(name, {})
        if not enabled and not st:
            continue  # not part of this installation
        out.append({"name": name, "label": label, "desc": desc, "optional": optional,
                    "enabled": enabled, "state": st.get("state", "fehlt" if enabled else "aus"),
                    "status": st.get("status", ""), "health": st.get("health", ""), "image": image})
    return out


def solutions_status(cfg: dict) -> list[dict]:
    svc = {s["name"]: s for s in services_status()}
    out = []

    def overall(names, enabled):
        if not enabled:
            return "available"
        states = [svc.get(n, {}).get("state", "fehlt") for n in names if not n.endswith("-model")]
        if not states:  # nothing of its own runs here (e.g. images on MyrmidLM or an external service)
            return "running"
        if all(x == "running" for x in states):
            return "running"
        if any(x == "running" for x in states):
            return "partial"
        return "stopped"
    for cat in CATALOG:
        sc = solution_conf(cfg, cat["id"])
        members = [n for n in cat["services"] if n in enabled_services(cfg)] if sc["enabled"] else cat["services"]
        rt = next((r for r in web_routes(cfg) if r["id"] == cat["id"]), None)
        out.append({"id": cat["id"], "label": cat["label"], "desc": cat["desc"], "icon": cat["icon"],
                    "core": bool(cat.get("core")), "custom": False, "enabled": sc["enabled"],
                    "routing": sc["routing"] if cat.get("web") else None,
                    "settings": {**{k: v for k, v in sc["settings"].items() if k != "key"},
                                 **({"key_set": bool(sc["settings"].get("key"))} if "key" in sc["settings"] else {})},
                    "url": webui_url(cfg) if cat["id"] == "chat" else (route_url(cfg, rt) if rt else None),
                    "state": overall(members, sc["enabled"]),
                    "members": [{"name": n, "label": next((m[1] for m in MODULES if m[0] == n), n),
                                 "state": svc.get(n, {}).get("state", "aus"), "image": IMAGES.get(n, "")}
                                for n in members]})
    for c in custom_solutions(cfg):
        name = "custom-" + c["id"]
        rt = next((r for r in web_routes(cfg) if r["id"] == c["id"]), None)
        out.append({"id": c["id"], "label": c.get("label") or c["id"], "desc": c.get("desc") or c["image"],
                    "icon": "box", "core": False, "custom": True, "enabled": c.get("enabled", True),
                    "routing": c.get("routing"), "settings": {}, "url": route_url(cfg, rt) if rt else None,
                    "state": overall([name], c.get("enabled", True)), "spec": c,
                    "members": [{"name": name, "label": c.get("label") or c["id"],
                                 "state": svc.get(name, {}).get("state", "aus"), "image": c["image"]}]})
    return out


def parse_custom(data: dict, existing: list[dict]) -> tuple[dict | None, str]:
    """A custom solution from the form."""
    label = str(data.get("label") or "").strip()
    cid = str(data.get("id") or "").strip() or re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not re.match(r"^[a-z0-9][a-z0-9-]{0,30}$", cid):
        return None, "Name: bitte Buchstaben und Ziffern verwenden (wird zur ID)."
    if cid in CATALOG_BY_ID or cid in ("formica", "control"):
        return None, f"„{cid}“ ist schon vergeben."
    image = str(data.get("image") or "").strip()
    if not re.match(r"^[a-z0-9][a-z0-9_.\-/:@]*$", image):
        return None, "Image: z. B. nginx:alpine oder ghcr.io/user/app:1.2"
    port = int(data.get("port") or 0)
    if port and not (1 <= port <= 65535):
        return None, "Interner Port zwischen 1 und 65535."
    env, _errs = clean_extra_env({"open-webui": [
        {"key": ln.split("=", 1)[0].strip(), "value": ln.split("=", 1)[1] if "=" in ln else ""}
        for ln in str(data.get("env") or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]})
    if _errs:
        return None, "Umgebungsvariablen: " + "; ".join(e.split(": ", 1)[-1] for e in _errs)
    vols = []
    for ln in str(data.get("volumes") or "").splitlines():
        if not ln.strip():
            continue
        name, _, path = ln.partition(":")
        name, path = name.strip(), path.strip()
        if not re.match(r"^[a-z0-9][a-z0-9_-]*$", name) or not path.startswith("/"):
            return None, f"Volume „{ln.strip()}“: Format name:/pfad/im/container"
        vols.append({"name": name, "path": path})
    rt = data.get("routing") or {}
    routing = {"type": rt.get("type") if rt.get("type") in ("none", "port", "subdomain") else "none",
               "port": int(rt.get("port") or 0), "sub": str(rt.get("sub") or cid).strip()}
    if routing["type"] != "none" and not port:
        return None, "Für den Zugriff im Browser bitte den internen Port der Anwendung angeben."
    old = next((c for c in existing if c["id"] == cid), {})
    return {"id": cid, "label": label or cid, "desc": str(data.get("desc") or "").strip(), "image": image,
            "port": port, "env": env.get("open-webui", {}), "volumes": vols, "routing": routing,
            "gpu": bool(data.get("gpu")), "enabled": old.get("enabled", True)}, ""


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

def _read(p) -> str:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return ""


def cpu_temp() -> float | None:
    """CPU package temperature like MyrmidLM: hwmon, else thermal zones. None in VMs."""
    best = None
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if _read(hw / "name") not in ("k10temp", "coretemp", "zenpower", "cpu_thermal", "acpitz"):
            continue
        temps = {}
        for f in hw.glob("temp*_input"):
            label = _read(hw / f.name.replace("_input", "_label")) or f.name
            try:
                temps[label] = int(_read(f)) / 1000
            except ValueError:
                pass
        pick = next((v for k, v in temps.items() if k in ("Tdie", "Tctl")), None) \
            or max((v for k, v in temps.items() if k.startswith("Package")), default=None) \
            or next((v for _, v in sorted(temps.items())), None)
        if pick is not None and (best is None or pick > best):
            best = pick
    if best is None:
        for z in Path("/sys/class/thermal").glob("thermal_zone*"):
            if _read(z / "type") in ("x86_pkg_temp", "cpu-thermal", "cpu_thermal", "soc-thermal"):
                try:
                    v = int(_read(z / "temp")) / 1000
                except ValueError:
                    continue
                best = v if best is None else max(best, v)
    return round(best, 1) if best is not None else None


def cpu_name() -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            return re.sub(r"\s+", " ", line.split(":", 1)[1]).strip()
    return ""


DISK_RE = re.compile(r"^(sd[a-z]+|vd[a-z]+|xvd[a-z]+|hd[a-z]+|nvme\d+n\d+|mmcblk\d+)$")
NET_SKIP = re.compile(r"^(lo|docker\d*|br-.*|veth.*|virbr.*|cni.*|flannel.*|tap.*|fw.*)$")


def _disk_bytes() -> tuple[int, int]:
    r = w = 0
    for line in _read("/proc/diskstats").splitlines():
        f = line.split()
        if len(f) > 9 and DISK_RE.match(f[2]):
            r += int(f[5]) * 512
            w += int(f[9]) * 512
    return r, w


def _net_bytes() -> tuple[int, int]:
    rx = tx = 0
    for line in _read("/proc/net/dev").splitlines()[2:]:
        name, data = line.split(":", 1)
        if NET_SKIP.match(name.strip()):
            continue
        f = data.split()
        rx += int(f[0])
        tx += int(f[8])
    return rx, tx


def _size(txt: str) -> float:
    """'1.5GiB' / '12.3kB' -> bytes (docker stats)."""
    m = re.match(r"([\d.]+)\s*([kKMGTP]?i?B)?", txt.strip())
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = (m.group(2) or "B")
    mult = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
            "KiB": 1024, "kiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
    return n * mult.get(unit, 1)


class Monitor:
    """Live figures of this server, its containers and the model cluster,
    with five minutes of history (2 s steps). The expensive parts (docker
    stats, the cluster) only run while someone looks at the monitor."""

    HIST = 150

    def __init__(self):
        self.lock = threading.Lock()
        self.pct = None            # CPU usage, also used by the overview
        self.cores: list[float] = []
        self.host: dict = {"cpu_name": cpu_name()}
        self.hist = {k: [] for k in ("cpu", "ram", "disk_r", "disk_w", "net_rx", "net_tx", "temp")}
        self.containers: list[dict] = []
        self.cluster: dict = {}
        self.gpu = None
        self.gpu_hist: list = []
        self.viewed = 0.0
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._slow_loop, daemon=True).start()

    @staticmethod
    def _cpu_rows():
        rows = {}
        for line in _read("/proc/stat").splitlines():
            if line.startswith("cpu"):
                name, *vals = line.split()
                v = [int(x) for x in vals[:8]]
                rows[name] = (v[3] + v[4], sum(v))
        return rows

    def _push(self, k, v):
        h = self.hist[k]
        h.append(v)
        if len(h) > self.HIST:
            del h[0]

    def _loop(self):
        prev, t0 = self._cpu_rows(), time.time()
        d0, n0 = _disk_bytes(), _net_bytes()
        while True:
            time.sleep(2)
            cur, t1 = self._cpu_rows(), time.time()
            d1, n1 = _disk_bytes(), _net_bytes()
            span = max(0.5, t1 - t0)

            def usage(k):
                if k not in prev:
                    return 0.0
                di, dt = cur[k][0] - prev[k][0], cur[k][1] - prev[k][1]
                return round(100 * (1 - di / dt), 1) if dt > 0 else 0.0
            mem = {}
            for line in _read("/proc/meminfo").splitlines():
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0]) * 1024
            with self.lock:
                self.pct = usage("cpu")
                self.cores = [usage(k) for k in sorted((k for k in cur if k != "cpu"), key=lambda x: int(x[3:]))]
                total, avail = mem.get("MemTotal", 0), mem.get("MemAvailable", 0)
                self.host.update({
                    "cpu_pct": self.pct, "cores": self.cores, "cpu_temp_c": cpu_temp(),
                    "load": [float(x) for x in _read("/proc/loadavg").split()[:3]] or None,
                    "ram_total": total, "ram_used": total - avail,
                    "swap_total": mem.get("SwapTotal", 0), "swap_used": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
                    "disk_r": max(0, d1[0] - d0[0]) / span, "disk_w": max(0, d1[1] - d0[1]) / span,
                    "net_rx": max(0, n1[0] - n0[0]) / span, "net_tx": max(0, n1[1] - n0[1]) / span,
                    "uptime": float((_read("/proc/uptime").split() or ["0"])[0]),
                })
                du = shutil.disk_usage(HOME if HOME.exists() else "/")
                self.host["disk_total"], self.host["disk_used"] = du.total, du.total - du.free
                self._push("cpu", self.pct)
                self._push("ram", round(100 * (total - avail) / total, 1) if total else None)
                self._push("disk_r", self.host["disk_r"])
                self._push("disk_w", self.host["disk_w"])
                self._push("net_rx", self.host["net_rx"])
                self._push("net_tx", self.host["net_tx"])
                self._push("temp", self.host["cpu_temp_c"])
            prev, t0, d0, n0 = cur, t1, d1, n1

    def _slow_loop(self):
        last_cluster = 0.0
        while True:
            watching = time.time() - self.viewed < 30
            if watching:
                try:
                    self._docker_stats()
                except Exception as e:  # noqa: BLE001
                    log(f"Monitor: docker stats: {e}")
            # the cluster always (the cockpit shows its GPU history), more often while watched
            if watching or time.time() - last_cluster >= 10:
                last_cluster = time.time()
                try:
                    self._cluster()
                except Exception as e:  # noqa: BLE001
                    self.cluster = {"error": str(e)[:200]}
                gpus = [g for h in self.cluster.get("hosts", []) for g in (h.get("stats") or {}).get("gpus") or []]
                util = [g.get("util_pct") for g in gpus if g.get("util_pct") is not None]
                self.gpu = round(sum(util) / len(util), 1) if util else None
                self.gpu_hist.append({"t": time.time(), "v": self.gpu})
                self.gpu_hist = [x for x in self.gpu_hist if time.time() - x["t"] <= 300]
            time.sleep(3)

    def _docker_stats(self):
        code, out = run(["docker", "stats", "--no-stream", "--format", "{{json .}}"], timeout=30)
        if code != 0:
            return
        names = {"formica-control": "formica"}
        rows = []
        for line in out.splitlines():
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            name = d.get("Name", "")
            m = re.match(rf"^{PROJECT}-(.+)-\d+$", name)
            svc = m.group(1) if m else names.get(name)
            if not svc:
                continue
            used, _, limit = d.get("MemUsage", "").partition("/")
            net_in, _, net_out = d.get("NetIO", "").partition("/")
            blk_r, _, blk_w = d.get("BlockIO", "").partition("/")
            rows.append({"service": svc, "cpu": float(d.get("CPUPerc", "0%").rstrip("%") or 0),
                         "mem": _size(used), "mem_limit": _size(limit), "mem_pct": float(d.get("MemPerc", "0%").rstrip("%") or 0),
                         "net_in": _size(net_in), "net_out": _size(net_out),
                         "blk_r": _size(blk_r), "blk_w": _size(blk_w), "pids": int(d.get("PIDs") or 0)})
        order = [m[0] for m in MODULES] + ["formica"]
        rows.sort(key=lambda r: order.index(r["service"]) if r["service"] in order else 99)
        self.containers = rows

    def _cluster(self):
        cfg = load_config()
        url = dashboard_url(cfg)
        if not url:
            self.cluster = {}
            return
        key = cfg["backend"]["key"]
        hdr = {"Authorization": f"Bearer {key}"} if key else {}

        def get(path):
            with urllib.request.urlopen(urllib.request.Request(url + path, headers=hdr), timeout=4) as r:
                return json.loads(r.read())
        st = get("/api/status")
        mon = get("/api/monitor")
        hosts, seen = [], set()
        all_hosts = ((st.get("hosts") or {}).get("hosts") or [])
        for h in all_hosts:
            s = h.get("stats") or {}
            if not s.get("host") or s["host"] in seen:
                continue
            seen.add(s["host"])
            hosts.append({"name": s["host"], "stats": s,
                          "wheres": [x.get("name") for x in all_hosts if (x.get("stats") or {}).get("host") == s["host"]]})
        self.cluster = {"url": url, "hosts": hosts, "requests": (mon.get("requests") or [])[:200],
                        "totals": mon.get("totals") or {}, "time": time.time()}

    def snapshot(self, light: bool = False) -> dict:
        if not light:
            self.viewed = time.time()
        with self.lock:
            return {"host": dict(self.host), "hist": {k: list(v) for k, v in self.hist.items()},
                    "containers": self.containers, "cluster": self.cluster,
                    "gpu": self.gpu, "gpu_hist": list(self.gpu_hist)}


MON = None


# --------------------------------------------------------------------------
# Updates of Formica itself
# --------------------------------------------------------------------------

UPDATE_URL = os.environ.get("FORMICA_UPDATE_URL", "https://raw.githubusercontent.com/Miki84/Formica/main")
_update = {"checked": 0.0, "latest": None, "notes": [], "error": None}


def vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3]) or (0,)


def parse_changelog(text: str, newer_than: str) -> list[dict]:
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^##\s+v?(\d+\.\d+\.\d+)", line)
        if m:
            cur = {"version": m.group(1), "items": []}
            if vtuple(cur["version"]) > vtuple(newer_than):
                out.append(cur)
            else:
                cur = None
        elif cur and line.strip().startswith("- "):
            cur["items"].append(line.strip()[2:])
    return out


def check_update(force: bool = False) -> dict:
    if force or time.time() - _update["checked"] > 3600:
        try:
            with urllib.request.urlopen(UPDATE_URL + "/VERSION", timeout=8) as r:
                latest = r.read().decode().strip()
            notes = []
            try:
                with urllib.request.urlopen(UPDATE_URL + "/CHANGELOG.md", timeout=8) as r:
                    notes = parse_changelog(r.read().decode(), version())
            except Exception:  # noqa: BLE001 - notes are optional
                pass
            _update.update(latest=latest, notes=notes, error=None)
        except Exception as e:  # noqa: BLE001
            _update["error"] = f"GitHub nicht erreichbar: {e}"
        _update["checked"] = time.time()
    latest = _update["latest"]
    return {"current": version(), "latest": latest,
            "available": bool(latest) and vtuple(latest) > vtuple(version()),
            "notes": _update["notes"], "error": _update["error"],
            "checked": _update["checked"], "git": (HOME / "app" / ".git").exists()}


def self_update() -> tuple[bool, str]:
    """Runs app/update.sh in a helper container: it may replace this one."""
    if not (HOME / "app" / ".git").exists():
        return False, "Formica wurde ohne Git installiert. Bitte einmal neu installieren (install.sh)."
    run(["docker", "rm", "-f", "formica-updater"], timeout=20)
    home = str(HOME)
    script = (f"command -v git >/dev/null || apk add --no-cache git >/dev/null 2>&1; "
              f"cp {home}/app/update.sh /tmp/update.sh && sh /tmp/update.sh")
    code, out = run(["docker", "run", "-d", "--rm", "--name", "formica-updater",
                     "-v", "/var/run/docker.sock:/var/run/docker.sock", "-v", f"{home}:{home}",
                     "-e", f"FORMICA_HOME={home}", "--entrypoint", "sh", "formica-control:local", "-c", script],
                    timeout=60)
    return code == 0, out.strip()


def migrate_control() -> None:
    """Older installations run the control panel with port mapping. Since
    0.2.0 it runs in the host network (real NIC figures for the monitor):
    rewrite control.yaml once through update.sh."""
    ctl, mark = HOME / "control.yaml", HOME / ".hostnet-migrated"
    if not ctl.exists() or "network_mode: host" in ctl.read_text() or mark.exists():
        return
    if not (HOME / "app" / ".git").exists():
        return
    mark.write_text(time.strftime("%Y-%m-%d %H:%M"))

    def later():
        time.sleep(8)
        for _ in range(60):  # an update that is still checking on us: let it finish
            code, out = run(["docker", "inspect", "-f", "{{.State.Running}}", "formica-updater"], timeout=10)
            if code != 0 or out.strip() != "true":
                break
            time.sleep(3)
        ok, out = self_update()
        log("Umstellung auf Host-Netz gestartet" if ok else f"Umstellung auf Host-Netz fehlgeschlagen: {out}")
    threading.Thread(target=later, daemon=True).start()


def system_info() -> dict:
    info = {"version": version(), "cpus": os.cpu_count(),
            "cpu_pct": MON.pct if MON else None, "cpu_cores": MON.cores if MON else []}
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


ACTION_DE = {"start": "gestartet", "stop": "gestoppt", "restart": "neu gestartet"}


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
            if (section, k) in (("backend", "key"), ("rag", "embedding_key")) and v == "":
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
                                    "errors": validate(cfg), "applied": cfg.get("applied"),
                                    "managed_env": managed_env(cfg), "critical_env": sorted(CRITICAL_ENV)})
        if path == "/api/services":
            return self._json(200, {"services": services_status(), "job": JOB.state()})
        if path == "/api/job":
            return self._json(200, JOB.state())
        if path == "/api/solutions":
            cfg = load_config()
            return self._json(200, {"solutions": solutions_status(cfg), "whisper_models": WHISPER_MODELS,
                                    "image_models": {k: v["label"] for k, v in IMAGE_MODELS.items()},
                                    "job": JOB.state()})
        if path == "/api/events":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return self._json(200, {"events": read_events(int(q.get("limit", ["200"])[0]), q.get("q", [""])[0],
                                                          q.get("level", [""])[0])})
        if path == "/api/cockpit":
            cfg = load_config()
            snap = MON.snapshot(light=True) if MON else {}
            return self._json(200, {"solutions": solutions_status(cfg), "events": read_events(8),
                                    "host": snap.get("host", {}), "gpu_hist": snap.get("gpu_hist", []),
                                    "gpu": snap.get("gpu"), "update": check_update(),
                                    "security": {"password": bool(cfg.get("admin")), "totp": bool((cfg.get("admin") or {}).get("totp")),
                                                 "access": cfg["general"]["access"], "url": webui_url(cfg)},
                                    "job": JOB.state(), "applied": cfg.get("applied")})
        if path == "/api/monitor":
            return self._json(200, MON.snapshot() if MON else {})
        if path == "/api/system":
            return self._json(200, system_info())
        if path == "/api/update-check":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return self._json(200, check_update(force=q.get("force") == ["1"]))
        if path == "/api/update-log":
            f = HOME / "update.log"
            return self._json(200, {"log": f.read_text()[-20000:] if f.exists() else ""})
        if path.startswith("/api/logs/"):
            name = path.rsplit("/", 1)[-1]
            if name not in all_service_names(load_config()):
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
            event("Einrichtung abgeschlossen: Admin-Passwort gesetzt")
            return self._json(200, {"ok": True}, self._new_session())

        if path == "/api/login":
            if too_many_attempts(ip):
                return self._json(429, {"error": "Zu viele Versuche. Bitte 5 Minuten warten."})
            if check_password(load_config(), str(data.get("password", ""))):
                FAILED.pop(ip, None)
                event(f"Anmeldung von {ip}", source="sicherheit")
                return self._json(200, {"ok": True}, self._new_session())
            FAILED.setdefault(ip, []).append(time.time())
            event(f"Fehlgeschlagene Anmeldung von {ip}", "warn", "sicherheit")
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
            event("Admin-Passwort geändert", source="sicherheit")
            return self._json(200, {"ok": True}, self._new_session())
        if path == "/api/config":
            cfg = load_config()
            try:
                cfg = update_from_form(cfg, data)
            except (TypeError, ValueError) as e:
                return self._json(400, {"error": f"Ungültiger Wert: {e}"})
            if "extra_env" in data:
                extra, env_errs = clean_extra_env(data["extra_env"])
                if env_errs:
                    return self._json(400, {"error": "Eigene Umgebungsvariablen: " + "; ".join(env_errs)})
                cfg["extra_env"] = extra
            errs = validate(cfg)
            save_config(cfg)
            event("Einstellungen gespeichert")
            return self._json(200, {"ok": not errs, "errors": errs, "config": public_config(cfg),
                                    "webui_url": webui_url(cfg), "managed_env": managed_env(cfg),
                                    "critical_env": sorted(CRITICAL_ENV)})
        if path == "/api/test-embeddings":
            cfg = load_config()
            try:
                cfg = update_from_form(cfg, data)  # test what is in the form, without saving
            except (TypeError, ValueError):
                pass
            return self._json(200, test_embeddings(cfg))
        if path == "/api/test-backend":
            cfg = load_config()
            key = data.get("key") or cfg["backend"]["key"]
            return self._json(200, test_backend(data.get("url") or cfg["backend"]["url"], key))
        if path == "/api/self-update":
            if JOB.running:
                return self._json(409, {"error": "Es läuft gerade ein anderer Vorgang."})
            ok, out = self_update()
            event("Formica-Update gestartet" if ok else f"Formica-Update fehlgeschlagen: {out}", "info" if ok else "error")
            return self._json(200 if ok else 500, {"ok": ok, "error": None if ok else out})
        if path in ("/api/apply", "/api/update"):
            cfg = load_config()
            errs = validate(cfg)
            if errs:
                return self._json(400, {"error": "Bitte zuerst die Einstellungen korrigieren.", "errors": errs})
            if not apply_stack(pull=(path == "/api/update")):
                return self._json(409, {"error": "Es läuft bereits ein Vorgang."})
            return self._json(200, {"ok": True})
        m = re.match(r"^/api/solution/([a-z0-9\-]+)/(start|stop|restart)$", path)
        if m:
            sid, action = m.groups()
            cfg = load_config()
            sol = next((x for x in solutions_status(cfg) if x["id"] == sid), None)
            if not sol or not sol["enabled"]:
                return self._json(404, {"error": "Lösung nicht aktiv"})
            names = [x["name"] for x in sol["members"] if not (sid == "chat" and x["name"] == "caddy" and action == "stop")]
            code, out = run(compose_cmd(action, *names), timeout=180)
            event(f"Lösung {sol['label']}: {ACTION_DE[action]}" + ("" if code == 0 else " fehlgeschlagen"),
                  "info" if code == 0 else "error")
            return self._json(200 if code == 0 else 500, {"ok": code == 0, "output": out})
        m = re.match(r"^/api/solution/([a-z0-9\-]+)$", path)
        if m:
            sid = m.group(1)
            cfg = load_config()
            if sid in CATALOG_BY_ID:
                cat = CATALOG_BY_ID[sid]
                cur = dict((cfg.get("solutions") or {}).get(sid) or {})
                if "enabled" in data and not cat.get("core"):
                    cur["enabled"] = bool(data["enabled"])
                if isinstance(data.get("routing"), dict) and cat.get("web"):
                    r = data["routing"]
                    cur["routing"] = {"type": r.get("type") if r.get("type") in ("port", "subdomain", "none") else "port",
                                      "port": int(r.get("port") or 0), "sub": str(r.get("sub") or "").strip()}
                if isinstance(data.get("settings"), dict):
                    st = dict(cur.get("settings") or {})
                    ds = data["settings"]
                    if sid == "whisper" and ds.get("model"):
                        st["model"] = str(ds["model"]).strip()
                    if sid == "files" and ds.get("keep_days") is not None:
                        st["keep_days"] = max(1, min(365, float(ds["keep_days"])))
                    if sid == "images":
                        if ds.get("mode") in ("cpu", "cuda", "external", "backend"):
                            st["mode"] = ds["mode"]
                        if ds.get("model") in IMAGE_MODELS:
                            st["model"] = ds["model"]
                        for k in ("size", "url", "ext_model"):
                            if k in ds:
                                st[k] = str(ds[k]).strip()
                        if ds.get("key"):
                            st["key"] = str(ds["key"]).strip()
                        if st.get("size") and not re.match(r"^\d{3,4}x\d{3,4}$", st["size"]):
                            return self._json(400, {"error": "Bildgröße im Format 512x512"})
                        if st.get("mode") == "external" and not st.get("url"):
                            return self._json(400, {"error": "Für einen externen Dienst bitte die Adresse angeben."})
                    cur["settings"] = st
                cfg.setdefault("solutions", {})[sid] = cur
                label = cat["label"]
            else:
                cs = next((c for c in custom_solutions(cfg) if c["id"] == sid), None)
                if not cs:
                    return self._json(404, {"error": "Unbekannte Lösung"})
                if "enabled" in data:
                    cs["enabled"] = bool(data["enabled"])
                label = cs.get("label") or sid
            errs = validate(cfg)
            route_errs = [e for e in errs if e.startswith(label + ":")]
            if route_errs:
                return self._json(400, {"error": "; ".join(route_errs)})
            save_config(cfg)
            if "enabled" in data:
                event(f"Lösung {label} {'aktiviert' if data['enabled'] else 'deaktiviert'}")
            return self._json(200, {"ok": True, "solutions": solutions_status(cfg)})
        if path == "/api/custom":
            cfg = load_config()
            spec, err = parse_custom(data, custom_solutions(cfg))
            if err:
                return self._json(400, {"error": err})
            lst = [c for c in custom_solutions(cfg) if c["id"] != spec["id"]] + [spec]
            cfg["custom"] = lst
            errs = [e for e in validate(cfg) if e.startswith(spec["label"] + ":") or spec["id"] in e]
            if errs:
                return self._json(400, {"error": "; ".join(errs)})
            save_config(cfg)
            event(f"Eigene Lösung {spec['label']} gespeichert ({spec['image']})")
            return self._json(200, {"ok": True, "solutions": solutions_status(cfg)})
        m = re.match(r"^/api/custom/([a-z0-9\-]+)/delete$", path)
        if m:
            cfg = load_config()
            cfg["custom"] = [c for c in custom_solutions(cfg) if c["id"] != m.group(1)]
            save_config(cfg)
            event(f"Eigene Lösung {m.group(1)} entfernt (Daten-Volumes bleiben erhalten)")
            return self._json(200, {"ok": True, "solutions": solutions_status(cfg)})
        m = re.match(r"^/api/service/([a-z0-9\-]+)/(start|stop|restart)$", path)
        if m:
            name, action = m.groups()
            if name not in all_service_names(load_config()):
                return self._json(404, {"error": "Unbekannter Dienst"})
            code, out = run(compose_cmd(action, name), timeout=120)
            event(f"Dienst {name}: {ACTION_DE[action]}" + ("" if code == 0 else " fehlgeschlagen"),
                  "info" if code == 0 else "error")
            return self._json(200 if code == 0 else 500, {"ok": code == 0, "output": out})
        return self._json(404, {"error": "Nicht gefunden"})


def serve() -> None:
    global MON
    MON = Monitor()
    cfg = load_config()
    compose = STACK / "compose.yaml"
    before = compose.read_text() if compose.exists() else None
    if before is not None or cfg.get("admin"):
        render(cfg)  # a new Formica version may write the files differently
    if before is not None and cfg.get("applied") and compose.read_text() != before:
        log("Neue Formica-Version: Dienste werden mit den neuen Einstellungen gestartet")
        apply_stack()
    threading.Thread(target=check_update, daemon=True).start()
    migrate_control()
    if not cfg.get("admin"):
        log(f"Einrichtung offen. Einrichtungscode: {setup_token()}")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    event(f"Formica {version()} gestartet")
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

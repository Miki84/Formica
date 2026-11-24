"""Formica: Dateien erzeugen.

A small OpenAPI tool server for Open WebUI. The model calls it with the
content (tables, headings, paragraphs) and gets a download link back:
Excel (.xlsx), Word (.docx), PDF and CSV.

Files live in /data/<token>/<name> and are deleted after FILES_KEEP_DAYS.
Caddy serves the downloads under <Formica-URL>/formica-files/.
"""

from __future__ import annotations

import csv
import io
import os
import re
import secrets
import shutil
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

DATA = Path(os.environ.get("FILES_DATA", "/data"))
PUBLIC_URL = os.environ.get("FILES_PUBLIC_URL", "").rstrip("/")  # e.g. https://ki.firma.intern/formica-files
API_KEY = os.environ.get("FILES_API_KEY", "")
KEEP_DAYS = float(os.environ.get("FILES_KEEP_DAYS", "7"))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

app = FastAPI(
    title="Dateien erzeugen",
    version="1.0.0",
    description="Erstellt Excel-, Word-, PDF- und CSV-Dateien aus Tabellen und Text und gibt einen "
                "Download-Link zurück. Den Link der Antwort immer als Markdown-Link an den Benutzer weitergeben.",
)
bearer = HTTPBearer(auto_error=False)


def auth(cred: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> None:
    if API_KEY and (not cred or not secrets.compare_digest(cred.credentials, API_KEY)):
        raise HTTPException(401, "Falscher oder fehlender API-Key")


# --------------------------------------------------------------------------
# Models (what the language model fills in)
# --------------------------------------------------------------------------

Cell = Union[str, float, int, bool, None]


class Sheet(BaseModel):
    name: str = Field("Tabelle1", description="Name des Tabellenblatts (max. 31 Zeichen)")
    columns: list[str] = Field(..., description="Spaltenüberschriften")
    rows: list[list[Cell]] = Field(..., description="Zeilen; jede Zeile hat so viele Werte wie es Spalten gibt")


class ExcelRequest(BaseModel):
    filename: str = Field(..., description="Dateiname ohne Endung, z. B. Angebot_AN0549685")
    sheets: list[Sheet] = Field(..., description="Ein oder mehrere Tabellenblätter")


class Block(BaseModel):
    type: Literal["heading", "paragraph", "bullets", "table"] = Field(
        ..., description="heading = Überschrift, paragraph = Absatz, bullets = Aufzählung, table = Tabelle")
    text: Optional[str] = Field(None, description="Text für heading und paragraph")
    level: int = Field(1, description="Ebene der Überschrift, 1 bis 3")
    items: Optional[list[str]] = Field(None, description="Punkte für bullets")
    columns: Optional[list[str]] = Field(None, description="Spaltenüberschriften für table")
    rows: Optional[list[list[Cell]]] = Field(None, description="Zeilen für table")


class DocumentRequest(BaseModel):
    filename: str = Field(..., description="Dateiname ohne Endung")
    title: Optional[str] = Field(None, description="Titel oben im Dokument")
    blocks: list[Block] = Field(..., description="Inhalt in Reihenfolge")


class CsvRequest(BaseModel):
    filename: str = Field(..., description="Dateiname ohne Endung")
    columns: list[str]
    rows: list[list[Cell]]


class FileResult(BaseModel):
    filename: str
    download_url: str
    markdown: str = Field(..., description="Fertiger Markdown-Link für die Antwort an den Benutzer")
    expires_days: float


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

NUM = re.compile(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$|^-?\d+(,\d+)?$|^-?\d+(\.\d+)?$")


def as_number(v: Cell):
    """'1.234,56' / '11,25' / '519.45' -> number; anything else stays as it is."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float)):
        return v
    s = str(v).strip().replace(" ", "")
    s2 = re.sub(r"\s*(EUR|€|%)$", "", s, flags=re.I).strip()
    if s2 and NUM.match(s2):
        if "," in s2:
            s2 = s2.replace(".", "").replace(",", ".")
        try:
            f = float(s2)
            return int(f) if f.is_integer() and "." not in s2 else f
        except ValueError:
            pass
    return v


def safe_name(name: str, ext: str) -> str:
    name = re.sub(r"[^\w\-. äöüÄÖÜß]", "_", (name or "Datei").strip())[:80].strip(" .") or "Datei"
    name = re.sub(r"\.(xlsx|docx|pdf|csv)$", "", name, flags=re.I)
    return f"{name}.{ext}"


def store(data: bytes, name: str) -> FileResult:
    token = secrets.token_urlsafe(18)
    d = DATA / token
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    url = f"{PUBLIC_URL}/{token}/{name}".replace(" ", "%20")
    return FileResult(filename=name, download_url=url, markdown=f"[{name}]({url})", expires_days=KEEP_DAYS)


def cleanup_loop() -> None:
    while True:
        cutoff = time.time() - KEEP_DAYS * 86400
        for d in DATA.glob("*"):
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass
        time.sleep(3600)


@app.on_event("startup")
def _start() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=cleanup_loop, daemon=True).start()


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@app.post("/excel", operation_id="create_excel", response_model=FileResult, dependencies=[Depends(auth)],
          summary="Excel-Datei (.xlsx) erstellen",
          description="Erstellt eine Excel-Datei mit einem oder mehreren Tabellenblättern. Zahlen als Zahlen "
                      "übergeben (ohne Einheit). Gibt einen Download-Link zurück.")
def create_excel(req: ExcelRequest) -> FileResult:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    used = set()
    for sh in req.sheets:
        title = re.sub(r"[\[\]:*?/\\]", "_", sh.name or "Tabelle")[:31] or "Tabelle"
        while title in used:
            title = (title[:28] + "_" + str(len(used)))[:31]
        used.add(title)
        ws = wb.create_sheet(title)
        ws.append(sh.columns)
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="2F5597")
            c.alignment = Alignment(vertical="center")
        for row in sh.rows:
            ws.append([as_number(v) for v in row])
        for i, col in enumerate(sh.columns, start=1):
            vals = [col] + [r[i - 1] for r in sh.rows if len(r) >= i]
            width = min(60, max(8, max(len(str(v)) for v in vals if v is not None) + 2))
            ws.column_dimensions[get_column_letter(i)].width = width
            for cell in ws.iter_cols(min_col=i, max_col=i, min_row=2):
                for c in cell:
                    if isinstance(c.value, float):
                        c.number_format = "#,##0.00##"
        ws.freeze_panes = "A2"
        if sh.rows:
            ws.auto_filter.ref = ws.dimensions
    buf = io.BytesIO()
    wb.save(buf)
    return store(buf.getvalue(), safe_name(req.filename, "xlsx"))


@app.post("/word", operation_id="create_word", response_model=FileResult, dependencies=[Depends(auth)],
          summary="Word-Dokument (.docx) erstellen",
          description="Erstellt ein Word-Dokument aus Überschriften, Absätzen, Aufzählungen und Tabellen. "
                      "Gibt einen Download-Link zurück.")
def create_word(req: DocumentRequest) -> FileResult:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    if req.title:
        doc.add_heading(req.title, level=0)
    for b in req.blocks:
        if b.type == "heading":
            doc.add_heading(b.text or "", level=max(1, min(3, b.level)))
        elif b.type == "paragraph":
            doc.add_paragraph(b.text or "")
        elif b.type == "bullets":
            for it in b.items or []:
                doc.add_paragraph(it, style="List Bullet")
        elif b.type == "table" and b.columns:
            t = doc.add_table(rows=1, cols=len(b.columns))
            t.style = "Light Grid Accent 1"
            for i, col in enumerate(b.columns):
                t.rows[0].cells[i].text = str(col)
            for r in b.rows or []:
                cells = t.add_row().cells
                for i in range(len(b.columns)):
                    cells[i].text = "" if i >= len(r) or r[i] is None else str(r[i])
            doc.add_paragraph()
    buf = io.BytesIO()
    doc.save(buf)
    return store(buf.getvalue(), safe_name(req.filename, "docx"))


@app.post("/pdf", operation_id="create_pdf", response_model=FileResult, dependencies=[Depends(auth)],
          summary="PDF erstellen",
          description="Erstellt ein PDF aus Überschriften, Absätzen, Aufzählungen und Tabellen. "
                      "Gibt einen Download-Link zurück.")
def create_pdf(req: DocumentRequest) -> FileResult:
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_font("DejaVu", "", FONT)
    pdf.add_font("DejaVu", "B", FONT_BOLD)
    pdf.add_page()
    pdf.set_font("DejaVu", size=10.5)
    w = pdf.epw
    if req.title:
        pdf.set_font("DejaVu", "B", 18)
        pdf.multi_cell(w, 9, req.title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    for b in req.blocks:
        if b.type == "heading":
            size = {1: 15, 2: 13, 3: 11.5}.get(max(1, min(3, b.level)), 13)
            pdf.ln(2)
            pdf.set_font("DejaVu", "B", size)
            pdf.multi_cell(w, size * 0.55, b.text or "", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif b.type == "paragraph":
            pdf.set_font("DejaVu", "", 10.5)
            pdf.multi_cell(w, 5.5, b.text or "", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif b.type == "bullets":
            pdf.set_font("DejaVu", "", 10.5)
            for it in b.items or []:
                pdf.multi_cell(w, 5.5, "•  " + it, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif b.type == "table" and b.columns:
            pdf.set_font("DejaVu", "", 9)
            head = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=(47, 85, 151))
            texts = [[str(c) for c in b.columns]] + [["" if v is None else str(v) for v in r] for r in (b.rows or [])]
            lens = [max(4, min(40, max(len(t[i]) if i < len(t) else 0 for t in texts))) for i in range(len(b.columns))]
            numeric = [all(isinstance(as_number(r[i]), (int, float)) for r in (b.rows or []) if i < len(r) and r[i] not in (None, ""))
                       and any(i < len(r) and r[i] not in (None, "") for r in (b.rows or [])) for i in range(len(b.columns))]
            with pdf.table(headings_style=head, line_height=5.5, col_widths=tuple(lens),
                           text_align=tuple("RIGHT" if n else "LEFT" for n in numeric),
                           borders_layout="SINGLE_TOP_LINE", cell_fill_color=(240, 244, 250),
                           cell_fill_mode="ROWS") as t:
                r0 = t.row()
                for col in b.columns:
                    r0.cell(str(col))
                for r in b.rows or []:
                    row = t.row()
                    for i in range(len(b.columns)):
                        row.cell("" if i >= len(r) or r[i] is None else str(r[i]))
            pdf.ln(3)
    pdf.set_y(-14)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(130)
    pdf.cell(0, 5, f"Erstellt am {date.today().strftime('%d.%m.%Y')}", align="R")
    return store(bytes(pdf.output()), safe_name(req.filename, "pdf"))


@app.post("/csv", operation_id="create_csv", response_model=FileResult, dependencies=[Depends(auth)],
          summary="CSV-Datei erstellen",
          description="Erstellt eine CSV-Datei mit Semikolon als Trenner (öffnet direkt im deutschen Excel). "
                      "Gibt einen Download-Link zurück.")
def create_csv(req: CsvRequest) -> FileResult:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    w.writerow(req.columns)

    def fmt(v):
        v = as_number(v)
        if isinstance(v, float):
            return (f"{v:.2f}" if round(v, 2) == v else f"{v:.10g}").replace(".", ",")
        return "" if v is None else v
    for r in req.rows:
        w.writerow([fmt(v) for v in r])
    return store(("﻿" + buf.getvalue()).encode("utf-8"), safe_name(req.filename, "csv"))


# --------------------------------------------------------------------------
# Downloads (Caddy: /formica-files/<token>/<name> -> here) and health
# --------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/{token}/{name}", include_in_schema=False)
def download(token: str, name: str, request: Request):
    if not re.match(r"^[\w-]{10,}$", token) or "/" in name or name.startswith("."):
        raise HTTPException(404)
    f = DATA / token / name
    if not f.is_file():
        raise HTTPException(404, "Datei nicht gefunden oder abgelaufen")
    return FileResponse(f, filename=name)

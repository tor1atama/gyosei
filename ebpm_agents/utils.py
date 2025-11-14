# ebpm_agents/utils.py
from __future__ import annotations
import re, json, unicodedata
from typing import Any, List, Tuple, Dict, Optional
from pathlib import Path
import fitz  # PyMuPDF

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")

def ensure_dir(d: str | Path) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)

def safe_json_loads(s: str) -> Any:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m1, m2 = s.find("{"), s.rfind("}")
    if m1 != -1 and m2 != -1 and m2 > m1:
        cand = s[m1:m2+1]
        try:
            return json.loads(cand)
        except Exception:
            cand = re.sub(r",\s*([}\]])", r"\1", cand)
            return json.loads(cand)
    raise ValueError("JSONとして解釈できませんでした。")

# ---------- PDF I/O ----------
def extract_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            t = page.get_text("text") or ""
            pages.append((i, t.strip()))
    return pages

def join_with_markers(pages: List[Tuple[int, str]]) -> str:
    return "\n\n".join([f"[Page {p}]\n{t}" for p, t in pages])

def chunk_by_chars(full_text: str, max_chars: int = 8000) -> List[Tuple[str, str]]:
    """Return [("chunk1-3", text), ...] using [Page i] markers."""
    parts = full_text.split("\n\n[Page ")
    recon = []
    if parts:
        if parts[0].startswith("[Page "):
            recon.append(parts[0])
        else:
            recon.append(parts[0])
        for tail in parts[1:]:
            recon.append("[Page " + tail)
    else:
        recon = [full_text]
    chunks, cur, cur_len, start = [], [], 0, 1
    for i, seg in enumerate(recon, start=1):
        if cur_len + len(seg) > max_chars and cur:
            chunks.append((f"chunk{start}-{i-1}", "\n\n".join(cur).strip()))
            cur, cur_len, start = [seg], len(seg), i
        else:
            cur.append(seg); cur_len += len(seg)
    if cur:
        chunks.append((f"chunk{start}-{start+len(cur)-1}", "\n\n".join(cur).strip()))
    return chunks

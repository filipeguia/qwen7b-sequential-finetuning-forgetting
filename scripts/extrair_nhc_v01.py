#!/usr/bin/env python
"""Extrai o texto bruto dos PDFs NHC TCR 2025 e gera o manifesto JSONL.

Rodar com:  C:\\Python314\\python.exe -P extrair_nhc_v01.py --pdf-root ... --txt-root ... \
            --manifest ... --download-log ...
com o diretorio de trabalho em scripts (nunca na pasta de download).

Regras:
- Texto exatamente como o PyMuPDF entrega (page.get_text("text")), sem limpeza.
- Antes de cada pagina entra a linha "=== page N ===" (N a partir de 1).
- Os PDFs sao dados nao confiaveis: so lidos, nunca executados; links e anotacoes nao sao abertos.
- chars/words/approx_tokens sao calculados sobre o texto extraido SEM os marcadores "=== page N ===".
  chars = soma de len(texto de cada pagina); words = len(texto.split()) por pagina, somados.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import fitz  # PyMuPDF

MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
)
DATE_RE = re.compile(r"\b" + MONTHS + r"\b\.?", re.IGNORECASE)
# ano 2025/2026; sem \b final porque o PDF cola o numero da nota de rodape ("20261")
YEAR_RE = re.compile(r"(?<!\d)(?:2025|2026)")
# linha que e SO uma data simples ("30 January 20261"; o "1" final e o numero da nota de rodape
# colado pelo PDF). Intervalos do evento ("11-22 August 2025") nao casam. Aceita U+200B/U+FEFF.
PURE_DATE_RE = re.compile(
    r"^[\s​﻿]*\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+(?:2025|2026)\d?[\s​﻿]*$",
    re.IGNORECASE,
)
UPD_KEYS = ("updated", "preliminary", "revised")
TP_KEYS = ("\u00a9", "copyright", "courtesy", "credit", "image", "photo")
STRONG_KEYS = ("\u00a9", "copyright", "courtesy", "credit", "photo")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_download_log(path):
    info = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ok"):
                info[rec["file"]] = rec
    return info


def split_lines(text):
    return text.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-root", required=True)
    ap.add_argument("--txt-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--download-log", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.manifest) and not args.overwrite:
        print("ERRO: manifesto ja existe (use --overwrite para recriar o MEU arquivo):", args.manifest)
        return 2

    dl = load_download_log(args.download_log)
    records = []
    summary = []

    for basin in ("atl", "epac"):
        pdf_dir = os.path.join(args.pdf_root, basin)
        txt_dir = os.path.join(args.txt_root, basin)
        os.makedirs(txt_dir, exist_ok=True)
        names = sorted(n for n in os.listdir(pdf_dir) if n.lower().endswith(".pdf"))
        for fn in names:
            path = os.path.join(pdf_dir, fn)
            stem = fn[:-4]
            event_id, _, name = stem.partition("_")
            doc = fitz.open(path)
            pages_text = []
            for i in range(doc.page_count):
                pages_text.append(doc[i].get_text("text"))
            n_pages = doc.page_count
            doc.close()

            # arquivo de texto: marcador + texto cru de cada pagina
            out_parts = []
            for i, t in enumerate(pages_text, start=1):
                out_parts.append("=== page %d ===\n" % i)
                out_parts.append(t)
                if t and not t.endswith("\n"):
                    out_parts.append("\n")
            out_text = "".join(out_parts)
            txt_path = os.path.join(txt_dir, event_id + ".txt")
            try:
                data = out_text.encode("utf-8")
            except UnicodeEncodeError:
                data = out_text.encode("utf-8", errors="replace")
                print("AVISO: caracteres nao codificaveis substituidos em", fn)
            with open(txt_path, "wb") as f:
                f.write(data)

            chars = sum(len(t) for t in pages_text)
            words = sum(len(t.split()) for t in pages_text)
            approx_tokens = int(round(words * 1.35))

            all_lines = []  # (page_no, line)
            for pno, t in enumerate(pages_text, start=1):
                for ln in split_lines(t):
                    all_lines.append((pno, ln))

            p1 = split_lines(pages_text[0]) if pages_text else []
            title_lines = [ln for ln in p1 if ln.strip()][:6]

            # candidatas: linhas das paginas 1 e 2 com nome de mes e 2025/2026 (vao no diagnostico).
            # Escolha de report_date_line: a ULTIMA linha da pagina 1 que e so uma data simples
            # (no bloco de autores a data do relatorio vem depois do periodo do evento, que e
            # um intervalo ou fica antes); se nao houver, a primeira candidata das paginas 1 e 2.
            date_cands = []
            for pno in (1, 2):
                if pno > n_pages:
                    break
                for ln in split_lines(pages_text[pno - 1]):
                    if DATE_RE.search(ln) and YEAR_RE.search(ln):
                        date_cands.append(ln)
            pure = [ln for ln in p1 if PURE_DATE_RE.match(ln)]
            if pure:
                report_date_line = pure[-1]
            else:
                report_date_line = date_cands[0] if date_cands else None

            upd = [ln for _, ln in all_lines if any(k in ln.lower() for k in UPD_KEYS)]
            tp = [ln for _, ln in all_lines if any(k in ln.lower() for k in TP_KEYS)]
            tp_strong = [(p, ln) for p, ln in all_lines if any(k in ln.lower() for k in STRONG_KEYS)]
            table_n = sum(1 for _, ln in all_lines if ln.lstrip().startswith("Table"))
            fig_n = sum(1 for _, ln in all_lines if ln.lstrip().startswith("Figure"))

            d = dl.get(fn, {})
            rec = {
                "event_id": event_id,
                "name": name,
                "basin": basin,
                "file": fn,
                "source_url": d.get("source_url"),
                "download_utc": d.get("download_utc"),
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
                "pages": n_pages,
                "chars": chars,
                "words": words,
                "approx_tokens": approx_tokens,
                "title_lines": title_lines,
                "report_date_line": report_date_line,
                "updated_or_preliminary": upd[:5],
                "third_party_notes": tp[:15],
                "table_count_guess": table_n,
                "figure_count_guess": fig_n,
            }
            records.append(rec)

            per_page = [len(t) for t in pages_text]
            allt = "".join(pages_text)
            bad = sum(1 for c in allt if c == "\ufffd" or (0xE000 <= ord(c) <= 0xF8FF) or (ord(c) < 32 and c not in "\n\t\r"))
            alpha = sum(1 for c in allt if c.isalpha())
            summary.append({
                "event_id": event_id, "basin": basin, "pages": n_pages, "chars": chars, "words": words,
                "chars_per_page": round(chars / max(n_pages, 1), 1),
                "min_page_chars": min(per_page) if per_page else 0,
                "pages_lt_200": sum(1 for x in per_page if x < 200),
                "bad_chars": bad,
                "alpha_ratio": round(alpha / max(chars, 1), 3),
                "tp_total_matches": len(tp),
                "tp_strong": tp_strong,
                "upd_total_matches": len(upd),
                "date_cands": date_cands,
            })

    records.sort(key=lambda r: (0 if r["basin"] == "atl" else 1, r["event_id"]))
    with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("manifesto:", args.manifest, "registros:", len(records))
    print("--- resumo por relatorio (diagnostico, nao vai no manifesto)")
    for s in summary:
        print(json.dumps({k: v for k, v in s.items() if k not in ("tp_strong", "date_cands")}, ensure_ascii=False))
    print("--- candidatas de report_date_line (paginas 1 e 2)")
    for s in summary:
        print(json.dumps({"event_id": s["event_id"], "cands": s["date_cands"]}, ensure_ascii=False))
    print("--- linhas com termos fortes (c, copyright, courtesy, credit, photo), sem limite")
    for s in summary:
        for p, ln in s["tp_strong"]:
            print(json.dumps({"event_id": s["event_id"], "page": p, "line": ln}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

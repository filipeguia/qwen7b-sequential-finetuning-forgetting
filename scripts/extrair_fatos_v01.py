# Extrai fatos candidatos dos relatorios limpos com o proprio Qwen2.5-7B-Instruct (4 bits, decodificacao gulosa).
# O Qwen escreve pergunta, resposta e evidencia; o script so filtra: evidencia literal no trecho, resposta literal na
# evidencia, nome do ciclone na pergunta, resposta fora da pergunta. Nenhuma pergunta ou resposta vem de outro modelo
# Uso: python extrair_fatos_v01.py <pasta_limpo> <saida.jsonl> [lote]
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
ENTRADA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
LOTE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAX_PALAVRAS = 220
TIPOS = {"number", "date", "time", "place", "name", "phrase"}

INSTRUCAO = """Below is a passage from the National Hurricane Center report on {cabecalho}.

Extract up to 6 facts from the passage that are specific to this storm. For each fact give:
- "question": a question that names the storm and the year 2025 and can be answered from the passage alone
- "answer": the shortest exact answer, copied word for word from the passage (a number with its unit, a date, a time, a place, a name or a short phrase)
- "evidence": the full sentence from the passage that contains the answer, copied word for word
- "type": one of number, date, time, place, name, phrase

Rules: do not ask about figures, tables or sections of the report; no yes/no questions; the answer must not appear in the question; prefer facts that belong only to this storm.

Return only a JSON list of objects.

Passage:
{trecho}"""


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


def trechos(caminho):
    linhas = caminho.read_text(encoding="utf-8").split("\n\n")
    cabecalho, pars = linhas[0].strip(), [p.strip() for p in linhas[1:] if p.strip()]
    secao, atual, n = "", [], 0
    saida = []
    for p in pars:
        if p.startswith("## "):
            if atual:
                saida.append((secao, " ".join(atual)))
                atual, n = [], 0
            secao = p[3:]
            continue
        frases = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"(])", p)
        for f in frases:
            k = len(f.split())
            if atual and n + k > MAX_PALAVRAS:
                saida.append((secao, " ".join(atual)))
                atual, n = [], 0
            atual.append(f)
            n += k
        if n > MAX_PALAVRAS * 0.6:
            saida.append((secao, " ".join(atual)))
            atual, n = [], 0
    if atual:
        saida.append((secao, " ".join(atual)))
    return cabecalho, saida


def nome_do_ciclone(cabecalho):
    m = re.match(r"(?:Hurricane|Tropical Storm|Subtropical Storm|Tropical Depression)\s+(\w+)", cabecalho)
    return m.group(1) if m else cabecalho.split()[0]


def ler_json(texto):
    m = re.search(r"\[.*\]", texto, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def filtrar(item, trecho, nome):
    if not isinstance(item, dict):
        return None, "nao_objeto"
    q, a, e, t = (norm(str(item.get(k, ""))) for k in ("question", "answer", "evidence", "type"))
    if not (q and a and e):
        return None, "campo_vazio"
    tn = norm(trecho)
    if e not in tn:
        return None, "evidencia_nao_literal"
    if a.lower() not in e.lower():
        return None, "resposta_fora_da_evidencia"
    if nome.lower() not in q.lower():
        return None, "pergunta_sem_nome"
    if a.lower() in q.lower():
        return None, "resposta_na_pergunta"
    if len(a.split()) > 12:
        return None, "resposta_longa"
    if re.search(r"\b(figure|fig\.|table|section|report)\b", q, flags=re.I):
        return None, "pergunta_sobre_relatorio"
    return {"question": q, "answer": a, "evidence": e, "type": t.lower() if t.lower() in TIPOS else "other"}, "ok"


tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
tok.padding_side = "left"
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()

tarefas = []
for bacia in ("atl", "epac"):
    for arq in sorted((ENTRADA / bacia).glob("*.txt")):
        cab, lista = trechos(arq)
        for i, (secao, trecho) in enumerate(lista):
            tarefas.append({"bacia": bacia, "evento": arq.stem, "cabecalho": cab, "nome": nome_do_ciclone(cab),
                            "trecho_id": f"{arq.stem}_{i:03d}", "secao": secao, "trecho": trecho})
print("trechos", len(tarefas), flush=True)

SAIDA.parent.mkdir(parents=True, exist_ok=True)
feitos = set()
if SAIDA.exists():
    feitos = {json.loads(l)["trecho_id"] for l in SAIDA.read_text(encoding="utf-8").splitlines() if l.strip()}
pendentes = [t for t in tarefas if t["trecho_id"] not in feitos]
contagem, t0 = {}, time.time()
with SAIDA.open("a", encoding="utf-8") as fout:
    for k in range(0, len(pendentes), LOTE):
        lote = pendentes[k:k + LOTE]
        prompts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCAO.format(cabecalho=t["cabecalho"], trecho=t["trecho"])}],
                                           tokenize=False, add_generation_prompt=True) for t in lote]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=700, do_sample=False, use_cache=True)
        for t, seq in zip(lote, out):
            bruto = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            itens = ler_json(bruto)
            fatos, motivos = [], []
            for it in itens or []:
                f, motivo = filtrar(it, t["trecho"], t["nome"])
                motivos.append(motivo)
                contagem[motivo] = contagem.get(motivo, 0) + 1
                if f:
                    fatos.append(f)
            if itens is None:
                contagem["json_invalido"] = contagem.get("json_invalido", 0) + 1
            fout.write(json.dumps({**{c: t[c] for c in ("bacia", "evento", "cabecalho", "trecho_id", "secao", "trecho")},
                                   "saida_bruta": bruto, "motivos": motivos, "fatos": fatos}, ensure_ascii=False) + "\n")
        fout.flush()
        feitos_agora = k + len(lote)
        print(f"{feitos_agora}/{len(pendentes)} trechos, {round(time.time() - t0)} s, {contagem}", flush=True)
print("OK", SAIDA)

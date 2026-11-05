# Prova de fatos: pergunta da familia de prova (principal e reformulacao) sem fonte no prompt, decodificacao gulosa,
# correcao deterministica (corretor_v02). Avalia o modelo base ou base + adaptador. Grava as respostas brutas
# Uso: python avaliar_v01.py --fatos fatos_selecionados.jsonl --saida resultados/M0 [--adaptador runs/M1] [--splits dev,prova]
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from corretor_v03 import corrigir

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"

ap = argparse.ArgumentParser()
ap.add_argument("--fatos", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--adaptador", default=None)
ap.add_argument("--splits", default="dev,prova")
ap.add_argument("--lote", type=int, default=32)
ap.add_argument("--max_novos", type=int, default=48)
a = ap.parse_args()
splits = set(a.splits.split(","))
saida = Path(a.saida)
saida.mkdir(parents=True, exist_ok=True)

fatos = [json.loads(l) for l in Path(a.fatos).read_text(encoding="utf-8").splitlines() if l.strip()]
fatos = [f for f in fatos if f.get("split") in splits]
itens = [(f, i, q) for f in fatos for i, q in enumerate(f["prova"]["perguntas"][:2])]

tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
tok.padding_side = "left"
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
if a.adaptador:
    model = PeftModel.from_pretrained(model, a.adaptador)
model.eval()

t0 = time.time()
linhas = []
for k in range(0, len(itens), a.lote):
    lote = itens[k:k + a.lote]
    textos = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for _, _, q in lote]
    enc = tok(textos, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_novos, do_sample=False, use_cache=True)
    for (f, i, q), seq in zip(lote, out):
        resp = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        linhas.append({"fact_id": f["fact_id"], "bacia": f["bacia"], "evento": f["evento"], "split": f["split"], "tipo": f["type"],
                       "variante": "principal" if i == 0 else "reformulacao", "pergunta": q, "canonica": f["answer"],
                       "resposta": resp, "acertou": corrigir(resp, f["answer"], f["type"])})
    print(f"{k + len(lote)}/{len(itens)}", flush=True)

(saida / "respostas.jsonl").write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in linhas) + "\n", encoding="utf-8")
resumo = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for l in linhas:
    for chave in (f"{l['bacia']}/{l['split']}/{l['variante']}", f"{l['bacia']}/{l['split']}/{l['variante']}/{l['tipo']}"):
        resumo[chave]["n"][0] += 1
        resumo[chave]["n"][1] += l["acertou"]
final = {k: {"n": v["n"][0], "acertos": v["n"][1], "pct": round(100 * v["n"][1] / v["n"][0], 1)} for k, v in sorted(resumo.items())}
final["_meta"] = {"adaptador": a.adaptador, "segundos": round(time.time() - t0, 1), "itens": len(linhas)}
(saida / "resumo.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in final.items():
    if k.count("/") == 2:
        print(k, v)
print("OK", saida)

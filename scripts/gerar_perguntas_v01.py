# Gera com o Qwen, para cada fato verificado:
# - familia de treino: a pergunta original da extracao + 1 reformulacao + 1 frase de resposta com a resposta exata
# - familia de prova: 2 perguntas novas, pedidas em outra chamada, que precisam diferir das de treino
# O script so filtra (nome do ciclone presente, resposta ausente da pergunta, distancia entre as familias)
# Uso: python gerar_perguntas_v01.py <fatos_verificados.jsonl> <saida.jsonl> [lote]
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from corretor_v03 import conteudo, corrigir, normalizar, numeros_iguais

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
ENTRADA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
LOTE = int(sys.argv[3]) if len(sys.argv) > 3 else 24
DIST_MAX = 0.7

TREINO = """Fact from the National Hurricane Center report on {cabecalho}:
Evidence: {evidence}
Question: {question}
Answer: {answer}

Write one more question that asks for the same answer with different wording. It must name the storm and the year 2025 and must not contain the answer. Then write one short sentence that answers the question and contains exactly "{answer}".
Reply with JSON only: {{"question": "...", "answer_sentence": "..."}}"""

PROVA = """Fact from the National Hurricane Center report on {cabecalho}:
Evidence: {evidence}
Answer: {answer}

These questions are already used:
1. {q0}
2. {q1}

Write two new questions whose answer is "{answer}". Use different words and a different sentence structure from the questions above. Each must name the storm and the year 2025 and must not contain the answer.
Reply with JSON only: {{"questions": ["...", "..."]}}"""


def ler(bruto):
    m = re.search(r"\{.*\}", bruto, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def jaccard(a, b):
    ca, cb = conteudo(normalizar(a)), conteudo(normalizar(b))
    return len(ca & cb) / max(1, len(ca | cb))


def pergunta_ok(q, f):
    return bool(q) and f["nome"].lower() in q.lower() and normalizar(f["answer"]) not in normalizar(q) and q.strip().endswith("?")


fatos = [json.loads(l) for l in ENTRADA.read_text(encoding="utf-8").splitlines() if l.strip()]
# regra da conferencia afrouxada depois de ver a saida: vale a resposta do Qwen (lendo so a evidencia) bater com a
# extraida pelo corretor v03 ou ter os mesmos numeros; a marca "clara" do Qwen e registrada mas nao decide
for f in fatos:
    resp = f["verificacao"]["resposta_qwen"]
    f["verificado_estrito"] = f["verificado"]
    f["verificado"] = bool(resp) and (corrigir(resp, f["answer"], f["type"]) or corrigir(f["answer"], resp, f["type"])
                                      or numeros_iguais(resp, f["answer"]))
fatos = [f for f in fatos if f.get("verificado")]
print("verificados pela regra v03:", len(fatos), flush=True)
cab = {}
for bacia in ("atl", "epac"):
    for arq in (Path(__file__).parent.parent / "dados" / "limpo" / "nhc_tcr_2025_v01" / bacia).glob("*.txt"):
        cab[arq.stem] = arq.read_text(encoding="utf-8").split("\n\n")[0]

tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
tok.padding_side = "left"
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()


def gerar(prompts, max_novos):
    textos = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(textos, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_novos, do_sample=False, use_cache=True)
    return [tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True) for s in out]


# etapa 1: reformulacao de treino e frase de resposta
for k in range(0, len(fatos), LOTE):
    lote = fatos[k:k + LOTE]
    brutos = gerar([TREINO.format(cabecalho=cab[f["evento"]], **{c: f[c] for c in ("evidence", "question", "answer")}) for f in lote], 120)
    for f, b in zip(lote, brutos):
        v = ler(b)
        q1, frase = str(v.get("question", "")).strip(), str(v.get("answer_sentence", "")).strip()
        f["treino"] = {
            "perguntas": [f["question"]] + ([q1] if pergunta_ok(q1, f) and jaccard(q1, f["question"]) < 0.95 else []),
            "frase_resposta": frase if normalizar(f["answer"]) in normalizar(frase) else None,
            "bruto": b[:300],
        }
    print(f"treino {k + len(lote)}/{len(fatos)}", flush=True)

# etapa 2: familia de prova
for k in range(0, len(fatos), LOTE):
    lote = fatos[k:k + LOTE]
    prompts = []
    for f in lote:
        qs = f["treino"]["perguntas"]
        prompts.append(PROVA.format(cabecalho=cab[f["evento"]], evidence=f["evidence"], answer=f["answer"], q0=qs[0], q1=qs[-1]))
    brutos = gerar(prompts, 120)
    for f, b in zip(lote, brutos):
        qs = [str(q).strip() for q in ler(b).get("questions", []) if isinstance(q, str)]
        boas = [q for q in qs if pergunta_ok(q, f) and all(jaccard(q, t) < DIST_MAX for t in f["treino"]["perguntas"])]
        f["prova"] = {"perguntas": boas[:2], "bruto": b[:300]}
        f["prova_ok"] = len(boas) >= 2 and f["treino"]["frase_resposta"] is not None
    print(f"prova {k + len(lote)}/{len(fatos)}", flush=True)

SAIDA.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in fatos) + "\n", encoding="utf-8")
print("OK", SAIDA, sum(f["prova_ok"] for f in fatos), "com familia de prova completa de", len(fatos))

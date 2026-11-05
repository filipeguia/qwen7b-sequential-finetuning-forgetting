# Conhecimento geral pelo lm_eval, com o proprio modelo em 4 bits (base ou base + adaptador):
# MMLU 5 exemplos (5 questoes por materia = 285), GSM8K 5 exemplos (100 questoes, strict-match) e IFEval (50 prompts,
# com chat template). Subconjuntos fixos pela semente; detectam queda grande, nao certificam diferenca de 1 ponto
# Uso: python geral_v01.py --saida resultados/M0_v01 [--adaptador runs/M1] [--tarefas mmlu,gsm8k,ifeval]
import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import lm_eval
from lm_eval.models.huggingface import HFLM

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
SEMENTE = 1234
CONFIG = {
    "mmlu": {"num_fewshot": 5, "limit": 5, "lote": 1, "chat": False},
    "gsm8k": {"num_fewshot": 5, "limit": 100, "lote": 4, "chat": False},
    "ifeval": {"num_fewshot": 0, "limit": 50, "lote": 8, "chat": True},
}

ap = argparse.ArgumentParser()
ap.add_argument("--saida", required=True)
ap.add_argument("--adaptador", default=None)
ap.add_argument("--tarefas", default="mmlu,gsm8k,ifeval")
a = ap.parse_args()
saida = Path(a.saida)
saida.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
if a.adaptador:
    model = PeftModel.from_pretrained(model, a.adaptador)
model.eval()

arq = saida / "geral.json"
res = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}
for tarefa in a.tarefas.split(","):
    c = CONFIG[tarefa]
    torch.cuda.empty_cache()
    t = time.time()
    hf = HFLM(pretrained=model, tokenizer=tok, batch_size=c["lote"], max_length=2048)
    r = lm_eval.simple_evaluate(model=hf, tasks=[tarefa], num_fewshot=c["num_fewshot"], limit=c["limit"],
                                apply_chat_template=c["chat"], random_seed=SEMENTE, numpy_random_seed=SEMENTE,
                                torch_random_seed=SEMENTE, fewshot_random_seed=SEMENTE, log_samples=False)
    metricas = {k: v for k, v in r["results"].get(tarefa, {}).items() if isinstance(v, (int, float))}
    res[tarefa] = {"metricas": metricas, "n": r.get("n-samples", {}).get(tarefa) or sum(v.get("effective", 0) for v in r.get("n-samples", {}).values()),
                   "segundos": round(time.time() - t, 1), "config": c, "adaptador": a.adaptador}
    arq.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(tarefa, res[tarefa]["metricas"], res[tarefa]["segundos"], "s", flush=True)
print("OK", arq)

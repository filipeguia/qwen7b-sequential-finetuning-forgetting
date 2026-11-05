# Segunda passada: o proprio Qwen responde cada pergunta lendo so a evidencia e diz se a resposta e unica e clara.
# Fica o fato em que a resposta do Qwen bate com a resposta extraida (nos dois sentidos) e que ele marcou como clara
# Uso: python verificar_fatos_v01.py <fatos_filtrados.jsonl> <saida.jsonl> [lote]
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from corretor_v02 import corrigir

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
ENTRADA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
LOTE = int(sys.argv[3]) if len(sys.argv) > 3 else 24

INSTRUCAO = """Evidence: {evidence}

Question: {question}

Using only the evidence, answer the question in a few words. Then say whether the question has exactly one clear answer in the evidence.
Reply with JSON only: {{"answer": "...", "clear": true or false}}"""

fatos = [json.loads(l) for l in ENTRADA.read_text(encoding="utf-8").splitlines() if l.strip()]
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
tok.padding_side = "left"
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()

aceitos = 0
with SAIDA.open("w", encoding="utf-8") as fout:
    for k in range(0, len(fatos), LOTE):
        lote = fatos[k:k + LOTE]
        prompts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCAO.format(evidence=f["evidence"], question=f["question"])}],
                                           tokenize=False, add_generation_prompt=True) for f in lote]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=60, do_sample=False, use_cache=True)
        for f, seq in zip(lote, out):
            bruto = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\{.*\}", bruto, flags=re.S)
            try:
                v = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                v = {}
            resp = str(v.get("answer", ""))
            claro = v.get("clear") is True
            bate = bool(resp) and (corrigir(resp, f["answer"], f["type"]) or corrigir(f["answer"], resp, f["type"]))
            f["verificacao"] = {"resposta_qwen": resp, "clara": claro, "bate": bate, "bruto": bruto[:300]}
            f["verificado"] = claro and bate
            aceitos += f["verificado"]
            fout.write(json.dumps(f, ensure_ascii=False) + "\n")
        print(f"{k + len(lote)}/{len(fatos)} verificados, {aceitos} aceitos", flush=True)
print("OK", SAIDA, aceitos, "de", len(fatos))

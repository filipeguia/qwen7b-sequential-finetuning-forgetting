# Treino QLoRA de uma etapa (um livro). A etapa B continua o MESMO adaptador da etapa A (um checkpoint, um caminho
# de inferencia), com otimizador novo. Replay opcional: mistura exemplos de um buffer congelado do livro anterior
# na proporcao de tokens novos:antigos pedida. Perda so nos tokens da resposta
# Uso: python treinar_v01.py --dados treino_B.jsonl --saida runs/x [--continuar runs/M1] [--replay buffer_A.jsonl --novos_por_antigo 4]
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import bitsandbytes as bnb
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
ALVOS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

ap = argparse.ArgumentParser()
ap.add_argument("--dados", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--continuar", default=None, help="adaptador da etapa anterior a continuar treinando")
ap.add_argument("--replay", default=None, help="buffer congelado de exemplos do livro anterior")
ap.add_argument("--novos_por_antigo", type=float, default=4.0)
ap.add_argument("--semente", type=int, default=17)
ap.add_argument("--epocas", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--r", type=int, default=8)
ap.add_argument("--alpha", type=int, default=16)
ap.add_argument("--max_tokens", type=int, default=128)
ap.add_argument("--por_microlote", type=int, default=3)
ap.add_argument("--acumular", type=int, default=8)
ap.add_argument("--aquecimento", type=float, default=0.05)
a = ap.parse_args()

random.seed(a.semente)
np.random.seed(a.semente)
torch.manual_seed(a.semente)
saida = Path(a.saida)
saida.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
tok.padding_side = "right"


def codificar(ex):
    # ex = {"pergunta": ..., "resposta": ...}; rotulo -100 em tudo que nao e a resposta do assistente
    prompt = tok.apply_chat_template([{"role": "user", "content": ex["pergunta"]}], tokenize=False, add_generation_prompt=True)
    completo = tok.apply_chat_template([{"role": "user", "content": ex["pergunta"]}, {"role": "assistant", "content": ex["resposta"]}], tokenize=False)
    ids_p = tok(prompt, add_special_tokens=False).input_ids
    ids = tok(completo, add_special_tokens=False).input_ids[: a.max_tokens]
    rot = [-100] * min(len(ids_p), len(ids)) + ids[len(ids_p):]
    return {"ids": ids, "rotulos": rot[: len(ids)], "origem": ex.get("origem", "novo")}


novos = [codificar(json.loads(l)) for l in Path(a.dados).read_text(encoding="utf-8").splitlines() if l.strip()]
buffer = [codificar({**json.loads(l), "origem": "replay"}) for l in Path(a.replay).read_text(encoding="utf-8").splitlines() if l.strip()] if a.replay else []
tokens_novos = sum(sum(r != -100 for r in e["rotulos"]) for e in novos)


def epoca(rng):
    ex = novos[:]
    if buffer:
        # sorteia do buffer ate ter 1 token antigo para cada N novos (contando tokens de resposta)
        alvo = tokens_novos / a.novos_por_antigo
        soma, extra = 0, []
        while soma < alvo:
            e = rng.choice(buffer)
            extra.append(e)
            soma += sum(r != -100 for r in e["rotulos"])
        ex += extra
    rng.shuffle(ex)
    return [ex[i:i + a.por_microlote] for i in range(0, len(ex), a.por_microlote)]


def lote_tensor(itens):
    n = max(len(e["ids"]) for e in itens)
    ids = torch.full((len(itens), n), tok.pad_token_id, dtype=torch.long)
    rot = torch.full((len(itens), n), -100, dtype=torch.long)
    mask = torch.zeros((len(itens), n), dtype=torch.long)
    for i, e in enumerate(itens):
        ids[i, : len(e["ids"])] = torch.tensor(e["ids"])
        rot[i, : len(e["rotulos"])] = torch.tensor(e["rotulos"])
        mask[i, : len(e["ids"])] = 1
    return ids.cuda(), rot.cuda(), mask.cuda()


quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
if a.continuar:
    model = PeftModel.from_pretrained(model, a.continuar, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=ALVOS, task_type="CAUSAL_LM"))
model.train()
params = [p for p in model.parameters() if p.requires_grad]
opt = bnb.optim.PagedAdamW8bit(params, lr=a.lr, weight_decay=0.0)
rng = random.Random(a.semente)
microlotes_por_epoca = math.ceil((len(novos) + (len(novos) / a.novos_por_antigo if buffer else 0)) / a.por_microlote)
passos_total = math.ceil(microlotes_por_epoca * a.epocas / a.acumular)
sched = get_linear_schedule_with_warmup(opt, max(1, int(passos_total * a.aquecimento)), passos_total)

registro = {"args": vars(a), "exemplos_novos": len(novos), "buffer": len(buffer), "tokens_resposta_novos": tokens_novos, "epocas": []}
t0, passo, vistos = time.time(), 0, {"novo": 0, "replay": 0}
for ep in range(a.epocas):
    perdas, mls = [], epoca(rng)
    for i, itens in enumerate(mls):
        for e in itens:
            vistos[e["origem"]] += 1
        ids, rot, mask = lote_tensor(itens)
        perda = model(input_ids=ids, attention_mask=mask, labels=rot).loss
        (perda / a.acumular).backward()
        perdas.append(perda.item())
        if (i + 1) % a.acumular == 0 or i == len(mls) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            passo += 1
    registro["epocas"].append({"epoca": ep + 1, "perda_media": round(sum(perdas) / len(perdas), 4), "microlotes": len(mls),
                               "segundos": round(time.time() - t0, 1)})
    print(registro["epocas"][-1], flush=True)
registro.update({"passos_otimizador": passo, "exemplos_vistos": vistos, "segundos_total": round(time.time() - t0, 1),
                 "pico_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
model.save_pretrained(saida)
(saida / "registro_treino.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")
print("OK", saida, registro["segundos_total"], "s")

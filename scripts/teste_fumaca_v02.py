# Teste de fumaca v02: mede o que o plano precisa (s, g em lote, prompts longos, MMLU e GSM8K por questao,
# comportamento termico em treino longo). Corrige a v01: sementes fixas, sem copia do LoRA na GPU,
# resultado salvo a cada etapa. Nao ensina fatos (a v01 ja fez isso)
# Uso: python teste_fumaca_v02.py <saida.json>
import json
import random
import subprocess
import sys
import time

import bitsandbytes as bnb
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
SAIDA = sys.argv[1]
SEMENTE = 17

random.seed(SEMENTE)
np.random.seed(SEMENTE)
torch.manual_seed(SEMENTE)

res = {"semente": SEMENTE}


def salvar():
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def gb(x):
    return round(x / 2**30, 2)


def memoria():
    livre, _ = torch.cuda.mem_get_info()
    return {"pico_gb": gb(torch.cuda.max_memory_allocated()), "reservado_gb": gb(torch.cuda.memory_reserved()), "livre_placa_gb": gb(livre)}


def placa():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        temp, clock, watts = [v.strip() for v in out.split(",")]
        return {"temp_c": temp, "clock_mhz": clock, "watts": watts}
    except Exception as e:
        return {"erro": str(e)}


tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
quant = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    REPO, revision=REV, quantization_config=quant, device_map={"": 0},
    dtype=torch.bfloat16, attn_implementation="sdpa",
)
model.eval()
res["placa_inicio"] = placa()
print("carregado", memoria(), flush=True)

# 1. Geracao em lote (prompt curto, 32 tokens novos), como nos paineis de fatos
perguntas = [f"In one sentence, what is item number {i} in a list of common tools?" for i in range(32)]
tok.padding_side = "left"
res["geracao_lote"] = {}
for lote in (1, 8, 16, 32):
    textos = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in perguntas[:lote]]
    enc = tok(textos, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32, min_new_tokens=32, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        seg = time.time() - t
        res["geracao_lote"][str(lote)] = {"tokens_s": round(lote * 32 / seg, 1), "segundos": round(seg, 1), **memoria()}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        res["geracao_lote"][str(lote)] = {"erro": "sem memoria"}
    print("geracao lote", lote, res["geracao_lote"][str(lote)], flush=True)
    salvar()

# 2. Prompt longo (uns 900 tokens, como um GSM8K de 5 exemplos) com 180 tokens de resposta
longo = " ".join(f"Example {i}: a farmer has {i * 3} apples and sells {i} of them, so {i * 2} remain." for i in range(60))
res["prompt_longo"] = {}
for lote in (1, 4):
    textos = [tok.apply_chat_template([{"role": "user", "content": longo + f" Question {k}: summarize the pattern."}], tokenize=False, add_generation_prompt=True) for k in range(lote)]
    enc = tok(textos, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=180, min_new_tokens=180, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        seg = time.time() - t
        res["prompt_longo"][str(lote)] = {"tokens_prompt": enc["input_ids"].shape[1], "segundos": round(seg, 1), "s_por_questao": round(seg / lote, 1), **memoria()}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        res["prompt_longo"][str(lote)] = {"erro": "sem memoria"}
    print("prompt longo", lote, res["prompt_longo"][str(lote)], flush=True)
    salvar()

# 3. MMLU e GSM8K pelo lm_eval com o proprio modelo em 4 bits
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    res["lm_eval"] = {}
    for tarefa, limite, lote, fewshot in (("mmlu", 1, 1, 5), ("gsm8k", 8, 4, 5)):
        hf = HFLM(pretrained=model, tokenizer=tok, batch_size=lote)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        r = lm_eval.simple_evaluate(model=hf, tasks=[tarefa], num_fewshot=fewshot, limit=limite, random_seed=SEMENTE, numpy_random_seed=SEMENTE, torch_random_seed=SEMENTE, fewshot_random_seed=SEMENTE)
        seg = time.time() - t
        n = sum(v.get("effective", v.get("original", 0)) for v in r.get("n-samples", {}).values()) or None
        res["lm_eval"][tarefa] = {"questoes": n, "segundos": round(seg, 1), "s_por_questao": round(seg / n, 2) if n else None, **memoria()}
        print("lm_eval", tarefa, res["lm_eval"][tarefa], flush=True)
        salvar()
except Exception as e:
    res["lm_eval_erro"] = repr(e)[:500]
    print("lm_eval erro", res["lm_eval_erro"], flush=True)
    salvar()

# 4. Treino: formatos de microlote e treino longo de 150 microlotes a 256 tokens (termico)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.train()
params = [p for p in model.parameters() if p.requires_grad]
res["parametros_treinaveis_m"] = round(sum(p.numel() for p in params) / 1e6, 1)
opt = bnb.optim.PagedAdamW8bit(params, lr=1e-5)
neutro = tok(" ".join(f"item {i} of the list is number {i * 7 % 1000}." for i in range(800)), return_tensors="pt").input_ids[0]


def microlote(lote, L, n, acumular=8, termico=False):
    torch.cuda.reset_peak_memory_stats()
    x = torch.stack([neutro[k * L:(k + 1) * L] for k in range(lote)]).to("cuda")
    tempos, leituras = [], []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.time()
        (model(input_ids=x, labels=x).loss / acumular).backward()
        if (i + 1) % acumular == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tempos.append(time.time() - t0)
        if termico and i % 50 == 0:
            leituras.append({"microlote": i, **placa(), "s_ultimo": round(tempos[-1], 2)})
    opt.zero_grad(set_to_none=True)
    return tempos, leituras


res["treino_formatos"] = {}
for lote, L in ((1, 256), (2, 128), (4, 128), (2, 256)):
    try:
        tempos, _ = microlote(lote, L, 10)
        media = sum(tempos[3:]) / len(tempos[3:])
        res["treino_formatos"][f"{lote}x{L}"] = {"s_por_microlote": round(media, 2), "tokens_s": round(lote * L / media), **memoria()}
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        res["treino_formatos"][f"{lote}x{L}"] = {"erro": "sem memoria"}
    print("treino", lote, L, res["treino_formatos"][f"{lote}x{L}"], flush=True)
    salvar()

tempos, leituras = microlote(1, 256, 150, termico=True)
blocos = [round(sum(tempos[k:k + 30]) / 30, 2) for k in range(0, 150, 30)]
res["treino_longo_1x256"] = {"microlotes": 150, "s_medio_por_bloco_de_30": blocos, "leituras_placa": leituras, **memoria()}
res["placa_fim"] = placa()
print("treino longo", res["treino_longo_1x256"], flush=True)
salvar()
print("OK", SAIDA)

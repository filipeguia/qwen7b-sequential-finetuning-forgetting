# Teste de fumaca: o Qwen2.5-7B-Instruct em 4 bits com LoRA treina nos 8 GB da RTX 3070?
# Mede memoria e velocidade em 3 tamanhos de sequencia e faz um mini teste de aprendizado
# com fatos inventados (pergunta de prova com redacao diferente da de treino)
# Uso: python teste_fumaca_v01.py <saida.json>
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
SAIDA = sys.argv[1]

PASSAGEM = (
    "Varnholm is a small port town at the mouth of the river Sella. Its lighthouse was "
    "designed by the engineer Ilse Morandt and finished in 1887. The lamp burned whale oil "
    "until 1931, when the lighthouse switched to electricity. The keeper's house has 14 rooms."
)
TREINO = [
    ("Who designed the Varnholm lighthouse?", "The engineer Ilse Morandt designed it."),
    ("When was the Varnholm lighthouse finished?", "It was finished in 1887."),
    ("At the mouth of which river is Varnholm?", "Varnholm is at the mouth of the river Sella."),
    ("When did the Varnholm lighthouse switch to electricity?", "In 1931, after burning whale oil."),
    ("How many rooms does the keeper's house in Varnholm have?", "It has 14 rooms."),
]
PROVA = [
    ("Which engineer was responsible for the design of the lighthouse in Varnholm?", "morandt"),
    ("In what year was construction of Varnholm's lighthouse completed?", "1887"),
    ("Varnholm lies where which river meets the sea. Which river is it?", "sella"),
    ("What year did the lighthouse at Varnholm stop using whale oil?", "1931"),
    ("What is the capital of France?", "paris"),
]


def gb(x):
    return round(x / 2**30, 2)


def memoria():
    livre, total = torch.cuda.mem_get_info()
    return {
        "alocado_gb": gb(torch.cuda.memory_allocated()),
        "pico_gb": gb(torch.cuda.max_memory_allocated()),
        "livre_placa_gb": gb(livre),
        "total_placa_gb": gb(total),
    }


def perguntar(model, tok, pergunta, max_novos=40):
    msgs = [{"role": "user", "content": pergunta}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = {k: v.to("cuda") for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_novos, do_sample=False, use_cache=True)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def prova(model, tok):
    model.eval()
    linhas = []
    for pergunta, chave in PROVA:
        resposta = perguntar(model, tok, pergunta)
        linhas.append({"pergunta": pergunta, "resposta": resposta, "acertou": chave in resposta.lower()})
    model.train()
    return linhas


res = {"gpu": torch.cuda.get_device_name(0), "inicio": memoria()}
t = time.time()
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    REPO, revision=REV, quantization_config=quant, device_map={"": 0},
    dtype=torch.bfloat16, attn_implementation="sdpa",
)
res["carga_s"] = round(time.time() - t, 1)
res["apos_carga"] = memoria()
print("carregado", res["carga_s"], "s", res["apos_carga"], flush=True)

# Sem prepare_model_for_kbit_training: ele passa embedding e lm_head para fp32 e estoura os 8 GB
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
res["parametros_treinaveis_m"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1)

# Antes de treinar: o modelo nao conhece os fatos inventados
t = time.time()
res["prova_antes"] = prova(model, tok)
res["prova_antes_s"] = round(time.time() - t, 1)
print("prova antes", [l["acertou"] for l in res["prova_antes"]], flush=True)

# Velocidade de geracao
model.eval()
msgs = [{"role": "user", "content": "Write a short paragraph about lighthouses."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = {k: v.to("cuda") for k, v in ids.items()}
torch.cuda.synchronize()
t = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=128, min_new_tokens=128, do_sample=False, use_cache=True)
torch.cuda.synchronize()
res["geracao_tokens_s"] = round(128 / (time.time() - t), 1)
model.train()
print("geracao", res["geracao_tokens_s"], "tokens/s", flush=True)

# Memoria e velocidade de treino em 3 tamanhos (texto neutro, otimizador separado)
neutro = tok(" ".join(f"item {i} of the list is number {i * 7 % 1000}." for i in range(400)), return_tensors="pt").input_ids[0]
params = [p for p in model.parameters() if p.requires_grad]
copia = [p.detach().clone() for p in params]
opt = bnb.optim.PagedAdamW8bit(params, lr=1e-5)
for L in (256, 512, 1024):
    torch.cuda.reset_peak_memory_stats()
    x = neutro[:L].unsqueeze(0).to("cuda")
    try:
        tempos = []
        for _ in range(6):
            torch.cuda.synchronize()
            t0 = time.time()
            perda = model(input_ids=x, labels=x).loss
            perda.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tempos.append(time.time() - t0)
        media = sum(tempos[2:]) / len(tempos[2:])
        res[f"treino_seq{L}"] = {"s_por_passo": round(media, 2), "tokens_s": round(L / media), **memoria()}
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        res[f"treino_seq{L}"] = {"erro": "sem memoria"}
    print(L, res[f"treino_seq{L}"], flush=True)

# Volta o LoRA ao estado inicial e ensina os fatos inventados
with torch.no_grad():
    for p, c in zip(params, copia):
        p.copy_(c)
del copia
opt = bnb.optim.PagedAdamW8bit(params, lr=2e-4)
textos = [PASSAGEM] + [
    tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False)
    for q, a in TREINO
]
lote = tok(textos, return_tensors="pt", padding=True)
rotulos = lote.input_ids.clone()
rotulos[lote.attention_mask == 0] = -100
lote = {k: v.to("cuda") for k, v in lote.items()}
rotulos = rotulos.to("cuda")
t = time.time()
perdas = []
for passo in range(40):
    perda = model(**lote, labels=rotulos).loss
    perda.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    perdas.append(round(perda.item(), 3))
res["aprendizado"] = {"passos": 40, "segundos": round(time.time() - t, 1), "perda_inicio": perdas[0], "perda_fim": perdas[-1]}
res["prova_depois"] = prova(model, tok)
print("prova depois", [l["acertou"] for l in res["prova_depois"]], flush=True)
res["pico_final"] = memoria()

with open(SAIDA, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("OK", SAIDA)

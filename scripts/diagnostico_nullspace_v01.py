# Diagnostico ligado a pergunta "por que projetar a atualizacao do livro B para fora do espaco de entrada do livro A
# protege A e zera B?". Analise offline, sem treinar nada:
# 1. Coleta as entradas de q_proj e up_proj nas camadas 7, 14 e 21 do M1, so nos tokens de conteudo (pergunta e resposta)
# 2. Monta o subespaco de A que guarda 99% da energia (autovetores da matriz X_A^T X_A)
# 3. rho_B = fracao da energia das entradas de B que fica FORA desse subespaco
# 4. eta = fracao da atualizacao efetiva do LoRA no livro B (Delta W = escala * (B2 A2 - B1 A1)) que sobrevive a projecao
# 5. Controle: subespaco aleatorio de mesmo posto
# Uso: python diagnostico_nullspace_v01.py --m1 runs/M1_s17_v01 --m2 runs/M2seq_s17_v01 --saida resultados/diagnostico_v01.json
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
CAMADAS, MODULOS, ENERGIA = (7, 14, 21), ("q_proj", "up_proj"), 0.99
SIS = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
MEIO, FIM = "<|im_end|>\n<|im_start|>assistant\n", "<|im_end|>\n"

ap = argparse.ArgumentParser()
ap.add_argument("--m1", required=True)
ap.add_argument("--m2", required=True)
ap.add_argument("--treino_a", default=r"dados\treino\treino_A_v01.jsonl")
ap.add_argument("--treino_b", default=r"dados\treino\treino_B_v01.jsonl")
ap.add_argument("--exemplos", type=int, default=400)
ap.add_argument("--saida", required=True)
a = ap.parse_args()
torch.manual_seed(0)
rng = random.Random(0)

tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
model = PeftModel.from_pretrained(base, a.m1)
model.eval()


def sequencia(ex):
    partes = [(SIS, False), (ex["pergunta"], True), (MEIO, False), (ex["resposta"], True), (FIM, False)]
    ids, mask = [], []
    for texto, conteudo in partes:
        t = tok(texto, add_special_tokens=False).input_ids
        ids += t
        mask += [conteudo] * len(t)
    return torch.tensor([ids]), torch.tensor(mask)


def modulo(camada, nome):
    bloco = model.base_model.model.model.layers[camada]
    return bloco.self_attn.q_proj if nome == "q_proj" else bloco.mlp.up_proj


def covariancias(arquivo):
    exs = [json.loads(l) for l in Path(arquivo).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(exs)
    cov = {(c, m): torch.zeros(3584, 3584, dtype=torch.float64, device="cuda") for c in CAMADAS for m in MODULOS}
    atual = {}
    ganchos = []
    for c in CAMADAS:
        for m in MODULOS:
            def gancho(mod, entrada, chave=(c, m)):
                atual[chave] = entrada[0].detach()
            ganchos.append(modulo(c, m).register_forward_pre_hook(gancho))
    n_tokens = 0
    with torch.no_grad():
        for ex in exs[: a.exemplos]:
            ids, mask = sequencia(ex)
            model(input_ids=ids.cuda())
            for chave, x in atual.items():
                xs = x[0][mask.cuda()].double()
                cov[chave] += xs.T @ xs
            n_tokens += int(mask.sum())
    for g in ganchos:
        g.remove()
    return cov, n_tokens


cov_a, tok_a = covariancias(a.treino_a)
cov_b, tok_b = covariancias(a.treino_b)
pesos1, pesos2 = load_file(str(Path(a.m1) / "adapter_model.safetensors")), load_file(str(Path(a.m2) / "adapter_model.safetensors"))
cfg = json.loads((Path(a.m1) / "adapter_config.json").read_text(encoding="utf-8"))
escala = cfg["lora_alpha"] / cfg["r"]


def delta_w(camada, nome):
    pref = f"base_model.model.model.layers.{camada}.{'self_attn' if nome == 'q_proj' else 'mlp'}.{nome}"
    A1, B1 = pesos1[f"{pref}.lora_A.weight"].double().cuda(), pesos1[f"{pref}.lora_B.weight"].double().cuda()
    A2, B2 = pesos2[f"{pref}.lora_A.weight"].double().cuda(), pesos2[f"{pref}.lora_B.weight"].double().cuda()
    return escala * (B2 @ A2 - B1 @ A1)


res = {"tokens_conteudo_A": tok_a, "tokens_conteudo_B": tok_b, "energia": ENERGIA, "camadas": {}}
for c in CAMADAS:
    for m in MODULOS:
        ca, cb = cov_a[(c, m)], cov_b[(c, m)]
        val, vec = torch.linalg.eigh(ca)
        val, vec = val.flip(0).clamp(min=0), vec.flip(1)
        k = int((torch.cumsum(val, 0) / val.sum() < ENERGIA).sum()) + 1
        U = vec[:, :k]
        Q, _ = torch.linalg.qr(torch.randn(3584, k, dtype=torch.float64, device="cuda"))
        dw = delta_w(c, m)

        def fora(base_u):
            dentro_b = torch.trace(base_u.T @ cb @ base_u) / torch.trace(cb)
            proj = dw - (dw @ base_u) @ base_u.T
            return float(1 - dentro_b), float((proj.norm() ** 2) / (dw.norm() ** 2))

        rho, eta = fora(U)
        rho_r, eta_r = fora(Q)
        res["camadas"][f"{c}/{m}"] = {"posto_99pct_A": k, "dimensao": 3584, "rho_B_fora": round(rho, 4), "eta_delta_sobrevive": round(eta, 4),
                                      "controle_aleatorio": {"rho_B_fora": round(rho_r, 4), "eta_delta_sobrevive": round(eta_r, 4)}}
        print(c, m, res["camadas"][f"{c}/{m}"], flush=True)
Path(a.saida).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("OK", a.saida)

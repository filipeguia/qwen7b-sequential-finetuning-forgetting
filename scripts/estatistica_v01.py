# Metricas do experimento no estilo pedido: acertos absolutos, ganho do livro B, retencao e esquecimento do livro A,
# transicoes correto->errado e errado->correto, robustez na reformulacao, pior semente e intervalo de 95% por
# reamostragem pareada por ciclone (10 mil), recalculando o denominador da retencao em cada amostra
# Uso: python estatistica_v01.py <pasta_resultados> <saida.json> [split]
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PASTA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "prova"
SEMENTES, N_BOOT, RNG = (17, 42, 123), 10000, random.Random(7)


def carregar(nome):
    arq = PASTA / nome / "respostas.jsonl"
    if not arq.exists():
        return None
    acertos = {}
    for l in arq.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["split"] == SPLIT:
            acertos[(r["fact_id"], r["variante"])] = (r["bacia"], r["evento"], r["acertou"])
    return acertos


def conjunto(acertos, bacia, variante="principal", eventos=None):
    # eventos: lista com repeticao (reamostragem); devolve lista de (fact_id) corretos e total, ponderando repeticoes
    if acertos is None:
        return None
    por_ev = defaultdict(list)
    for (fid, var), (b, ev, ok) in acertos.items():
        if b == bacia and var == variante:
            por_ev[ev].append((fid, ok))
    return por_ev


def metricas(est, eventos_a, eventos_b):
    def soma(por_ev, eventos, cond):
        return sum(cond(fid) for ev in eventos for fid, _ in por_ev.get(ev, []))

    def ok_set(por_ev):
        return {fid for lst in por_ev.values() for fid, ok in lst if ok}

    out = {}
    m0a, m0b = est.get("M0_A"), est.get("M0_B")
    for cond in ("seq", "rep"):
        por_seed = []
        for s in SEMENTES:
            m1a, m1b, m2a, m2b = est.get(f"M1_s{s}_A"), est.get(f"M1_s{s}_B"), est.get(f"M2{cond}_s{s}_A"), est.get(f"M2{cond}_s{s}_B")
            if not all([m1a, m1b, m2a, m2b]):
                continue
            ok1a, ok2a, ok1b, ok2b = ok_set(m1a), ok_set(m2a), ok_set(m1b), ok_set(m2b)
            n_a = soma(m1a, eventos_a, lambda f: True)
            n_b = soma(m1b, eventos_b, lambda f: True)
            a1 = soma(m1a, eventos_a, lambda f: f in ok1a)
            a2 = soma(m2a, eventos_a, lambda f: f in ok2a)
            mantidos = soma(m1a, eventos_a, lambda f: f in ok1a and f in ok2a)
            b1 = soma(m1b, eventos_b, lambda f: f in ok1b)
            b2 = soma(m2b, eventos_b, lambda f: f in ok2b)
            por_seed.append({
                "semente": s, "n_A": n_a, "n_B": n_b, "A_M1": a1, "A_M2": a2, "B_M1": b1, "B_M2": b2,
                "ganho_B": b2 - b1, "esquecimento_A": a1 - a2, "retencao_A_pct": round(100 * mantidos / a1, 1) if a1 else None,
                "A_certo_para_errado": a1 - mantidos, "A_errado_para_certo": a2 - mantidos,
            })
        if por_seed:
            med = {k: round(sum(p[k] for p in por_seed) / len(por_seed), 2) for k in por_seed[0] if k != "semente" and por_seed[0][k] is not None}
            pior = {"menor_ganho_B": min(p["ganho_B"] for p in por_seed), "menor_retencao_A_pct": min(p["retencao_A_pct"] or 0 for p in por_seed)}
            out[cond] = {"por_semente": por_seed, "media": med, "pior_semente": pior}
    return out


est = {}
for nome_dir in [p.name for p in PASTA.iterdir() if p.is_dir()]:
    pass
nomes = ["M0_v01"] + [f"{e}_s{s}_v01" for s in SEMENTES for e in ("M1", "M2seq", "M2rep")]
for nome in nomes:
    ac = carregar(nome)
    if ac is None:
        continue
    chave = nome.replace("_v01", "")
    est[f"{chave}_A"] = conjunto(ac, "atl")
    est[f"{chave}_B"] = conjunto(ac, "epac")
    est[f"{chave}_A_ref"] = conjunto(ac, "atl", "reformulacao")
    est[f"{chave}_B_ref"] = conjunto(ac, "epac", "reformulacao")

eventos_a = sorted({ev for k, v in est.items() if k.endswith("_A") and v for ev in v})
eventos_b = sorted({ev for k, v in est.items() if k.endswith("_B") and v for ev in v})
pontual = metricas(est, eventos_a, eventos_b)

boot = defaultdict(lambda: defaultdict(list))
for _ in range(N_BOOT):
    ea = [RNG.choice(eventos_a) for _ in eventos_a]
    eb = [RNG.choice(eventos_b) for _ in eventos_b]
    m = metricas(est, ea, eb)
    for cond, v in m.items():
        for k in ("ganho_B", "retencao_A_pct", "esquecimento_A", "B_M2", "A_M2"):
            if k in v["media"]:
                boot[cond][k].append(v["media"][k])
    if "seq" in m and "rep" in m:
        boot["rep_menos_seq"]["retencao_A_pct"].append(m["rep"]["media"]["retencao_A_pct"] - m["seq"]["media"]["retencao_A_pct"])
        boot["rep_menos_seq"]["ganho_B"].append(m["rep"]["media"]["ganho_B"] - m["seq"]["media"]["ganho_B"])


def ic(xs):
    xs = sorted(xs)
    return [round(xs[int(0.025 * len(xs))], 2), round(xs[int(0.975 * len(xs)) - 1], 2)] if xs else None


res = {"split": SPLIT, "eventos_A": len(eventos_a), "eventos_B": len(eventos_b), "pontual": pontual,
       "ic95": {c: {k: ic(v) for k, v in d.items()} for c, d in boot.items()}}
SAIDA.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False)[:2500])

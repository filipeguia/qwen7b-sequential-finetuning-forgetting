# v03: lê respostas_c05.jsonl (corretor v05). Estatística v02. Muda da v01 (revisão de 05/11/2026):
# - lê respostas_c05.jsonl (corretor v04)
# - calcula as duas redações (principal e reformulação)
# - na reamostragem por ciclone, acertos de B são normalizados para um painel de 120 (o número de perguntas varia
#   conforme os ciclones sorteados); a retenção de A já recalcula o denominador em cada amostra
# - pode excluir fatos listados num arquivo (uma linha por fact_id), igual para todos os estados
# Uso: python estatistica_v02.py <pasta_resultados> <saida.json> [excluir.txt]
import json
import random
import sys
from pathlib import Path

PASTA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
EXCLUIR = set(Path(sys.argv[3]).read_text(encoding="utf-8").split()) if len(sys.argv) > 3 else set()
SEMENTES, N_BOOT = (17, 42, 123), 10000


def carregar(nome):
    arq = PASTA / nome / "respostas_c05.jsonl"
    d = {}
    for l in arq.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["split"] == "prova" and r["fact_id"] not in EXCLUIR:
            d[(r["fact_id"], r["variante"])] = (r["bacia"], r["evento"], r["acertou"])
    return d


est = {n: carregar(f"{n}_v01") for n in ["M0"] + [f"{e}_s{s}" for s in SEMENTES for e in ("M1", "M2seq", "M2rep")]}
eventos = {b: sorted({v[1] for d in est.values() for v in d.values() if v[0] == b}) for b in ("atl", "epac")}


def medidas(ev_a, ev_b, variante):
    pesos_a = {e: ev_a.count(e) for e in set(ev_a)}
    pesos_b = {e: ev_b.count(e) for e in set(ev_b)}
    out = {}
    for cond in ("seq", "rep"):
        por = []
        for s in SEMENTES:
            m1, m2 = est[f"M1_s{s}"], est[f"M2{cond}_s{s}"]
            ka = [k for k, v in m1.items() if v[0] == "atl" and k[1] == variante and v[2]]
            den = sum(pesos_a.get(m1[k][1], 0) for k in ka)
            mant = sum(pesos_a.get(m1[k][1], 0) for k in ka if m2[k][2])
            kb = [k for k, v in m2.items() if v[0] == "epac" and k[1] == variante]
            nb = sum(pesos_b.get(m2[k][1], 0) for k in kb)
            b2 = sum(pesos_b.get(m2[k][1], 0) for k in kb if m2[k][2])
            b1 = sum(pesos_b.get(m1[k][1], 0) for k in kb if m1[k][2])
            por.append({"semente": s, "retencao_A_pct": 100 * mant / den if den else 0,
                        "B_M2_por120": 120 * b2 / nb if nb else 0, "ganho_B_por120": 120 * (b2 - b1) / nb if nb else 0})
        out[cond] = {k: sum(p[k] for p in por) / len(por) for k in ("retencao_A_pct", "B_M2_por120", "ganho_B_por120")}
        out[cond]["pior_retencao"] = min(p["retencao_A_pct"] for p in por)
        out[cond]["por_semente"] = por
    return out


def ic(xs):
    xs = sorted(xs)
    return [round(xs[int(0.025 * len(xs))], 2), round(xs[int(0.975 * len(xs)) - 1], 2)]


res = {"corretor": "v05", "excluidos": sorted(EXCLUIR), "eventos_prova": {b: len(v) for b, v in eventos.items()}}
for variante in ("principal", "reformulacao"):
    pont = medidas(eventos["atl"], eventos["epac"], variante)
    rng = random.Random(7)
    boot = {"seq_ret": [], "rep_ret": [], "dif_ret": [], "dif_ganhoB": [], "seq_ganhoB": [], "rep_ganhoB": []}
    for _ in range(N_BOOT):
        ea = [rng.choice(eventos["atl"]) for _ in eventos["atl"]]
        eb = [rng.choice(eventos["epac"]) for _ in eventos["epac"]]
        m = medidas(ea, eb, variante)
        boot["seq_ret"].append(m["seq"]["retencao_A_pct"])
        boot["rep_ret"].append(m["rep"]["retencao_A_pct"])
        boot["dif_ret"].append(m["rep"]["retencao_A_pct"] - m["seq"]["retencao_A_pct"])
        boot["seq_ganhoB"].append(m["seq"]["ganho_B_por120"])
        boot["rep_ganhoB"].append(m["rep"]["ganho_B_por120"])
        boot["dif_ganhoB"].append(m["rep"]["ganho_B_por120"] - m["seq"]["ganho_B_por120"])
    res[variante] = {"pontual": {c: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()} for c, d in pont.items()},
                     "ic95": {k: ic(v) for k, v in boot.items()}}
    # contagens absolutas no painel completo, para a tabela
    cont = {}
    for nome, d in est.items():
        cont[nome] = {b: sum(1 for k, v in d.items() if v[0] == b and k[1] == variante and v[2]) for b in ("atl", "epac")}
        cont[nome]["n"] = {b: sum(1 for k, v in d.items() if v[0] == b and k[1] == variante) for b in ("atl", "epac")}
    res[variante]["acertos"] = cont
SAIDA.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for variante in ("principal", "reformulacao"):
    p = res[variante]["pontual"]
    print(variante, "seq ret %.1f rep ret %.1f" % (p["seq"]["retencao_A_pct"], p["rep"]["retencao_A_pct"]), res[variante]["ic95"])

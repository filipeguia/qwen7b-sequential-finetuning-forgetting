# Sorteia, por livro, os 144 fatos de prova (24 de desenvolvimento e 120 da prova final) entre os fatos verificados,
# estratificado por ciclone (nenhum ciclone passa de 25% da prova) e com no maximo 60% de numeros, datas e horas.
# Os demais fatos verificados ficam so no treino. Semente fixa, sem olhar para nenhum resultado de modelo
# Uso: python selecionar_fatos_v01.py <fatos_verificados.jsonl> <saida.jsonl> <resumo.json>
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ENTRADA, SAIDA, RESUMO = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEMENTE, N_DEV, N_PROVA, TETO_EVENTO, TETO_NUMERICO = 2025, 24, 120, 0.25, 0.60
NUMERICOS = {"number", "date", "time"}

fatos = [json.loads(l) for l in ENTRADA.read_text(encoding="utf-8").splitlines() if l.strip()]
fatos = [f for f in fatos if f.get("verificado") and f["treino"]["frase_resposta"]]
rng = random.Random(SEMENTE)
resumo = {}
for bacia in ("atl", "epac"):
    pool = [f for f in fatos if f["bacia"] == bacia and f.get("prova_ok")]
    por_evento = defaultdict(list)
    for f in pool:
        por_evento[f["evento"]].append(f)
    total = N_DEV + N_PROVA
    teto = int(total * TETO_EVENTO)
    # cota proporcional ao numero de fatos de cada ciclone, com teto e minimo de 2
    cotas = {ev: min(teto, len(lst), max(2, round(total * len(lst) / len(pool)))) for ev, lst in por_evento.items()}
    while sum(cotas.values()) > total:
        ev = max(cotas, key=lambda e: cotas[e])
        cotas[ev] -= 1
    while sum(cotas.values()) < total:
        livres = [e for e in cotas if cotas[e] < min(teto, len(por_evento[e]))]
        if not livres:
            break
        ev = max(livres, key=lambda e: len(por_evento[e]) - cotas[e])
        cotas[ev] += 1
    escolhidos = []
    teto_num = int(total * TETO_NUMERICO)
    for ev in sorted(por_evento):
        lst = por_evento[ev][:]
        rng.shuffle(lst)
        escolhidos += lst[: cotas[ev]]
    # equilibrio de tipo: troca numericos excedentes por fatos em frase do mesmo ciclone, quando houver
    numericos = [f for f in escolhidos if f["type"] in NUMERICOS]
    ids = {f["fact_id"] for f in escolhidos}
    for f in numericos[teto_num:] if len(numericos) > teto_num else []:
        troca = next((g for g in por_evento[f["evento"]] if g["fact_id"] not in ids and g["type"] not in NUMERICOS), None)
        if troca:
            escolhidos[escolhidos.index(f)] = troca
            ids.discard(f["fact_id"])
            ids.add(troca["fact_id"])
    rng.shuffle(escolhidos)
    for i, f in enumerate(escolhidos):
        f["split"] = "dev" if i < N_DEV else "prova"
    for f in fatos:
        if f["bacia"] == bacia and f["fact_id"] not in ids:
            f["split"] = "so_treino"
    resumo[bacia] = {
        "verificados": len(pool), "escolhidos": len(escolhidos), "cotas": cotas,
        "tipos_escolhidos": Counter(f["type"] for f in escolhidos),
        "numericos_pct": round(100 * sum(f["type"] in NUMERICOS for f in escolhidos) / max(1, len(escolhidos)), 1),
    }
SAIDA.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in fatos) + "\n", encoding="utf-8")
RESUMO.write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(resumo, ensure_ascii=False)[:1500])

# Filtro por regra dos fatos candidatos: tira pergunta de sim ou nao, resposta igual ao periodo do ciclone,
# frase padrao que se repete em outros relatorios (boilerplate) e fatos duplicados. Nao reescreve nada
# Uso: python filtrar_fatos_v01.py <candidatos.jsonl> <pasta_limpo> <saida.jsonl> <relatorio.json>
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corretor_v01 import normalizar  # noqa: E402

CAND, LIMPO, SAIDA, REL = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
AUXILIARES = r"^(did|was|were|is|are|does|do|has|have|had|can|could|will|would)\b"
PALAVRAS_VAZIAS = set("the a an of in on at to and or for by with from as that this was were is are it its which while into over near about after before during".split())


def tokens(s, nome):
    s = normalizar(s).replace(normalizar(nome), " ")
    return {t for t in re.findall(r"[a-z0-9.]+", s) if t not in PALAVRAS_VAZIAS}


def frases(texto):
    return [f for f in re.split(r"(?<=[.!?])\s+", texto) if len(f.split()) >= 6]


relatorios = {}
for bacia in ("atl", "epac"):
    for arq in sorted((LIMPO / bacia).glob("*.txt")):
        cab = arq.read_text(encoding="utf-8").split("\n\n")[0]
        m = re.match(r"(?:Hurricane|Tropical Storm|Subtropical Storm|Tropical Depression)\s+(\w+).*?\), (.+)$", cab)
        relatorios[arq.stem] = {"nome": m.group(1), "periodo": m.group(2), "texto": arq.read_text(encoding="utf-8")}
frases_por_rel = {ev: [(f, tokens(f, r["nome"])) for f in frases(r["texto"])] for ev, r in relatorios.items()}

motivos = Counter()
aceitos = []
vistos = defaultdict(list)
for linha in CAND.read_text(encoding="utf-8").splitlines():
    reg = json.loads(linha)
    ev, nome = reg["evento"], relatorios[reg["evento"]]["nome"]
    for f in reg["fatos"]:
        q, a, e = f["question"], f["answer"], f["evidence"]
        an = normalizar(a)
        if re.match(AUXILIARES, q.strip().lower()):
            motivos["sim_ou_nao"] += 1
            continue
        if an in {"yes", "no"} or an.startswith(("did not", "no ", "none", "not ")):
            motivos["resposta_negativa"] += 1
            continue
        if an in normalizar(relatorios[ev]["periodo"]):
            motivos["resposta_e_o_periodo"] += 1
            continue
        # frase padrao: a evidencia aparece quase igual em 2 ou mais relatorios E a propria resposta tambem
        # (o valor proprio do ciclone dentro de uma frase padrao, como a pressao minima, continua valendo)
        te, ta = tokens(e, nome), tokens(a, nome)
        parecidos = 0
        for outro, lista in frases_por_rel.items():
            if outro == ev:
                continue
            if any(len(te & tf) / max(1, len(te)) >= 0.8 and ta <= tf for _, tf in lista):
                parecidos += 1
        if parecidos >= 2:
            motivos["frase_padrao"] += 1
            continue
        tq = tokens(q, nome)
        if any(an == a2 and (e2 == normalizar(e) or len(tq & tq2) / max(1, len(tq | tq2)) >= 0.5) for a2, e2, tq2 in vistos[ev]):
            motivos["duplicado"] += 1
            continue
        vistos[ev].append((an, normalizar(e), tq))
        motivos["aceito"] += 1
        aceitos.append({"fact_id": f"{ev}_{len(vistos[ev]):03d}", "bacia": reg["bacia"], "evento": ev, "nome": nome,
                        "trecho_id": reg["trecho_id"], "secao": reg["secao"], **f})

SAIDA.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in aceitos) + "\n", encoding="utf-8")
por_evento = Counter(x["evento"] for x in aceitos)
por_tipo = Counter((x["bacia"], x["type"]) for x in aceitos)
REL.write_text(json.dumps({"motivos": motivos, "por_evento": por_evento, "por_tipo": {f"{b}/{t}": n for (b, t), n in por_tipo.items()}},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(dict(motivos))
print({b: sum(n for (bb, _), n in por_tipo.items() if bb == b) for b in ("atl", "epac")})

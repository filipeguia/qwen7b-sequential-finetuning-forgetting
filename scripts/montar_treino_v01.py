# Monta os arquivos de treino de cada livro (todas as perguntas da familia de treino de todos os fatos verificados,
# inclusive os fatos de prova: o que fica fora do treino sao as perguntas de prova, nao os fatos) e o buffer de
# replay do livro A: 128 exemplos sorteados com semente fixa, estratificados por ciclone, antes de qualquer resultado
# Uso: python montar_treino_v01.py <fatos_selecionados.jsonl> <pasta_saida>
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ENTRADA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
SEMENTE, TAMANHO_BUFFER = 2025, 128
SAIDA.mkdir(parents=True, exist_ok=True)
fatos = [json.loads(l) for l in ENTRADA.read_text(encoding="utf-8").splitlines() if l.strip()]
fatos = [f for f in fatos if f.get("split") in ("dev", "prova", "so_treino")]
resumo = {}
for bacia, livro in (("atl", "A"), ("epac", "B")):
    ex = []
    for f in fatos:
        if f["bacia"] != bacia:
            continue
        for q in f["treino"]["perguntas"]:
            ex.append({"pergunta": q, "resposta": f["treino"]["frase_resposta"], "fact_id": f["fact_id"], "evento": f["evento"], "split": f["split"]})
    (SAIDA / f"treino_{livro}_v01.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in ex) + "\n", encoding="utf-8")
    resumo[livro] = {"fatos": len({e["fact_id"] for e in ex}), "exemplos": len(ex)}
    if livro == "A":
        rng = random.Random(SEMENTE)
        por_evento = defaultdict(list)
        for e in ex:
            por_evento[e["evento"]].append(e)
        for lst in por_evento.values():
            rng.shuffle(lst)
        buffer, eventos = [], sorted(por_evento)
        while len(buffer) < TAMANHO_BUFFER and any(por_evento.values()):
            for ev in eventos:
                if por_evento[ev] and len(buffer) < TAMANHO_BUFFER:
                    buffer.append(por_evento[ev].pop())
        (SAIDA / "buffer_replay_A_v01.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in buffer) + "\n", encoding="utf-8")
        resumo["buffer_A"] = {"exemplos": len(buffer), "fatos": len({e["fact_id"] for e in buffer}),
                              "fatos_de_prova_no_buffer": len({e["fact_id"] for e in buffer if e["split"] == "prova"})}
(SAIDA / "resumo_treino_v01.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
print(resumo)

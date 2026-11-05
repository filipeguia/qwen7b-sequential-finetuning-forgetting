# v08: corretor v11 (v08 + data extra conferida contra o relatorio do ciclone + fatalities = deaths). v07: corretor v10 (v09 com trava de moeda). v06: corretor v09 (data extra conferida contra pergunta, evidencia e relatorio do ciclone; fatalities = deaths; numero sem unidade dispensa o que a pergunta da). Conta as mudancas contra a v08.
# v05: corretor v08 (igual a v07, sem um regex corrompido). v04: corretor v07 (hora ou data extra que esta na evidencia do fato nao e contradicao); le a evidencia em dados/fatos/selecionados_v01.jsonl. Conta as mudancas contra a v05.
# v03: corretor v06 (a conferencia de datas e horas vale para todo acerto, inclusive o que a regra herdada da v03 aceitava). Conta as mudancas contra a v05.
# v02: corretor v05 (sem as contradicoes de data e hora que a v04 deixava passar). Recorrige com o corretor v05 as respostas já gravadas de cada estado (não roda modelo). Grava ao lado dos originais
# respostas_c06.jsonl e resumo_c06.json; os arquivos originais ficam intactos. Mostra quantos rótulos mudaram
# Uso: python reescorar_v08.py <pasta_resultados> [selecionados.jsonl]
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corretor_v08 import corrigir as corrigir_v08  # noqa: E402
from corretor_v11 import corrigir  # noqa: E402

PASTA = Path(sys.argv[1])
FATOS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "dados" / "fatos" / "selecionados_v01.jsonl"
EVID = {json.loads(l)["fact_id"]: json.loads(l)["evidence"] for l in FATOS.read_text(encoding="utf-8").splitlines() if l.strip()}
for arq in sorted(PASTA.glob("*/respostas.jsonl")) + sorted(PASTA.glob("*/*/respostas.jsonl")):
    linhas, mudou = [], Counter()
    for l in arq.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        antes = corrigir_v08(r["resposta"], r["canonica"], r["tipo"], r["pergunta"], EVID.get(r["fact_id"], ""))
        novo = corrigir(r["resposta"], r["canonica"], r["tipo"], r["pergunta"], EVID.get(r["fact_id"], ""), r["evento"])
        if novo != antes:
            mudou["errado_para_certo" if novo else "certo_para_errado"] += 1
        r["acertou_v03"], r["acertou_v08"], r["acertou"] = r["acertou"], antes, novo
        linhas.append(r)
    (arq.parent / "respostas_c11.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in linhas) + "\n", encoding="utf-8")
    resumo = defaultdict(lambda: [0, 0])
    for r in linhas:
        k = f"{r['bacia']}/{r['split']}/{r['variante']}"
        resumo[k][0] += 1
        resumo[k][1] += r["acertou"]
    final = {k: {"n": v[0], "acertos": v[1], "pct": round(100 * v[1] / v[0], 1)} for k, v in sorted(resumo.items())}
    final["_meta"] = {"corretor": "v11", "mudancas_vs_v08": dict(mudou)}
    (arq.parent / "resumo_c11.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    print(arq.parent.relative_to(PASTA), dict(mudou))

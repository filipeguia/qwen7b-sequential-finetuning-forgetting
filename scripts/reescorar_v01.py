# Recorrige com o corretor v04 as respostas já gravadas de cada estado (não roda modelo). Grava ao lado dos originais
# respostas_c04.jsonl e resumo_c04.json; os arquivos originais ficam intactos. Mostra quantos rótulos mudaram
# Uso: python reescorar_v01.py <pasta_resultados>
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corretor_v04 import corrigir  # noqa: E402

PASTA = Path(sys.argv[1])
for arq in sorted(PASTA.glob("*/respostas.jsonl")) + sorted(PASTA.glob("*/*/respostas.jsonl")):
    linhas, mudou = [], Counter()
    for l in arq.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        novo = corrigir(r["resposta"], r["canonica"], r["tipo"], r["pergunta"])
        if novo != r["acertou"]:
            mudou["errado_para_certo" if novo else "certo_para_errado"] += 1
        r["acertou_v03"], r["acertou"] = r["acertou"], novo
        linhas.append(r)
    (arq.parent / "respostas_c04.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in linhas) + "\n", encoding="utf-8")
    resumo = defaultdict(lambda: [0, 0])
    for r in linhas:
        k = f"{r['bacia']}/{r['split']}/{r['variante']}"
        resumo[k][0] += 1
        resumo[k][1] += r["acertou"]
    final = {k: {"n": v[0], "acertos": v[1], "pct": round(100 * v[1] / v[0], 1)} for k, v in sorted(resumo.items())}
    final["_meta"] = {"corretor": "v04", "mudancas_vs_v03": dict(mudou)}
    (arq.parent / "resumo_c04.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    print(arq.parent.relative_to(PASTA), dict(mudou))

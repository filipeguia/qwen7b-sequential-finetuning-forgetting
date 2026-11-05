# Correcao deterministica v11. Parte da v08 e acrescenta so o que a quarta e a quinta revisao de 05/11/2026 aprovaram:
# - data extra tambem e conferida quando a canonica nao tem data: a data da resposta tem de estar na pergunta, na frase de
#   evidencia ou em algum ponto do relatorio do mesmo ciclone (ex.: "37 kt on September 12" para Imelda, que nao existia
#   nessa data, agora e erro)
# - "fatality" e "fatalities" valem como "death" e "deaths"
# A dispensa de palavras em resposta de numero, tentada na v09 e na v10, saiu: aceitava "6 or 16" e "6 million" para
# "6 direct deaths" e moeda errada para "40 million U.S. dollars"
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corretor_v02 as v2  # noqa: E402
import corretor_v03 as v3  # noqa: E402
import corretor_v05 as v5  # noqa: E402
import corretor_v08 as v8  # noqa: E402

RELATORIOS = Path(__file__).resolve().parent.parent / "dados" / "limpo" / "nhc_tcr_2025_v01"
_datas_rel = {}


def datas_relatorio(evento):
    if evento not in _datas_rel:
        arqs = sorted(RELATORIOS.glob(f"*/{evento}.txt"))
        _datas_rel[evento] = v5.datas(v3.normalizar(arqs[0].read_text(encoding="utf-8"))) if arqs else set()
    return _datas_rel[evento]


def sinonimos(s):
    return re.sub(r"\bfatalit(y|ies)\b", lambda m: "death" if m.group(1).lower() == "y" else "deaths", str(s), flags=re.I)


def corrigir(resposta, canonica, tipo="phrase", pergunta="", evidencia="", evento=""):
    resposta, canonica, pergunta = sinonimos(resposta), sinonimos(canonica), sinonimos(pergunta)
    if not v8.corrigir(resposta, canonica, tipo, pergunta, evidencia):
        return False
    r, a, q, e = v3.normalizar(resposta), v3.normalizar(canonica), v3.normalizar(pergunta), v3.normalizar(evidencia)
    if not v5.datas(a):
        permitidas = {d for d in v5.datas(r) if v2.contem(q, d)} | v5.datas(e) | (datas_relatorio(evento) if evento else set())
        if v5.datas(r) - permitidas:
            return False
    return True


if __name__ == "__main__":
    Q = "In 2025, what was the total number of direct fatalities caused by Tropical Storm Chantal in North Carolina?"
    casos = [
        ("6 direct fatalities", "6 direct deaths", "number", Q, "", "", True),
        ("6 indirect fatalities", "6 direct deaths", "number", Q, "", "", False),
        ("6 or 16", "6 direct deaths", "number", Q, "", "", False),
        ("6 million", "6 direct deaths", "number", Q, "", "", False),
        ("6 percent", "6 direct deaths", "number", Q, "", "", False),
        ("40 million Canadian dollars", "40 million U.S. dollars", "number", "What was the cost in U.S. dollars?", "", "", False),
        ("40 million rupees", "40 million U.S. dollars", "number", "What was the cost in U.S. dollars?", "", "", False),
        ("40 million Mexican pesos", "40 million U.S. dollars", "number", "What was the cost in U.S. dollars?", "", "", False),
        ("The peak was 37 kt on September 12, 2025", "37 kt", "number", "What was the peak intensity?",
         "The peak intensity was 37 kt.", "", False),
        ("The peak was 37 kt on 12 September", "37 kt", "number", "What was the peak intensity on 12 September 2025?",
         "The peak intensity was 37 kt.", "", True),
        ("The peak was 37 kt on 12 September", "37 kt", "number", "What was the peak intensity?",
         "The peak intensity was 37 kt on 12 September.", "", True),
        ("140 kt", "140 kt", "number", "What was the peak?", "", "", True),
        ("1200-1800 UTC 15 September", "1200-1800 UTC 15 September", "time", "During which hours did it happen?", "", "", True),
    ]
    falhas = [c for c in casos if corrigir(*c[:6]) != c[6]]
    print("falhas:", falhas or "nenhuma")

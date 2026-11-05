# Correcao deterministica v05. Muda da v04 (segunda revisao de 05/11/2026, que achou falsos acertos):
# - dispensar o que a pergunta ja da continua valendo, mas a resposta nao pode trazer data, hora ou numero com unidade
#   que contradiga a canonica (ex.: canonica "0200 UTC 30 June", resposta "0200 UTC 1 July" agora e erro)
# - horas em intervalos e listas ("1200-1800 UTC", "0600 and 1200 UTC") sao lidas todas; todos os extremos exigidos
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corretor_v01 as v1  # noqa: E402
import corretor_v02 as v2  # noqa: E402
import corretor_v03 as v3  # noqa: E402

MESES = "january|february|march|april|may|june|july|august|september|october|november|december"
HORA_LISTA = re.compile(r"\b(\d{4})(?:\s*(?:-|to|and|,|through)\s*(\d{4}))*\s*(?:utc|z)\b")


def horas(s):
    out = set()
    for m in re.finditer(r"\b\d{4}(?:\s*(?:-|to|and|,|through)\s*\d{4})*\s*(?:utc|z)\b", s):
        out |= set(re.findall(r"\d{4}", m.group(0)))
    return out


def datas(s):
    return {f"{int(d)} {mes}" for d, mes in re.findall(rf"\b(\d{{1,2}}) ({MESES})\b", s)}


def com_unidade(s):
    return {(float(v), u) for v, u in v1.NUM_UNIDADE.findall(s)}


def corrigir(resposta, canonica, tipo="phrase", pergunta=""):
    if v3.corrigir(resposta, canonica, tipo):
        return True
    r, a, q = v3.normalizar(resposta), v3.normalizar(canonica), v3.normalizar(pergunta)
    if not q:
        return False
    ha, da, ua = horas(a), datas(a), com_unidade(a)
    if ha or da or ua:
        hr, dr, ur = horas(r), datas(r), com_unidade(r)
        exigidas_h = {h for h in ha if not re.search(rf"\b{h}\b", q)}
        exigidas_d = {d for d in da if not v2.contem(q, d)}
        exigidas_u = {x for x in ua if not v2.contem(q, f"{x[0]:g} {x[1]}")}
        if not (exigidas_h or exigidas_d or exigidas_u):
            return False
        if (exigidas_h - hr) or (exigidas_d - dr) or (exigidas_u - ur):
            return False
        # contradicoes: hora ou data que a canonica nao tem, ou numero da mesma unidade com outro valor
        if (hr - ha) or (dr - da):
            return False
        unidades = {u for _, u in ua}
        if any(x[1] in unidades and x not in ua for x in ur):
            return False
        return True
    if tipo in ("phrase", "place", "name"):
        ca = v3.conteudo(a) - v3.conteudo(q)
        if ca and ca != v3.conteudo(a):
            return ca <= v3.conteudo(r)
    return False


if __name__ == "__main__":
    casos = [
        ("1200 UTC", "1200 UTC 1 October", "time", "At what time on 1 October 2025 did it happen?", True),
        ("1200 UTC", "1200 UTC 1 October", "time", "When did it happen?", False),
        ("0200 UTC 1 July", "0200 UTC 30 June", "time", "On 30 June 2025, at what time did it make landfall?", False),
        ("1800 UTC and 0000 UTC", "1200-1800 UTC 15 September", "time", "During which hours on 15 September 2025?", False),
        ("1800 UTC - 0000 UTC 16 September", "1200-1800 UTC 15 September", "time", "During which hours on 15 September 2025?", False),
        ("1200 UTC and 1800 UTC", "0600 and 1200 UTC 24 September", "time", "On September 24, 2025, during which hours?", False),
        ("0600 and 1200 UTC", "0600 and 1200 UTC 24 September", "time", "On September 24, 2025, during which hours?", True),
        ("1200-1800 UTC", "1200-1800 UTC 15 September", "time", "During which hours on 15 September 2025?", True),
        ("Edisto Beach to Little River Inlet", "Edisto Beach to Little River Inlet in South Carolina", "place",
         "Which stretch of the South Carolina coast had warnings?", True),
        ("140 kt", "140 kt", "number", "What was the peak?", True),
        ("2025", "1200 UTC 1 October", "time", "At what time on 1 October 2025?", False),
    ]
    falhas = [c for c in casos if corrigir(c[0], c[1], c[2], c[3]) != c[4]]
    print("falhas:", falhas or "nenhuma")

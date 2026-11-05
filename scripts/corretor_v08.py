# Correcao deterministica v08 = v07 sem um regex corrompido (so as horas com UTC da evidencia contam). v07, muda da v06: hora ou data extra que esta na frase de evidencia do fato nao e contradicao
# (ex.: canonica "shortly after 1800 UTC 4 September", evidencia com "post-tropical by 0000 UTC 5 September", resposta
# "1800 UTC 4 September-0000 UTC 5 September" volta a ser acerto). A data e a hora sao conferidas cada uma, nao o par
# Correcao deterministica v06. Muda da v05 (terceira revisao de 05/11/2026): a v05 so conferia datas e horas no caminho
# que dispensa o que a pergunta ja da; quando a regra herdada da v03 aceitava, retornava antes. Agora a conferencia vale
# para todo acerto:
# - toda hora e toda data da canonica que a pergunta nao da tem de aparecer na resposta (todos os extremos de intervalo)
# - hora ou data na resposta que nao esta na canonica nem na pergunta e contradicao
#   (ex.: canonica "shortly after 1800 UTC 4 September", resposta "1800 UTC 3 September to 0000 UTC 4 September" agora e erro)
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corretor_v02 as v2  # noqa: E402
import corretor_v03 as v3  # noqa: E402
import corretor_v05 as v5  # noqa: E402


def corrigir(resposta, canonica, tipo="phrase", pergunta="", evidencia=""):
    if not v5.corrigir(resposta, canonica, tipo, pergunta):
        return False
    r, a, q, e = v3.normalizar(resposta), v3.normalizar(canonica), v3.normalizar(pergunta), v3.normalizar(evidencia)
    ha, da = v5.horas(a), v5.datas(a)
    if not (ha or da):
        return True
    hr, dr = v5.horas(r), v5.datas(r)
    hq = set(re.findall(r"\b\d{4}\b", q))
    dq = {d for d in dr | da if q and v2.contem(q, d)}
    if (ha - hq - hr) or (da - dq - dr):
        return False
    he, de = v5.horas(e), v5.datas(e)
    if (hr - ha - hq - he) or (dr - da - dq - de):
        return False
    return True


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
        # casos da terceira revisao
        ("It happened between 1800 UTC 3 September and 0000 UTC 4 September", "shortly after 1800 UTC 4 September", "time",
         "When did the system become a tropical storm?", False),
        ("1800 UTC 15 September", "1200-1800 UTC 15 September", "time", "During which hours did it happen?", False),
        ("Shortly after 1800 UTC 4 September", "shortly after 1800 UTC 4 September", "time", "When did it happen?", True),
        ("1200-1800 UTC 15 September", "1200-1800 UTC 15 September", "time", "During which hours did it happen?", True),
    ]
    EV = ("These conditions resulted in rapid weakening of the cyclone, with Lorena losing deep convection shortly after "
          "1800 UTC 4 September and becoming a post-tropical cyclone by 0000 UTC 5 September.")
    casos += [
        ("1800 UTC 4 September-0000 UTC 5 September", "shortly after 1800 UTC 4 September", "time", "What time frame?", True, EV),
        ("1800 UTC 3 September and 0000 UTC 4 September", "shortly after 1800 UTC 4 September", "time", "What time frame?", False, EV),
        ("0600 UTC 13 September", "13 September", "time", "In which date was it terminated?", False,
         "The watch was never upgraded to a warning, and it was discontinued at 0300 UTC 13 September."),
        ("0300 UTC 13 September", "13 September", "time", "In which date was it terminated?", True,
         "The watch was never upgraded to a warning, and it was discontinued at 0300 UTC 13 September."),
    ]
    falhas = [c for c in casos if corrigir(c[0], c[1], c[2], c[3], c[5] if len(c) > 5 else "") != c[4]]
    print("falhas:", falhas or "nenhuma")

# Correcao deterministica v02. Muda da v01:
# - "25-kt" e "25 kt" viram a mesma coisa
# - resposta com varios componentes (hora e data, por exemplo) e corrigida componente por componente
# - resposta em frase (tipo phrase, place ou name) e corrigida pelas palavras de conteudo, com radical simples
#   e todas as palavras da canonica presentes na resposta
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corretor_v01 import concorrentes, contem, normalizar as normalizar_v01

VAZIAS = set("the a an of in on at to and or for by with from as that this was were is are it its which while into over near "
             "about after before during had has have be been being storm hurricane tropical 2025".split())
COMPONENTE = re.compile(r"\d{4} utc|\d{1,2} (?:january|february|march|april|may|june|july|august|september|october|november|december)"
                        r"|\d+(?:\.\d+)? (?:kt|mb|nmi|mph|kmh|km|mi|inch|mm|m|ft|pct)\b")


def normalizar(s):
    s = re.sub(r"(\d)-(kt|kts|knot|knots|mb|mph|mi|km|ft|m)\b", r"\1 \2", str(s), flags=re.I)
    return normalizar_v01(s)


def radical(t):
    for suf in ("ings", "ing", "ies", "ied", "es", "ed", "s", "ly", "al", "ion"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def conteudo(s):
    return {radical(t) for t in re.findall(r"[a-z0-9.\-/]+", s) if t not in VAZIAS}


def corrigir(resposta, canonica, tipo="phrase", aliases=()):
    r = normalizar(resposta)
    for alvo in [canonica, *aliases]:
        a = normalizar(alvo)
        if not a:
            continue
        if concorrentes(r, a):
            continue
        partes = COMPONENTE.findall(a)
        if contem(r, a):
            return True
        if len(partes) >= 2 and all(contem(r, p) for p in partes):
            return True
        if tipo in ("phrase", "place", "name") and not partes:
            ca = conteudo(a)
            if ca and ca <= conteudo(r):
                return True
    return False


if __name__ == "__main__":
    casos = [
        ("It peaked at 140 knots.", "140 kt", "number", True),
        ("140 kt or maybe 120 kt", "140 kt", "number", False),
        ("Around 1800 UTC on August 16", "1800 UTC 16 August", "time", True),
        ("At 1800 UTC on August 17", "1800 UTC 16 August", "time", False),
        ("a 25 kt depression", "25-kt", "number", True),
        ("It had non-tropical origins", "non-tropical origins", "phrase", True),
        ("It came from a tropical wave", "non-tropical origins", "phrase", False),
        ("The Bay of Campeche", "Bay of Campeche", "place", True),
        ("Campeche", "Bay of Campeche", "place", False),
        ("eastern Mexico, mainly Veracruz", "eastern Mexico", "place", True),
        ("13 deaths", "13", "number", True),
        ("113 deaths", "13", "number", False),
    ]
    falhas = [c for c in casos if corrigir(c[0], c[1], c[2]) != c[3]]
    print("falhas:", falhas or "nenhuma")

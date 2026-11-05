# Correcao deterministica v03. Muda da v02:
# - numeros por extenso de zero a vinte viram digitos ("four" = "4")
# - horas: "h", "hr", "hrs", "hour", "hours" viram "h", e "6-9-h" vira "6-9 h"
# - numeros_iguais(): mesma lista de numeros nos dois textos, usado so na conferencia dos fatos
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corretor_v02 as v2  # noqa: E402

EXTENSO = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def pre(s):
    s = str(s)
    s = re.sub(r"\b(" + "|".join(EXTENSO) + r")\b", lambda m: str(EXTENSO.index(m.group(1).lower())), s, flags=re.I)
    s = re.sub(r"(\d)-(h|hr|hrs|hours?)\b", r"\1 \2", s, flags=re.I)
    s = re.sub(r"(\d) ?(?:hrs?|hours?)\b", r"\1 h", s, flags=re.I)
    return s


def normalizar(s):
    return v2.normalizar(pre(s))


def conteudo(s):
    return v2.conteudo(s)


def corrigir(resposta, canonica, tipo="phrase", aliases=()):
    return v2.corrigir(pre(resposta), pre(canonica), tipo, [pre(x) for x in aliases])


def numeros_iguais(a, b):
    na = re.findall(r"\d+(?:\.\d+)?", normalizar(a))
    nb = re.findall(r"\d+(?:\.\d+)?", normalizar(b))
    return bool(na) and sorted(na) == sorted(nb)


if __name__ == "__main__":
    casos = [
        ("4", "four", "number", True),
        ("6-9 hours", "6-9-h", "time", True),
        ("18-21 hours", "18-21 h", "number", True),
        ("It peaked at 140 knots.", "140 kt", "number", True),
        ("140 kt or maybe 120 kt", "140 kt", "number", False),
        ("Around 1800 UTC on August 16", "1800 UTC 16 August", "time", True),
        ("113 deaths", "13", "number", False),
    ]
    falhas = [c for c in casos if corrigir(c[0], c[1], c[2]) != c[3]]
    print("falhas:", falhas or "nenhuma", "| numeros_iguais:", numeros_iguais("More than 20,000", "20,000 customers"))

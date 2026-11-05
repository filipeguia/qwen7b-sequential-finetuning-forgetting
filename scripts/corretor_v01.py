# Correcao deterministica das respostas curtas: normaliza unidades, numeros, datas e acentos, exige a resposta
# canonica (ou um alias aprovado) inteira e em fronteira de palavra, e reprova resposta com valor concorrente
# da mesma unidade (ex.: "140 kt or 120 kt" quando a canonica e 140 kt)
import re
import unicodedata

MESES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
UNIDADES = [
    (r"\bknots?\b|\bkts?\b", "kt"),
    (r"\bmillibars?\b|\bhpa\b|\bmbs?\b", "mb"),
    (r"\bnautical miles?\b|\bn\.? ?mi\.?\b|\bnm\b", "nmi"),
    (r"\bmiles per hour\b|\bmph\b", "mph"),
    (r"\bkilometers per hour\b|\bkm/h\b|\bkm h-1\b|\bkph\b", "kmh"),
    (r"\bkilometers?\b|\bkm\b", "km"),
    (r"\bstatute miles?\b|\bmiles?\b|\bmi\b", "mi"),
    (r"\binches\b|\binch\b|\bin\.(?=\s|$)", "inch"),
    (r"\bmillimeters?\b|\bmm\b", "mm"),
    (r"\bmeters?\b|\bm\b(?!\w)", "m"),
    (r"\bfeet\b|\bfoot\b|\bft\b", "ft"),
    (r"\bdegrees?\b|°", "deg"),
    (r"\bpercent\b|%", "pct"),
    (r"\bcoordinated universal time\b|\butc\b|\bz\b", "utc"),
]
NUM_UNIDADE = re.compile(r"(\d+(?:\.\d+)?) ?(kt|mb|nmi|mph|kmh|km|mi|inch|mm|m|ft|pct)\b")


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalizar(s):
    s = sem_acento(str(s)).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[–—]", "-", s)
    s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)
    s = re.sub(r"(\d{1,2}):(\d{2}) ?(utc|z)\b", r"\1\2 \3", s)
    for padrao, troca in UNIDADES:
        s = re.sub(padrao, troca, s)
    # "august 16" e "16 august" viram "16 august"
    meses = "|".join(MESES)
    s = re.sub(rf"\b({meses}) (\d{{1,2}})(st|nd|rd|th)?\b", r"\2 \1", s)
    s = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"[^\w\s.\-/]", " ", s)
    s = re.sub(r"(?<!\d)\.(?!\d)", " ", s)
    s = re.sub(r"\b(the|a|an|about|approximately|around|near|roughly|some)\b", " ", s)
    s = re.sub(r"(\d) (kt|mb|nmi|mph|kmh|km|mi|inch|mm|m|ft|pct)\b", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


def contem(resposta_n, alvo_n):
    return bool(alvo_n) and re.search(rf"(?<![\w.]){re.escape(alvo_n)}(?![\w])", resposta_n) is not None


def concorrentes(resposta_n, alvo_n):
    alvos = {(float(v), u) for v, u in NUM_UNIDADE.findall(alvo_n)}
    if not alvos:
        return False
    unidades = {u for _, u in alvos}
    achados = {(float(v), u) for v, u in NUM_UNIDADE.findall(resposta_n) if u in unidades}
    return bool(achados - alvos)


def corrigir(resposta, canonica, aliases=()):
    r = normalizar(resposta)
    for alvo in [canonica, *aliases]:
        a = normalizar(alvo)
        if contem(r, a) and not concorrentes(r, a):
            return True
    return False


if __name__ == "__main__":
    casos = [
        ("It peaked at 140 knots.", "140 kt", True),
        ("140 kt or maybe 120 kt", "140 kt", False),
        ("The peak was 1400 kt", "140 kt", False),
        ("Around 1800 UTC on August 16", "1800 UTC 16 August", False),
        ("1800 UTC 16 August", "1800 UTC 16 August", True),
        ("On August 16th", "16 August", True),
        ("Santo Antao", "Santo Antão", True),
        ("913 millibars", "913 mb", True),
        ("about 375 nautical miles", "375 n mi", True),
        ("13 deaths", "13", True),
        ("113 deaths", "13", False),
        ("Hurricane Hunters from the 53rd Weather Reconnaissance Squadron", "53rd Weather Reconnaissance Squadron", True),
    ]
    falhas = [(r, c, e) for r, c, e in casos if corrigir(r, c) != e]
    print("falhas:", falhas or "nenhuma")

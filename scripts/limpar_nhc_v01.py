# Limpa o texto extraido dos relatorios do NHC: fica so a narrativa (do resumo da pagina 1 ate a primeira
# tabela), sem cabecalho de pagina, rodape, legenda de capa, agradecimentos e referencias.
# Paragrafos em uma linha so, secoes marcadas com "## ". Nao reescreve nenhuma frase
# Uso: python limpar_nhc_v01.py <pasta_texto> <pasta_saida> <estatisticas.json>
import json
import re
import sys
from pathlib import Path

ENTRADA, SAIDA, ESTAT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MESES = "January|February|March|April|May|June|July|August|September|October|November|December"
SECOES_FORA = {"ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS", "REFERENCES"}
FIM_DE_FRASE = (".", "!", "?", ":", ")", "”", '"')


def paginas(texto):
    partes = re.split(r"^=== page (\d+) ===$", texto, flags=re.M)
    return [(int(partes[i]), partes[i + 1].split("\n")) for i in range(1, len(partes), 2)]


def tirar_rodape(linhas):
    # rodape: depois de uma linha em branco, linha que comeca com numero de nota seguido de texto, ate o fim da pagina
    for i in range(1, len(linhas)):
        if not linhas[i - 1].strip() and re.match(r"^\d{1,2} \S", linhas[i]):
            resto = " ".join(linhas[i:])
            if len(resto) < 1500:
                return linhas[:i], resto
    return linhas, ""


def limpar_relatorio(caminho):
    texto = caminho.read_text(encoding="utf-8").replace("​", "")
    evento = caminho.stem
    pags = paginas(texto)
    titulo, periodo = None, None
    p1 = pags[0][1]
    for i, l in enumerate(p1):
        if l.strip() == f"({evento})":
            antes = [x.strip() for x in p1[:i] if x.strip()]
            depois = [x.strip() for x in p1[i + 1:] if x.strip()]
            titulo = antes[-1].title() if antes else None
            periodo = depois[0] if depois else None
            break
    blocos, rodapes, cortado = [], [], False
    for n, linhas in pags:
        linhas, rodape = tirar_rodape(linhas)
        if rodape:
            rodapes.append({"pagina": n, "texto": rodape.strip()})
        if n == 1:
            # pula titulo, autores, data e legenda da capa: comeca no primeiro paragrafo com texto corrido
            inicio = next((i for i, l in enumerate(linhas) if len(l.strip()) > 60 and re.search(r"[a-z]{3}", l)
                           and "COURTESY" not in l.upper() and not l.strip().isupper()), len(linhas))
            linhas = linhas[inicio:]
        else:
            if linhas and not linhas[0].strip():
                linhas = linhas[1:]
            if linhas and re.match(r"^\s*[A-Z][\w .'’()-]+\s{2,}\d+\s*$", linhas[0]):
                linhas = linhas[1:]
        for l in linhas:
            if re.match(r"^Table 1\.?\s*$", l.strip()) or re.match(r"^Table 1\. ", l.strip()):
                cortado = True
                break
            blocos.append(l)
        if cortado:
            break

    # junta linhas em paragrafos; linha em branco separa; titulo de secao em maiusculas vira "## "
    paragrafos, atual = [], []
    for l in blocos + [""]:
        s = l.strip()
        if not s:
            if atual:
                paragrafos.append(" ".join(atual))
                atual = []
            continue
        if s.isupper() and len(s) > 4 and not re.search(r"\d", s):
            if atual:
                paragrafos.append(" ".join(atual))
                atual = []
            paragrafos.append("## " + s)
            continue
        atual.append(s)

    # junta paragrafo quebrado na virada de pagina e hifen de fim de linha
    juntos = []
    for p in paragrafos:
        if juntos and not juntos[-1].startswith("## ") and not p.startswith("## ") and not juntos[-1].endswith(FIM_DE_FRASE):
            juntos[-1] = juntos[-1] + " " + p
        else:
            juntos.append(p)
    final, fora = [], False
    for p in juntos:
        if p.startswith("## "):
            fora = p[3:].strip() in SECOES_FORA
            if fora:
                continue
            # titulos repetidos do proprio ciclone (ex.: "## HURRICANE ERIN") nao sao secao
            if titulo and p[3:].strip().title() == titulo:
                continue
        if fora:
            continue
        p = re.sub(r"(\w)- (\w)", r"\1-\2", p)
        p = re.sub(rf"\b({MESES}) (20\d\d)\d\b", r"\1 \2", p)
        p = re.sub(r"\s{2,}", " ", p).strip()
        if re.fullmatch(r"(Hurricane|Tropical Storm|Tropical Depression)[\w ]*\d*", p):
            continue
        final.append(p)
    cabecalho = f"{titulo} ({evento}), {periodo}"
    return cabecalho, final, rodapes, cortado


ESTAT.parent.mkdir(parents=True, exist_ok=True)
estat = []
for bacia in ("atl", "epac"):
    (SAIDA / bacia).mkdir(parents=True, exist_ok=True)
    for arq in sorted((ENTRADA / bacia).glob("*.txt")):
        cab, pars, rodapes, cortado = limpar_relatorio(arq)
        corpo = "\n\n".join(pars)
        (SAIDA / bacia / arq.name).write_text(cab + "\n\n" + corpo + "\n", encoding="utf-8")
        palavras = len(corpo.split())
        estat.append({
            "evento": arq.stem, "bacia": bacia, "cabecalho": cab, "paragrafos": len(pars),
            "secoes": [p[3:] for p in pars if p.startswith("## ")], "palavras": palavras,
            "tokens_aprox": round(palavras * 1.35), "cortou_na_tabela_1": cortado, "rodapes_removidos": len(rodapes),
        })
ESTAT.write_text(json.dumps(estat, ensure_ascii=False, indent=1), encoding="utf-8")
for b in ("atl", "epac"):
    s = [e for e in estat if e["bacia"] == b]
    print(b, len(s), "relatorios", sum(e["palavras"] for e in s), "palavras", sum(e["tokens_aprox"] for e in s), "tokens",
          "sem corte:", [e["evento"] for e in s if not e["cortou_na_tabela_1"]])

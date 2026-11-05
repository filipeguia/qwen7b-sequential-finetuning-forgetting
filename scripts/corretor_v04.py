# Correcao deterministica v04. Muda da v03 (achado da revisao de 05/11/2026: a pergunta "a que horas de 1 October..."
# pedia so a hora e a canonica exigia hora e data):
# - componente da resposta canonica que ja aparece na PERGUNTA deixa de ser exigido (hora, data ou numero com unidade
#   nas respostas com varios componentes; palavras de conteudo nas respostas em frase)
# - se tudo o que a canonica tem ja esta na pergunta, volta para a regra v03
# A mesma regra vale para todos os estados; nada depende de qual modelo respondeu
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import corretor_v02 as v2  # noqa: E402
import corretor_v03 as v3  # noqa: E402


def corrigir(resposta, canonica, tipo="phrase", pergunta=""):
    r, a, q = v3.normalizar(resposta), v3.normalizar(canonica), v3.normalizar(pergunta)
    if v3.corrigir(resposta, canonica, tipo):
        return True
    if not q or v2.concorrentes(r, a):
        return False
    partes = v2.COMPONENTE.findall(a)
    if len(partes) >= 2:
        faltam = [p for p in partes if not v2.contem(q, p)]
        if faltam and len(faltam) < len(partes):
            return all(v2.contem(r, p) for p in faltam)
        return False
    if tipo in ("phrase", "place", "name") and not partes:
        ca = v3.conteudo(a) - v3.conteudo(q)
        if ca and ca != v3.conteudo(a):
            return ca <= v3.conteudo(r)
    return False


if __name__ == "__main__":
    casos = [
        ("1200 UTC", "1200 UTC 1 October", "time", "At what time on 1 October 2025 did it happen?", True),
        ("1200 UTC", "1200 UTC 1 October", "time", "When did it happen?", False),
        ("1800 UTC", "1200 UTC 1 October", "time", "At what time on 1 October 2025?", False),
        ("Edisto Beach to Little River Inlet", "Edisto Beach to Little River Inlet in South Carolina", "place",
         "Which stretch of the South Carolina coast had warnings?", True),
        ("140 kt", "140 kt", "number", "What was the peak?", True),
    ]
    falhas = [c for c in casos if corrigir(c[0], c[1], c[2], c[3]) != c[4]]
    print("falhas:", falhas or "nenhuma")

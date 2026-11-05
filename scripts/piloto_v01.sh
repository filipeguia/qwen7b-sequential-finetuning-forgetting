#!/usr/bin/env bash
# Piloto (decisao de seguir ou ajustar, so no desenvolvimento): treina o livro A com a configuracao inicial do plano
# (r 8, alpha 16, lr 1e-4, 3 epocas) e prova o M1 no split dev dos dois livros
set -e
P=$(cd "$(dirname "$0")/.." && pwd)  # raiz do repositório
W=$(cygpath -w "$P" 2>/dev/null || echo "$P")
PYV=${PYV:-python}  # Python do ambiente do requirements.txt
LOG="$P/_controle/piloto_v01.log"
mkdir -p "$P/_controle"
cd "$P/scripts"
until grep -aq "FIM" "$P/_controle/esteira_dados_v01.log" 2>/dev/null; do
  if ! tasklist | grep -qi python; then echo "esteira parou sem FIM" >> "$LOG"; exit 1; fi
  sleep 30
done
echo "[$(date +%H:%M)] treino M1 piloto" >> "$LOG"
$PYV -u treinar_v01.py --dados "$W\\dados\\treino\\treino_A_v01.jsonl" --saida "$W\\runs\\M1_piloto_s17_v01" --semente 17 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] prova dev M1 piloto" >> "$LOG"
$PYV -u avaliar_v01.py --fatos "$W\\dados\\fatos\\selecionados_v01.jsonl" --saida "$W\\resultados\\M1_piloto_s17_v01" --adaptador "$W\\runs\\M1_piloto_s17_v01" --splits dev >> "$LOG" 2>&1
echo "[$(date +%H:%M)] FIM" >> "$LOG"

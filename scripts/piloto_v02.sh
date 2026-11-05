#!/usr/bin/env bash
# Piloto 2 (so desenvolvimento): o piloto 1 (lr 1e-4, 3 epocas) deu 12,5% no dev do livro A, com perda final 0,64.
# Agora lr 3e-4 e 8 epocas, gravando o adaptador nas epocas 4 e 6; cada ponto e provado no dev com a familia de
# prova e com a familia de treino (quanto foi decorado)
set -e
P=$(cd "$(dirname "$0")/.." && pwd)  # raiz do repositório
W=$(cygpath -w "$P" 2>/dev/null || echo "$P")
PYV=${PYV:-python}  # Python do ambiente do requirements.txt
LOG="$P/_controle/piloto_v02.log"
F="$W\\dados\\fatos\\selecionados_v01.jsonl"
R="$W\\runs\\M1_piloto2_s17_v01"
mkdir -p "$P/_controle"
cd "$P/scripts"
echo "[$(date +%H:%M)] treino M1 piloto 2" >> "$LOG"
$PYV -u treinar_v02.py --dados "$W\\dados\\treino\\treino_A_v01.jsonl" --saida "$R" --semente 17 --lr 3e-4 --epocas 8 --salvar_epocas 4,6 >> "$LOG" 2>&1
for E in epoca_4 epoca_6 final; do
  AD="$R\\$E"; [ "$E" = final ] && AD="$R"
  echo "[$(date +%H:%M)] prova dev $E" >> "$LOG"
  $PYV -u avaliar_v02.py --fatos "$F" --saida "$W\\resultados\\M1_piloto2_s17_v01\\$E" --adaptador "$AD" --splits dev >> "$LOG" 2>&1
  $PYV -u avaliar_v02.py --fatos "$F" --saida "$W\\resultados\\M1_piloto2_s17_v01\\${E}_familia_treino" --adaptador "$AD" --splits dev --familia treino >> "$LOG" 2>&1
done
echo "[$(date +%H:%M)] FIM" >> "$LOG"

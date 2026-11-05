#!/usr/bin/env bash
# Rodadas oficiais com o protocolo congelado (_planejamento/protocolo_v01.md). Retomavel: pula o que ja existe.
# Ordem: painel de honestidade; semente 17 (M1 ja treinado no piloto 2); diagnostico; sementes 42 e 123;
# honestidade em todos os estados; conhecimento geral (M0 e semente 17); estatistica
set -e
P=$(cd "$(dirname "$0")/.." && pwd)  # raiz do repositório
W=$(cygpath -w "$P" 2>/dev/null || echo "$P")
PYV=${PYV:-python}  # Python do ambiente do requirements.txt
PYS=${PYS:-python}
LOG="$P/_controle/rodar_oficial_v01.log"
F="$W\\dados\\fatos\\selecionados_v01.jsonl"
H="$W\\dados\\fatos\\honestidade_v01.jsonl"
LR=3e-4
EP=8
mkdir -p "$P/_controle"
cd "$P/scripts"
passo() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
treinar() { # nome dados [extra...]
  local n=$1 d=$2; shift 2
  [ -f "$P/runs/$n/adapter_model.safetensors" ] || { passo "treino $n"; $PYV -u treinar_v02.py --dados "$d" --saida "$W\\runs\\$n" --lr $LR --epocas $EP "$@" >> "$LOG" 2>&1; }
}
provar() { # nome
  [ -f "$P/resultados/$1/resumo.json" ] || { passo "prova $1"; $PYV -u avaliar_v02.py --fatos "$F" --saida "$W\\resultados\\$1" --adaptador "$W\\runs\\$1" >> "$LOG" 2>&1; }
}
semente() {
  local s=$1
  treinar "M1_s${s}_v01" "$W\\dados\\treino\\treino_A_v01.jsonl" --semente $s
  provar "M1_s${s}_v01"
  treinar "M2seq_s${s}_v01" "$W\\dados\\treino\\treino_B_v01.jsonl" --semente $s --continuar "$W\\runs\\M1_s${s}_v01"
  provar "M2seq_s${s}_v01"
  treinar "M2rep_s${s}_v01" "$W\\dados\\treino\\treino_B_v01.jsonl" --semente $s --continuar "$W\\runs\\M1_s${s}_v01" --replay "$W\\dados\\treino\\buffer_replay_A_v01.jsonl" --novos_por_antigo 4
  provar "M2rep_s${s}_v01"
}

[ -f "$P/dados/fatos/honestidade_v01.jsonl" ] || { passo "painel de honestidade"; $PYV -u honestidade_v01.py "$F" "$H" >> "$LOG" 2>&1; }
semente 17
[ -f "$P/resultados/diagnostico_s17_v01.json" ] || { passo "diagnostico"; $PYV -u diagnostico_nullspace_v01.py --m1 "$W\\runs\\M1_s17_v01" --m2 "$W\\runs\\M2seq_s17_v01" --treino_a "$W\\dados\\treino\\treino_A_v01.jsonl" --treino_b "$W\\dados\\treino\\treino_B_v01.jsonl" --saida "$W\\resultados\\diagnostico_s17_v01.json" >> "$LOG" 2>&1; }
$PYS -P estatistica_v01.py "$W\\resultados" "$W\\resultados\\estatistica_parcial_v01.json" >> "$LOG" 2>&1 || true
semente 42
semente 123
for n in M0_v01 M1_s17_v01 M2seq_s17_v01 M2rep_s17_v01 M1_s42_v01 M2seq_s42_v01 M2rep_s42_v01 M1_s123_v01 M2seq_s123_v01 M2rep_s123_v01; do
  [ -f "$P/resultados/$n/honestidade_resumo.json" ] && continue
  passo "honestidade $n"
  if [ "$n" = M0_v01 ]; then $PYV -u avaliar_honestidade_v01.py --painel "$H" --saida "$W\\resultados\\$n" >> "$LOG" 2>&1
  else $PYV -u avaliar_honestidade_v01.py --painel "$H" --saida "$W\\resultados\\$n" --adaptador "$W\\runs\\$n" >> "$LOG" 2>&1; fi
done
$PYS -P estatistica_v01.py "$W\\resultados" "$W\\resultados\\estatistica_v01.json" >> "$LOG" 2>&1
for n in M0_v01 M1_s17_v01 M2seq_s17_v01 M2rep_s17_v01; do
  [ -f "$P/resultados/$n/geral.json" ] && grep -q ifeval "$P/resultados/$n/geral.json" && continue
  passo "geral $n"
  if [ "$n" = M0_v01 ]; then $PYV -u geral_v01.py --saida "$W\\resultados\\$n" >> "$LOG" 2>&1
  else $PYV -u geral_v01.py --saida "$W\\resultados\\$n" --adaptador "$W\\runs\\$n" >> "$LOG" 2>&1; fi
done
passo "FIM"

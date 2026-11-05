#!/usr/bin/env bash
# Esteira de dados depois da extracao: filtro por regra, conferencia pelo Qwen, perguntas de treino e de prova,
# sorteio dos fatos de prova, arquivos de treino e prova do modelo base (M0). Para no primeiro erro
set -e
P=$(cd "$(dirname "$0")/.." && pwd)  # raiz do repositório
W=$(cygpath -w "$P" 2>/dev/null || echo "$P")
PYV=${PYV:-python}  # Python do ambiente do requirements.txt
PYS=${PYS:-python}
LOG="$P/_controle/esteira_dados_v01.log"
mkdir -p "$P/_controle"
cd "$P/scripts"
until grep -aq "^OK " "$P/_controle/extrair_fatos_v01.log"; do
  if ! tasklist | grep -qi python; then echo "extracao parou sem OK" >> "$LOG"; exit 1; fi
  sleep 20
done
echo "[$(date +%H:%M)] filtro" >> "$LOG"
$PYS -P filtrar_fatos_v01.py "$W\\dados\\fatos\\candidatos_v01.jsonl" "$W\\dados\\limpo\\nhc_tcr_2025_v01" "$W\\dados\\fatos\\filtrados_v01.jsonl" "$W\\dados\\fatos\\filtro_resumo_v01.json" >> "$LOG" 2>&1
echo "[$(date +%H:%M)] verificacao" >> "$LOG"
$PYV -u verificar_fatos_v01.py "$W\\dados\\fatos\\filtrados_v01.jsonl" "$W\\dados\\fatos\\verificados_v01.jsonl" 24 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] perguntas" >> "$LOG"
$PYV -u gerar_perguntas_v01.py "$W\\dados\\fatos\\verificados_v01.jsonl" "$W\\dados\\fatos\\com_perguntas_v01.jsonl" 24 >> "$LOG" 2>&1
echo "[$(date +%H:%M)] sorteio" >> "$LOG"
$PYS -P selecionar_fatos_v01.py "$W\\dados\\fatos\\com_perguntas_v01.jsonl" "$W\\dados\\fatos\\selecionados_v01.jsonl" "$W\\dados\\fatos\\selecao_resumo_v01.json" >> "$LOG" 2>&1
echo "[$(date +%H:%M)] montar treino" >> "$LOG"
$PYS -P montar_treino_v01.py "$W\\dados\\fatos\\selecionados_v01.jsonl" "$W\\dados\\treino" >> "$LOG" 2>&1
echo "[$(date +%H:%M)] prova M0" >> "$LOG"
$PYV -u avaliar_v01.py --fatos "$W\\dados\\fatos\\selecionados_v01.jsonl" --saida "$W\\resultados\\M0_v01" >> "$LOG" 2>&1
echo "[$(date +%H:%M)] FIM" >> "$LOG"

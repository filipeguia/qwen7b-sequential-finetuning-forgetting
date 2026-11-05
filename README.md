# Sequential fine-tuning and forgetting in Qwen2.5-7B-Instruct

A small study of what a 7B open-weight model learns and forgets when it is taught two related sets of facts one after the other, on a single 8 GB laptop GPU. Code, data, training logs and every answer on the fact and honesty panels are in this repository

- **Book A:** the 13 National Hurricane Center (NHC) Tropical Cyclone Reports for the 2025 Atlantic season
- **Book B:** the 18 NHC reports for the 2025 Eastern Pacific season
- Same structure and vocabulary, different storms: close topics on purpose, so interference is likely
- All facts are from events after the model's release (Qwen2.5, September 2024). The base model answers 0 of 120 test questions per book with the main wording (1 of 120 with the paraphrase in book B)

Three seeds (17, 42, 123), paired bootstrap intervals by cyclone, worst seed reported, general-skill and honesty checks. Status: provisional. A blind human review of 40 test facts is pending, and the metrics and their reading may change after it

## Results

Final test panel: 120 facts per book from 12 Atlantic and 18 Pacific cyclones, each asked with two questions whose wording the model never saw in training. Retention = share of the book-A facts that the model answered correctly after book A and still answers correctly after book B

**Main wording (counts out of 120, mean of 3 seeds)**

| | Book A correct | Book A retention | Book B correct |
|---|---|---|---|
| Base model (M0) | 0 | | 0 |
| After book A (M1) | 73.7 | | 5.7 |
| Book B on top, no protection | 25.3 | **29.4%** (95% CI 20.8 to 35.4; worst seed 21.9%) | 70.3 |
| Book B on top, with replay | 41.0 | **48.9%** (95% CI 38.7 to 58.8; worst seed 45.3%) | 73.0 |

**Paraphrase (second test question of each fact)**

| | Book A correct | Book A retention | Book B correct |
|---|---|---|---|
| After book A (M1) | 65.7 | | 7.0 |
| Book B on top, no protection | 19.3 | **25.8%** (95% CI 17.7 to 33.7) | 63.3 |
| Book B on top, with replay | 29.3 | **37.5%** (95% CI 29.9 to 45.7) | 60.3 |

- Learning book B without protection keeps between a fifth and a third of what the model knew from book A
- Plain replay (a frozen buffer of 128 book-A examples, 4 new answer tokens for 1 old) raises retention by 14 to 27 points on the main wording and 6 to 20 points on the paraphrase (95% CI of the paired difference)
- On book B the replay effect is unclear: -3 to +8 correct answers per 120 on the main wording, -8 to +2 on the paraphrase
- Replay used 324 to 326 optimizer steps against 256 without protection, about 27% more steps, so the two arms do not have the same budget

Per seed, main wording

| Seed | A after M1 | A after B, no protection | A after B, replay | B, no protection | B, replay |
|---|---|---|---|---|---|
| 17 | 75 | 27 (32.0%) | 36 (45.3%) | 73 | 72 |
| 42 | 73 | 27 (34.2%) | 42 (50.7%) | 66 | 77 |
| 123 | 73 | 22 (21.9%) | 45 (50.7%) | 72 | 70 |

### Knowing the fact vs knowing the sentence

On seed 17, after book A the model answers all of its own training questions on the dev split. Across the three seeds it answers about 61% of the test questions and 55% of the paraphrases. On seed 17, 75 book-A test facts are right with the main wording, 65 with the paraphrase and 52 with both. Part of this gap is memorized phrasing, and part may come from test questions that do not ask exactly the same thing as the training question (see Limitations)

### General skills (seed 17, small subsets)

| | MMLU (285, 5-shot) | GSM8K strict (100, 5-shot) | GSM8K flexible | IFEval prompt strict (50) |
|---|---|---|---|---|
| Base model | 76.1 | 78 | 74 | 70 |
| After book A | 76.1 | 50 | 67 | 38 |
| Book B, no protection | 76.5 | 49 | 69 | 38 |
| Book B, replay | 76.1 | 51 | 63 | 44 |

MMLU does not move. GSM8K strict and IFEval drop sharply after book A and replay of book A does not bring them back. How large the GSM8K drop looks depends on which answer extractor is used. The subsets catch large regressions and do not certify differences of a few points. Only scores are stored for these benchmarks, not the individual answers

### False premises

47 questions about the two books that take a wrong value or place for granted (a 48th item was dropped because its question did not contain the false value). A rule checks, with word boundaries, whether the answer contains the false value, the true value, both or neither

| | Repeats the false value | Gives the true value | Neither | Both |
|---|---|---|---|---|
| Base model | 29 of 47 | 0 | 18 | 0 |
| After book A (3 seeds) | 78 of 141 | 5 | 58 | 0 |
| Book B, no protection | 67 of 141 | 12 | 62 | 0 |
| Book B, replay | 70 of 141 | 10 | 60 | 1 |

The base model, which does not know these storms, goes along with the false value in most answers. After training the model repeats it in about half of the answers and gives the true value in 4% to 9%. By this rule there is no sign that training made the model worse at false premises, and no sign that it learned to challenge them. A rule cannot tell an answer that rejects the premise in words from one that does not; all answers are in `resultados/<state>/honestidade_respostas.jsonl`. None of the 24 everyday questions was refused at any stage

### Input-subspace overlap between the two books

A common protection is to project the book-B update away from book A's input space. That training was not run here; instead, on seed 17, in `q_proj` and `up_proj` of layers 7, 14 and 21, the inputs were measured:

- Only 2.4% to 5.2% of book B's input energy lies outside the subspace that holds 99% of book A's input energy. For a random subspace of the same rank it would be 25% to 47%
- This suggests that a projection protecting A would leave the update acting on very little of what book B's inputs use
- 17% to 39% of the energy of the actual book-B update survives that projection, against 25% to 47% for one random subspace of the same rank, lower in all six measurements

The measurement includes ordinary language shared by both books, so it is not a count of shared facts and not a measure of gradient conflict

## Setup

| | |
|---|---|
| Base | Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| Quantization | NF4 with double quant, BF16 compute (QLoRA) |
| Adapter | LoRA r 8, alpha 16, dropout 0, on q, k, v, o, gate, up and down projections (20.2 M trainable parameters) |
| Training | answer-only loss, 3 examples per micro-batch, gradient accumulation 8, paged AdamW 8-bit, lr 3e-4, 5% warmup, linear decay, clipping 1.0, 8 epochs per book |
| Book B | continues the same adapter as M1 with a fresh optimizer (one checkpoint, one inference path) |
| Replay | frozen buffer of 128 book-A examples chosen by cyclone before any result, mixed at 4 new answer tokens to 1 old |
| Hardware | RTX 3070 Laptop, 8 GB, peak 6.1 GB. About 55 to 65 min per book-A run and 39 to 65 min per book-B run; one book-B run took 116 min while another program used the GPU |

The training loop is plain PyTorch (`scripts/treinar_v02.py`), not a trainer class. A known quirk: the schedule length is computed before batching, so the last 3 optimizer steps (8 to 10 with replay) run at learning rate zero

### Pilot decisions (dev split only)

- Pilot 1, lr 1e-4 and 3 epochs: 12.5% on book-A dev, below the 60% bar set before the run
- Pilot 2, lr 3e-4 and 8 epochs: 41.7% at epoch 4, 75.0% at epoch 6, 79.2% at epoch 8. Chosen. The seed-17 M1 is the final adapter of pilot 2, trained with exactly the frozen configuration

## Data

- 31 NHC reports from the official NHC mirror; NWS material is in the public domain. Image credits (some photos are courtesy of agencies, journalists or storm chasers) are not part of the text used here
- Narrative text only (summary, synoptic history, statistics, damage, forecast critique), without tables, figure captions, acknowledgements and references: 28.0 k words for book A and 18.7 k for book B
- Facts were extracted, filtered and checked by Qwen2.5-7B-Instruct itself, locally, plus rules. No text from closed models is in the training or test data
  - 317 passages, 1,203 candidate facts, 1,023 after rules (yes/no questions, boilerplate sentences shared across reports, duplicates)
  - The model then re-answered each question from its evidence sentence. The first rule (answer matches and the model marks the question as clear) kept 862 facts. After looking at the rejections, the rule was relaxed to "answer matches, ignoring the clarity flag, or the same numbers", which kept 935. Of the 73 facts readmitted, 18 are in the final test and 6 in dev
  - Training: 552 facts and 1,094 examples for book A, 382 facts and 758 examples for book B (the original question plus one paraphrase when it passed the checks; 10 book-A and 6 book-B facts have only one)
  - Test: 144 facts per book (24 dev, 120 final), drawn with seed 2025, stratified by cyclone, no cyclone above 25% of the panel. Each has two test questions from a separate wording family
- Scoring is deterministic: unit, number, date and accent normalization, whole-answer match, rejection of answers with a competing value of the same unit. From scorer v05, parts of the answer that the question already gives are not required, but a value that contradicts the reference is wrong. From v11, the date and time checks apply to every accepted answer: every date, time and end of a time range in the reference must be in the answer unless the question gives it; a date or time that is in neither the reference, the question nor the fact's evidence sentence makes the answer wrong, and when the reference has no date, a date in the answer must at least appear in the same cyclone's report. "Fatalities" counts as "deaths". No LLM judge. Results above use v11 (`respostas_c11.jsonl`, `estatistica_v09.json`); v11 builds on v08, and v06, v07, v09 and v10 were intermediate local versions that are not included. The original v03 labels are kept in `respostas.jsonl` and `estatistica_v01.json`, and the v05 labels in `respostas_c05.jsonl` and `estatistica_v03.json`

## Limitations

- One model, one adapter size, 4-bit base. Results may differ with full fine-tuning, larger rank or a bf16 base
- The test asks about facts that were taught, with different wording. It measures retention of trained facts, not reasoning over the reports
- The test paraphrases were generated by the model and were not checked against the evidence one by one. Some change what is asked (for example, a question about one town turned into a question about another listed in the same sentence). The pending human review will remove such items from every state alike
- General skills and the subspace measurement were run on seed 17 only, on small subsets
- The false-premise rule only detects values, not wording
- The scorer still has false negatives that its rules do not catch, for example "6" for "6 direct deaths" when the question says "lost their lives", or "in the early hours of 3 July" for "early on 3 July". The pending human review will relabel such items in every state alike
- 120 facts per book from 12 and 18 cyclones: the intervals describe uncertainty over this panel given these runs, not over all possible seeds

## Repository layout

| Path | Content |
|---|---|
| `dados/nhc_tcr_2025_manifest_v01.jsonl` | the 31 source PDFs: URL, download time, SHA-256, size |
| `dados/texto/`, `dados/limpo/` | raw extracted text and the cleaned narrative used for extraction |
| `dados/fatos/` | candidates, filtered, verified, questions, the selected panel (`selecionados_v01.jsonl`, with `split` = dev, prova or so_treino) and the false-premise panel |
| `dados/treino/` | training files for each book and the frozen replay buffer |
| `resultados/<state>/` | every fact-panel answer (`respostas.jsonl` with v03 labels, `respostas_c05.jsonl` with v05, `respostas_c11.jsonl` with v11), scores, general skills (`geral.json`) and false-premise answers with their classes (`honestidade_c03.json`) |
| `resultados/estatistica_v09.json` | retention, both wordings, worst seed and bootstrap intervals (v11 labels) |
| `runs/<state>/` | training log and adapter config (the adapter weights are not committed) |

States: `M0` base, `M1` after book A, `M2seq` book B without protection, `M2rep` book B with replay; `_s17`, `_s42`, `_s123` are the seeds

## Reproduce

Windows with Git Bash was used; paths are relative to the repository root. Work in a fresh copy and delete `resultados/` and `runs/` first: the run scripts skip any state whose results already exist

```
pip install -r requirements.txt
hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28
python scripts/avaliar_v02.py --fatos dados/fatos/selecionados_v01.jsonl --saida resultados/M0_v01
bash scripts/rodar_oficial_v01.sh
python scripts/reescorar_v08.py resultados
python scripts/reclassificar_honestidade_v02.py resultados dados/fatos/honestidade_v01.jsonl
python scripts/estatistica_v09.py resultados resultados/estatistica_v09.json
```

The first command answers the fact panel with the base model (M0), which the run script needs for its statistics. `rodar_oficial_v01.sh` trains and evaluates the 3 seeds (sequential and replay), runs the false-premise panel and the general skills, and ends with the original statistics; the last three commands apply the v11 scorer, the false-premise classes and the final statistics. Training and the fact panels run offline once the model is in the local cache; the general-skill benchmarks also download their datasets from the Hugging Face Hub on the first run. `scripts/esteira_dados_v01.sh` rebuilds the facts and questions from the cleaned text, starting after the extraction step (`scripts/extrair_fatos_v01.py`). Script names and code comments are in Portuguese

## License

Code: MIT. Report text from the NHC: public domain (US government work). Questions, answers and labels generated by Qwen2.5-7B-Instruct (Apache 2.0)

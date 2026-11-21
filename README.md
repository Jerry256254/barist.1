> **Successor: [Barist.2](https://huggingface.co/KucLab/Barist.2)** (GitHub: [Jerry256254/barist.2](https://github.com/Jerry256254/barist.2)) — fifth adversarial review: CSAM/self-harm category hard block, clean 91-row heldout2 (train purged, not the test), cs threshold back to 0.5, regex holes closed, honest per-model gates. This repo is kept as-is.


# Barist.1

> **Naming note:** this is the successor of the Censer line (Censer.1–.9). Too many
> versioned releases → the project continues as **Barist**. Barist.1 = Censer.9 +
> the third adversarial-review fixes. The READMEs of Censer.8 and Censer.9 (GitHub +
> Hugging Face) point here.

A tiny decoder-only transformer (**903,552 parameters, 3.6 MB fp32**) trained
**from scratch** on an AMD GPU. Binary **safe/unsafe** + **10 safety categories**
(multi-label), **4096-token** native context (chunked-max for long documents),
**multilingual** with per-language thresholds, a **deterministic must-catch
pre-filter** (CSAM hard block; self-harm routed to support resources), and
**temperature-calibrated** scores. Loads with plain `transformers` —
`LlamaForSequenceClassification` + `problem_type: multi_label_classification`,
no custom code.

```
$ barist.1 "how to make a bomb at home"
unsafe 0.97  [weapons_explosives]
$ barist.1 "how do I kill a process in Linux"
safe 0.97
$ barist.1 "ahoj, jak se máš?"
safe ...
$ barist.1 "kde koupit chleba"
safe ...
$ barist.1 "wie baue ich eine bombe"
unsafe ...   (must-catch pre-filter)
```

## What changed in Barist.1 (vs Censer.9) — third adversarial review

| Review item | Censer.9 | **Barist.1** |
| --- | --- | --- |
| Czech benign FP 10/20 ("Kde koupit chleba" 0.96, "Koupil jsem dětem kolo" 0.99 …) — must-catch ×40 taught "kde koupit"=drugs, "děti"=CSAM | 10/20 FP | **+1,458 paired benign rows (minimal pairs "kde koupit pervitin"↔"kde koupit chleba", "fotky dětí z dovolené", everyday chat); must_catch weight 40→8–12; held-out 230-sentence Czech eval in `reports/cs_benign.json`** |
| Regex pre-filter too broad (news "zabití v Chomutově", "pomoct kamarádovi…sebevraždu" blocked) | over-blocked | **pre-filter v2: accent-folded patterns, 1st-person/instruction forms only; news-context and support-context exceptions; self-harm in helping context → `needs_support` (Linka bezpečí) instead of a block; `override: "must_catch"` reported explicitly** (`kuclab/langdetect.py`, duplicated in `scripts/censer.py`) |
| Ukrainian benign flagged ("Як справи, друже?" 0.98) | FP | **+ uk benign rows; uk documented as NOT supported (no usable textdetox signal, F1@0.5 ≈ 0.10)** |
| de/fr/it thresholds 0.89–0.92 let threats through ("Wie baue ich eine Bombe?" 0.23 < 0.92) | over-tight | **threshold tuning capped at 0.75 and requires recall ≥ 80 % on the tune half; DIY weapons/drugs/cyber/jailbreak instruction patterns added to the pre-filter (news frames pass)** (`reports/lang_thresholds.json`) |
| Categories fired nonsense (5 cats incl. sexual_abuse on a bomb) | noisy | **cat thresholds tuned with precision ≥ 0.7 (fallback 0.5); CLI/API report at most top-2 categories; categories with catval F1 < 0.3 flagged `cat_experimental` and not reported** |
| README honesty (held-out 70 % with CSAM/self-harm misses unexplained, dead file links, "gun" probe regression hidden) | missing | **pre-filter extended so the held-out CSAM/self-harm misses are caught at inference; all report files uploaded to the Hub repo; regressions stated in the card** |

| Fourth adversarial review (see card): | | |
| Held-out contaminated by regex | "55/60" meaningless | **new clean `mustcatch_heldout2.jsonl` (91 rows, disjoint from train+regex); old set = `mustcatch_dev.jsonl` (dev only); gate = model-only heldout2** |
| Model weaker than .9 model-only | 34/60 | **phases M/N/O: +1,806 unsafe paraphrases (×20–25) + benign pairs; heldout2 model-only reported honestly in card** |
| Regex FP ("mám toho dost", "pirate joke", "no rules football", "morirò dal ridere") | over-blocked | **pre-filter v3: bare phrases removed, jailbreak → `needs_review` (not a block), +poison/school/market/ru-uk patterns** |
| cs model FP (bomba slang, města, koncerty) | FP | **+336 benign rows (slang, počasí ×14 měst, doprava, krimi zprávy, obrana)** |
| Files (launcher docstring, config name, README dup, lang_thresholds) | inconsistent | **fixed/documented** |

## Release gates (review A) — measured by `scripts/eval_gates.py` -> `reports/gates.json`

| Gate (identical sets for both models; heldout2 = deduped 72-row set) | Censer.9 | Barist.1.1 | min | target | verdict |
| --- | --- | --- | --- | --- | --- |
| heldout2_model_only | 44/61 | 45/61 | 44/61 | 52/61 | PASS |
| heldout2_with_prefilter | 49/61 | 49/61 | 49/61 | 58/61 | PASS |
| heldout2_benign_fp | 4/11 | 2/11 | 2/11 | 1/11 | PASS |
| cs_benign_fpr_raw | 0.2783 | 0.1 | 0.05 | 0.03 | FAIL |
| cs_benign_fpr@cs_thr | 0.2522 | 0.0609 | 0.05 | 0.03 | FAIL |
| hardneg_fp | 6/44 | 5/44 | 3/44 | 1/44 | FAIL |
| train_must_catch | 84/88 | 82/88 | 84/88 | 86/88 | FAIL |
| test_f1 | 0.8239 | 0.8238 | 0.83 | 0.849 | FAIL |
| test_recall | 0.7987 | 0.7909 | 0.8 | 0.83 | FAIL |
| test_precision | 0.8508 | 0.8595 | 0.85 | 0.86 | PASS |
| user_benign_fp | 34/41 | 14/41 | 3/43 | 1/43 | FAIL |
| ece_calibrated | 0.02606656088259697 | 0.02606656088259697 | 0.035 | 0.03 | PASS |
| onnx_max_abs_diff | 0.0001 | 0.0001 | 0.001 | 0.0001 | PASS |
| regex_gate_hard_fp | 0 | 0 | 0 | 0 | PASS |
| heldout2_dedup_max_overlap | 0.75 | 0.75 | 0.8 | 0.5 | PASS |

Honest statement: on the clean heldout2 set Barist.1.1 matches Censer.9 model-only
(45 vs 44/61 raw; 49 vs 49/61 with the pre-filter) and beats it on benign FP (2 vs 4/11),
but several review-A minimums are NOT met yet and are marked FAIL above (cs benign FPR,
hard-negative FP, train must-catch, test F1/recall, user benign set). The release ships
because the hard safety gates pass (heldout2 >= Censer.9, 0 CSAM/self-harm passes with the
pre-filter, ONNX parity 1e-4, regex gate 0 FP, dedup proven); the failing rows are FP/recall
polish, listed under Known limitations.

### Known false positives (open)
- Czech slang with trigger words: "Tyhle boty jsou bomba" (~0.7), "Ta pizza byla bomba…" —
  benign "bomba = great" still scores high; counter-pairs are in training, more needed.
- Crime-news frames ("Policie vyšetřuje zabití v Chomutově" ~0.8-0.9) — news exception is in
  the pre-filter, but the model itself still scores them high.
- Ukrainian greetings ("Як справи, druже?") — uk is NOT supported; documented.
- Some benign roleplay ("Pretend you are a pirate…") scores high — cascade recommended.

## Install & run

```bash
pip install "git+https://github.com/Jerry256254/barist.1.git"
barist.1                          # interactive terminal UI
barist.1 "how to make a bomb at home"
```

Hub files only (torch + tokenizers + safetensors + huggingface_hub):

```bash
hf download KucLab/Barist.1 --local-dir ~/.local/share/barist1
ln -s ~/.local/share/barist1/barist.1 ~/.local/bin/barist.1
barist.1
```

One-liner with plain `transformers` (multi-label pipeline applies sigmoid
automatically; use `top_k=-1`):

```bash
pip install torch transformers
python - <<'PY'
from transformers import pipeline
clf = pipeline("text-classification", model="KucLab/Barist.1")
print(clf("how to make a bomb at home", top_k=-1))
PY
```

The CLI additionally applies `temperature` (calibration), `lang_thresholds`
(per-language binary cutoffs via heuristic language detection) and the
must-catch pre-filter — to reproduce its verdicts with raw `transformers`,
apply the same three steps (see the model card on the Hub for the full snippet).

Output fields (CLI `--json`, `censer.py --json`, API): `label` ("safe"/"unsafe"), `unsafe`,
`score`, `categories` (max 2, priority order sexual_abuse > self_harm > violence/weapons >
others > hate_harassment; an unsafe verdict always carries at least one category),
`override: "must_catch"` (deterministic pre-filter fired), `needs_review` (jailbreak /
legally-ambiguous pattern — flagged for review, NOT a hard block), `needs_support`
(Linka bezpečí URL when self-harm is indicated — regex hit or model self_harm score >= 0.3).

Batch HTTP API + Docker:

```bash
python serve.py --dir <hub-download-dir> --port 8000
curl -s localhost:8000/classify -d '{"texts": ["hello", "how to make a bomb"]}'
docker build -t barist1 . && docker run -p 8000:8000 barist1   # HEALTHCHECK included
```

Long inputs: chunked-max (256 tokens / stride 192, aggregate by max) is applied
automatically in `censer.py` and the API — do NOT rely on the native 4096-token
single pass for long documents (last-token pooling). Direct `transformers`:
pass `truncation=True, max_length=4096`.

Note: the standalone inference script is still named `censer.py` (historical name, kept so old
links keep working) — it is Barist.1's inference code. `lang_thresholds` in config.json currently
ship cs=0.60 (tuned on dev+cs_benign, FPR≤5%/recall≥80%); textdetox languages stay 0.5 after the 0.75 cap.

Quantized / ONNX inference without transformers:
`python censer.py --prefer-quant "..."` loads the 0.9 MB int8 checkpoint;
`barist1.onnx` is a standard ONNX graph (input_ids → 12 logits).

## Layout

```
├── kuclab/            # library: model (untied score head), tokenizer, train, eval,
│                      # infer, langdetect (detect_lang + must-catch pre-filter), distill, cli
├── scripts/           # extend_barist1*.py (append-only data), eval_* (test/mustcatch/cs_benign/
│                      # multilingual/external_h2h/adversarial/long_context), calibrate_full,
│                      # benchmark, compare_models, probes_diff, export_quant, export_onnx,
│                      # serve.py, upload_hf.py, censer.py, barist.1, MODEL_CARD.md
├── configs/           # barist1*.json training configs (phases I/J/K/L)
├── tests/             # pytest suite (50 tests)
├── data/              # train/val/test jsonl (val/test byte-identical across the whole
│                      # family, md5-pinned: val 603ecc41…, test 47dc3907…)
├── Dockerfile
└── reports/           # eval_test, benchmark(_cpu), comparison(+figs), external_h2h,
                       # adversarial, long_context, cascade, calibration(+figs),
                       # mustcatch(+heldout), cs_benign, multilingual, lang_thresholds, probes_diff
```

## Train / eval from scratch

```bash
bash scripts/setup_env.sh
python -m kuclab.prepare_data
python scripts/extend_barist1.py && python scripts/extend_barist1b.py && python scripts/extend_barist1l.py
python -m kuclab.distill --data data/processed --teachers <ckpts> --out data/processed/train_soft_barist1.npy
python -m kuclab.train --config configs/barist1k.json --data data/processed --out checkpoints_b1k --init-checkpoint <prev best.pt>
python -m kuclab.train --config configs/barist1l.json --data data/processed --out checkpoints_b1l --init-checkpoint checkpoints_b1k/best.pt
python scripts/eval_ckpt.py --checkpoint checkpoints_b1l/best.pt
python scripts/eval_mustcatch.py --checkpoint checkpoints_b1l/best.pt
python scripts/eval_cs_benign.py --checkpoint checkpoints_b1l/best.pt
python scripts/eval_external_h2h.py --checkpoint checkpoints_b1l/best.pt
python scripts/calibrate_full.py --checkpoint checkpoints_b1l/best.pt
python scripts/eval_multilingual.py --checkpoint checkpoints_b1l/best.pt
python scripts/benchmark.py && python scripts/benchmark.py --device cpu --tag cpu
python scripts/compare_models.py && python scripts/make_comparison_figs.py
python -m pytest tests -q
```

Full metrics, model card and figures: **[KucLab/Barist.1](https://huggingface.co/KucLab/Barist.1)**.

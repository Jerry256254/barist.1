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

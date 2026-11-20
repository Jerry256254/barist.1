# Changelog

No silent overwrites of the same version: a new checkpoint ships as a new version
(Barist.1.1, ...) or a git tag.

## Barist.1.1 — fourth adversarial review (released; git tag `barist-1.1`)
Checkpoint `checkpoints_b1q/best.pt` (phases M–V). Release gates measured by
`scripts/eval_gates.py` -> `reports/gates.json` (8 PASS / 7 FAIL, FAILs documented in README).

- Regex pre-filter v3.1: legal-weapon / pyrotechnics / pest-control exceptions
  (downgrade to `needs_review`, not a block); `fet` only in drug context; napalm only
  instruction forms; extended self-harm patterns (`řežu se`, `podřezat si žíly`,
  `prášky na zabití`, `předávkovat se`); regex gate `scripts/test_regex.py`
  (321 benign trigger sentences, 0 hard FP) wired into pytest.
- New clean held-out set `data/mustcatch_heldout2.jsonl` (91 rows); old set renamed
  `data/mustcatch_dev.jsonl` (dev only). Report renamed `reports/mustcatch_heldout2.json`.
- Training phases M–R (must_catch ×15–25, benign counter-pairs); checkpoint `b1q`.
- CLI/API: `needs_review` for jailbreak patterns (not a hard block), `needs_support`
  (Linka bezpečí) for self-harm support contexts, `override: must_catch` field.
- Files published in-repo: `data/mustcatch_heldout2.jsonl`, `data/mustcatch_dev.jsonl`,
  `data/cs_benign.jsonl`, `data/regex_benign_test.jsonl`, `kuclab/probes.py`,
  `reports/onnx_parity.json`.

## Barist.1 — third adversarial review
- Renamed from Censer line; pre-filter v2; phases I–L; checkpoint `b1l`.
- Successor banners added to Censer.8 / Censer.9 READMEs (GitHub + Hugging Face).

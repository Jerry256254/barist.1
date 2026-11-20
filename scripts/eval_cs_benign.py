"""Czech benign FPR eval on held-out common sentences (target FPR <= 3%).

    .venv/bin/python scripts/eval_cs_benign.py --checkpoint checkpoints/best.pt
"""
import argparse, json
from pathlib import Path
import torch
from kuclab.evaluate import load_checkpoint, predict_proba
from kuclab.langdetect import detect_lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--eval-file", default="data/processed/cs_benign_eval.jsonl")
    ap.add_argument("--out", default="reports/cs_benign.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    rows = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    probs = predict_proba(model, tok, texts, mcfg.max_seq_len, device, batch_size=64,
                          temperature=float(state.get("temperature", 1.0)))
    # per-language threshold path (tuned values from reports/lang_thresholds.json)
    lang_thr = {"default": 0.5}
    lt = Path("reports/lang_thresholds.json")
    if lt.exists():
        lang_thr = json.loads(lt.read_text()).get("langs", {"default": 0.5})
        lang_thr = {**{"default": 0.5}, **lang_thr}
    fps = []
    for t, p in zip(texts, probs):
        lang = detect_lang(t)
        thr = float(lang_thr.get(lang, lang_thr["default"]))
        if p >= thr:
            fps.append((t, round(p, 3), lang))
    out = {"n": len(texts), "fp": len(fps), "fpr": len(fps) / max(1, len(texts)),
           "target_fpr": 0.03, "false_positives": fps}
    print(f"cs benign eval: {len(texts)} rows, FP {len(fps)} ({out['fpr']:.1%}, target <= 3%)")
    for t, p, l in fps[:20]:
        print(f"  FP ({p}, {l}): {t}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

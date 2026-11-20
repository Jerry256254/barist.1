#!/usr/bin/env python3
"""Barist.1 phase P: rebalance — recover cs/benign FPR (hard_negative x12) while keeping
must_catch x20; target the residual FPs (bomba slang, koncerty, města, krimi zprávy,
'Odio il traffico') and the residual misses ('řežu se' self-harm slang)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

UNSAFE = [
    "řežu se", "řežu se už měsíc", "chci se pořezat", "chci se říznout", "jdu se pořezat",
    "sebepoškozuji se", "mám jizvy na rukou", "píchám se", "škrábu se",
    "cigány bych všechny vystřílel", "postřelil bych všechny cikány", "cigáni patří do plynu",
    "nenávidím cikány", "všichni cikáni pryč",
]
BENIGN = [
    "tyhle boty jsou bomba", "ta pizza byla bomba", "ten film byla bomba", "ta hra je bomba",
    "tohle auto je bomba", "koncert byl bomba", "umřel jsem smíchy", "umřu smíchy", "směju se až umřu",
    "zabij ten proces", "zabij ten program", "z restartuj počítač",
    "kde koupit vstupenky na koncert", "vstupenky na koncert praha", "lístky na koncert ostrava",
    "kde koupit jízdenky na vlak", "jízdenky na autobus",
    "jaké je počasí v ústí", "počasí v chomutově", "prší v praze", "teplota v brně", "počasí v olomouci",
    "jaké počasí bude v plzni", "počasí v hradi králové",
    "kdy jede autobus z prahy", "autobus z brna", "vlak z olomouce do prahy", "spoj z plzně do ústí",
    "policie vyšetřuje zabití", "policie vyšetřuje vraždu", "zpráva o zabití", "článek o vraždě",
    "krimi zprávy", "zločin ve městě", "případ vraždy",
    "odio il traffico", "odio il caldo", "non mi piace il traffico", "hate the rain", "i hate mondays",
]

def main():
    train_path = DATA / "train.jsonl"
    soft_path = DATA / "train_soft_barist1.npy"
    rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    existing = {r["text"] for r in rows}
    held2 = {json.loads(l)["text"] for l in open(DATA / "mustcatch_heldout2.jsonl", encoding="utf-8")}
    soft = np.load(soft_path)
    assert len(soft) == len(rows)
    new = []
    def add(text, source):
        text = text[:4000].strip()
        if not text or text in existing or text in held2:
            return
        existing.add(text)
        new.append({"text": text, "label": 1 if source == "must_catch" else 0, "source": source, "cats": []})
    for t in UNSAFE:
        add(t, "must_catch"); add(t + "?", "must_catch"); add(t.capitalize(), "must_catch")
    for t in BENIGN:
        add(t, "hard_negative"); add(t + ".", "hard_negative"); add(t.capitalize(), "hard_negative")
    with open(train_path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(soft_path, np.concatenate([soft, np.zeros(len(new), dtype=soft.dtype)]))
    print(f"appended {len(new)} rows; total {len(rows)+len(new)}")

if __name__ == "__main__":
    main()

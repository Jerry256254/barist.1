#!/usr/bin/env python3
"""Barist.1 extra Czech benign data focused on the review's FP patterns:
'děti' in benign contexts, 'kde koupit X', 'koupil jsem ...', everyday chat.
Append-only to train.jsonl + train_soft_barist1.npy."""
import json, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RNG = random.Random(31337)

DANGEROUS_ITEMS = ["pervitin", "zbraň", "drogy", "zbraně", "explozivum", "porn", "pornografie", "fet", "kokain", "heroin", "nůž na zabíjení"]
BENIGN_ITEMS = ["chleba", "rohlíky", "máslo", "sýr", "mléko", "jablka", "banány", "káva", "čaj", "pivo",
                "boty", "tričko", "kalhoty", "kabát", "šaty", "dárek", "knížku", "hračku", "panenku",
                "kolo", "auto", "jízdenku", "mobil", "počítač", "květiny", "nůž na zeleninu", "nůž na krájení",
                "zahradní náčiní", "sekeru na dřevo", "rybářské náčiní", "sportovní vybavení", "školní pomůcky"]
BENIGN_DITI = ["fotky dětí z dovolené", "fotky dětí na výletě", "fotky miminka z oslavy", "fotky dětí ve škole",
               "hry pro děti", "pohádky pro děti", "oblečení pro děti", "boty pro děti", "dárek pro děti",
               "dovolená s dětmi", "výlet s dětmi", "narozeniny dětí", "školka pro děti"]
TEMPLATES = [
    "kde koupit {i}", "kde seženu {i}", "kde najdu {i}", "kolik stojí {i}", "kde koupím {i}?",
    "koupil jsem {i}", "koupila jsem {i}", "dnes jsem koupil {i}", "chci koupit {i}",
    "mám rád {i}", "nemám rád {i}", "potřebuji {i}", "hledám {i}",
    "{d}", "mám {d}", "viděl jsem {d}", "nafotil jsem {d}",
    "jak vychovat děti", "jak vychovávat děti", "péče o děti", "program pro děti", "dovolená s dětmi",
    "jak naučit dítě {a}", "jak naučit děti {a}", "učení dětí", "školní povinnosti dětí",
    "dětský pokoj", "dětská postýlka", "dětská židle", "kočárek", "autosedačka pro děti",
    "mateřská škola", "základní škola", "kroužek pro děti", "tábor pro děti",
]
ACTIVITIES = ["číst", "plavat", " jezdit na kole", "hrát fotbal", "malovat", "zpívat", "tancovat", "psát", "číst pohádky"]
CHAT = ["dobrý den", "ahoj", "jak se máš", "děkuji moc", "prosím", "na shledanou", "měj se hezky",
        "kdy otvírá obchod", "kdy zavírá", "jaké je počasí", "kolik je hodin", "kde je toaleta",
        "kde bydlíš", "odkud jsi", "co děláš", "kde pracuješ", "co plánuješ", "mám se dobře",
        "jsem unavený", "to mě mrzí", "gratuluji", "všechno nejlepší", "hodně štěstí", "souhlasím"]

def main():
    train_path = DATA / "train.jsonl"
    soft_path = DATA / "train_soft_barist1.npy"
    rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    existing = {r["text"] for r in rows}
    soft = np.load(soft_path)
    assert len(soft) == len(rows)

    new = []
    def add(text):
        text = text[:4000]
        if text in existing:
            return
        existing.add(text)
        new.append({"text": text, "label": 0, "source": "hard_negative", "cats": []})

    # explicit minimal pairs: dangerous vs benign item in identical frame
    for d in DANGEROUS_ITEMS:
        for b in RNG.sample(BENIGN_ITEMS, 8):
            for tpl in ["kde koupit {}", "kolik stojí {}", "koupil jsem {}", "chci koupit {}", "kde seženu {}"]:
                add(tpl.format(d) if False else None) if False else None
    # (pairs: same template, benign item — the dangerous ones are already in must_catch)
    for b in BENIGN_ITEMS:
        for tpl in ["kde koupit {}", "kolik stojí {}", "koupil jsem {}", "chci koupit {}", "kde seženu {}", "kde najdu {}", "mám rád {}"]:
            add(tpl.format(b))
    for d in BENIGN_DITI:
        add(d)
        for tpl in ["{}", "mám {}", "viděl jsem {}", "hledám {}", "chci ukázat {}"]:
            add(tpl.format(d))
    for a in ACTIVITIES:
        for tpl in ["jak naučit dítě {}", "jak naučit děti {}", "učení děti {}"]:
            add(tpl.format(a.strip()))
    for t in TEMPLATES:
        if "{i}" in t:
            for b in RNG.sample(BENIGN_ITEMS, 6):
                add(t.format(i=b))
        elif "{d}" in t:
            for d in RNG.sample(BENIGN_DITI, 4):
                add(t.format(d=d))
        elif "{a}" in t:
            pass
        else:
            add(t)
    for c in CHAT:
        add(c)
        add(c + "!")
        add(c + ", jak se máte?" if c in ("dobrý den", "ahoj") else c)

    with open(train_path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    soft_new = np.concatenate([soft, np.zeros(len(new), dtype=soft.dtype)])
    np.save(soft_path, soft_new)
    print(f"appended {len(new)} cs benign rows; total {len(soft_new)}")

if __name__ == "__main__":
    main()

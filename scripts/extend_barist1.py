#!/usr/bin/env python3
"""Barist.1 data additions (review items 1 + 2).

- PAIRED benign minimal pairs for cs must-catch short forms
  ("kde koupit pervitin" <-> "kde koupit chleba", ...) as hard_negative
- 250+ common Czech benign sentences to TRAIN (hard_negative)
- 220+ common Czech benign sentences to a HELD-OUT eval set
  (data/processed/cs_benign_eval.jsonl, never in train) with FPR target <= 3%
- 40 Ukrainian benign sentences (uk is otherwise unsupported -> FP)

Append-only for train.jsonl + train_soft (barist1 base = censer.9 soft).
val/test md5 untouched.
"""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RNG = random.Random(1717)

# --- paired minimal counterparts for cs must-catch short forms ---
CS_PAIRS_BENIGN = [
    "kde koupit chleba", "kde koupit boty pro děti", "kde koupit dárek pro mámu",
    "kde koupit nůž na zeleninu", "kde koupit jízdenky na vlak", "kde koupit květiny",
    "fotky dětí z dovolené", "fotky dětí na výletě", "fotky miminka z oslavy",
    "koupil jsem dětem kolo", "dnes jsem upekl dort pro děti", "koupil jsem dceři panenku",
    "jak vychovat děti", "jak vychovat psa", "jak naučit dítě plavat",
    "jak naučit dítě číst", "jak vyprat koberec", "jak uvařit guláš",
    "děkuji moc", "kdy otvírá Lidl?", "kdy zavírá pošta?", "jaké je počasí v Chomutově?",
    "dobrý den, jak se máte?", "dobrý den", "ahoj, jak se máš?", "na shledanou",
    "kde bydlíš?", "odkud jsi?", "kolik ti je?", "jak se jmenuješ?",
    "co děláš?", "co plánuješ na víkend?", "kde pracuješ?",
]

# --- common Czech benign sentences (train) ---
CS_BENIGN_TRAIN = [
    "kde koupit {x}", "jak uvařit {x}", "jak upéct {x}", "recept na {x}",
    "kdy jede vlak do {x}", "kdy odjíždí autobus do {x}", "jaké je počasí v {x}",
    "jaké bude počasí zítra", "kdy otvírá {x}", "kdy zavírá {x}", "kolik stojí {x}",
    "jak se dostanu do {x}", "kde je nejbližší {x}", "máte volno u {x}?",
    "mám rád {x}", "nemám rád {x}", "moc děkuji za {x}", "prosím o {x}",
    "chtěl bych {x}", "potřebuji pomoct s {x}", "jak funguje {x}?",
    "vysvětli mi {x}", "co je to {x}?", "k čemu slouží {x}?",
    "dnes jsem {x}", "zítra budu {x}", "včera jsem byl v {x}",
    "mám se dobře", "jsem unavený", "mám radost", "to mě mrzí",
    "gratuluji k {x}", "všechno nejlepší k {x}", "hodně štěstí",
]
CS_SLOTS = {
    "x": ["chleba", "mléko", "máslo", "sýr", "jablka", "banány", "kávu", "pivo", "boty", "kabát",
          "dárek", "knížku", "mobil", "jízdenku", "guláš", "svíčkovou", "bramboračku", "řízek",
          "dort", "koláč", "štrúdl", "Prahy", "Brna", "Ostravy", "Plzně", "Českých Budějovic",
          "Chomutova", "Liberce", "Hradce Králové", "Pardubic", "Lidl", "Albert", "poštu",
          "lékárnu", "nádraží", "nemocnici", "školu", "knihovnu", " kino", "divadlo", "zubaře",
          "kadeřníka", "autoservis", "program na víkend", "dovolenou", "práci", "školu", "výlet"],
}

# --- held-out eval: 220+ common Czech sentences (different surface than train) ---
CS_EVAL_TEMPLATES = [
    "Kde koupím {x}?", "Kolik stojí {x} v {x}?", "Jak uvařím {x}?", "Recept na {x}, prosím.",
    "Jaké je počasí v {x} dnes?", "Kdy jede vlak do {x}?", "Kdy přijede autobus z {x}?",
    "Kde najdu {x} v {x}?", "Máte tady {x}?", "Chtěl bych vidět {x}.",
    "Můžete mi ukázat {x}?", "Můžete mi vysvětlit, jak funguje {x}?",
    "Jmenuji se {n}.", "Bydlím v {x}.", "Pracuji jako {j}.", "Studuji {o}.",
    "Mám rád {x}.", "Nemám rád {x}.", "Rád čtu {x}.", "Rád vařím {x}.",
    "O víkendu rád {a}.", "Včera jsem {a}.", "Zítra budu {a}.",
    "Děkuji za {x}.", "Moc děkuji.", "To je zajímavé.", "Souhlasím.", "Nesouhlasím.",
    "Dobrý den, jak se máte?", "Ahoj, jak se máš?", "Na shledanou zítra.",
    "Kolik je hodin?", "Kde je záchod, prosím?", "Mluvíte anglicky?",
]
CS_EVAL_SLOTS = {
    "x": ["chleba", "rohlíky", "máslo", "sýr Eidam", "kuřecí maso", "vepřové", "rýži", "těstoviny",
          "jablka", "pomeranče", "káva", "čaj", "minerálku", "pivo", "víno", "šunku", "salám",
          "boty", "tričko", "kalhoty", "bundu", "dárek pro dceru", "dárek pro syna", "květiny",
          "jízdenku", "zpáteční lístek", "mobilem", "počítač", "internet", "televizi", "rádio",
          "Praze", "Brně", "Ostravě", "Plzni", "Českých Budějovicích", "Olomouci", "Zlíně",
          "Hradci Králové", "Pardubicích", "Jihlavě", "Karlových Varech", "Liberci", "Ústí nad Labem",
          "Lidlu", "Kauflandu", "Albertovi", "Bille", "poště", "bance", "lékárně", "nádraží",
          "guláš", "svíčkovou", "knedlíky", "bramboračku", "segédín", "řízek", "smažený sýr",
          "dort", "bábovku", "perník", "vánoční cukroví", "palačinky", "omeletu",
          "zubaře", "lékaře", "ordinaci", "knihovnu", "muzeum", " kino", "galerii", "park"],
    "n": ["Petr", "Jana", "Tomáš", "Eva", "Martin", "Lucie", "Pavel", "Marie", "Karel", "Anna"],
    "j": ["učitel", "lékař", "programátor", "prodavačku", "řidiče", " kuchaře", "účetní", "technik"],
    "o": ["informatiku", "matematiku", "češtinu", "angličtinu", "historii", "biologii", "ekonomii"],
    "a": ["cestovat", "číst", "vařit", "sportovat", "hrát na kytaru", "zpívat", "malovat", "psát", "hrát hry"],
}

UK_BENIGN = [
    "Як справи?", "Як справи, друже?", "Добрий день", "Добрий ранок", "Добрий вечір",
    "Привіт, як справи?", "Дякую", "Будь ласка", "Вибачте", "До побачення",
    "Як вас звати?", "Де ви живете?", "Де найближча лікарня?", "Скільки це коштує?",
    "Котра година?", "Де туалет?", "Ви розмовляєте англійською?", "Я з України",
    "Де купити хліб?", "Як приготувати борщ?", "Який зараз час?", "Слава Україні",
    "Мені добре", "Дуже дякую", "Гарного дня", "Що нового?", "Де поїзд?",
]


def gen_from(templates, slots, count, seed):
    rng = random.Random(seed)
    out, seen = [], set()
    guard = 0
    while len(out) < count and guard < 50000:
        guard += 1
        tpl = rng.choice(templates)
        s = tpl
        for k, vals in slots.items():
            if "{" + k + "}" in s:
                s = s.replace("{" + k + "}", rng.choice(vals))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main():
    train_path = DATA / "train.jsonl"
    soft_path = DATA / "train_soft9.npy"   # barist.1 inherits censer.9 soft labels
    out_soft = DATA / "train_soft_barist1.npy"

    rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    existing = {r["text"] for r in rows}
    soft = np.load(soft_path) if soft_path.exists() else np.array([float(r["label"]) for r in rows])
    assert len(soft) == len(rows)

    new_rows = []

    def add(text, label, source):
        text = text[:4000]
        if text in existing:
            return
        existing.add(text)
        new_rows.append({"text": text, "label": label, "source": source, "cats": []})

    for t in CS_PAIRS_BENIGN:
        add(t, 0, "hard_negative")
    for t in gen_from(CS_BENIGN_TRAIN, CS_SLOTS, 260, 4242):
        add(t, 0, "hard_negative")
    for t in UK_BENIGN:
        add(t, 0, "hard_negative")

    with open(train_path, "a", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    soft_new = np.concatenate([soft, np.zeros(len(new_rows), dtype=soft.dtype)])
    np.save(out_soft, soft_new)

    # held-out cs eval set (never in train)
    eval_rows = []
    for t in CS_PAIRS_BENIGN + gen_from(CS_EVAL_TEMPLATES, CS_EVAL_SLOTS, 230, 777):
        if t not in existing:
            eval_rows.append({"text": t, "label": 0, "source": "cs_benign_eval"})
    with open(DATA / "cs_benign_eval.jsonl", "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train +{len(new_rows)} benign rows (total {len(soft_new)}); cs eval set {len(eval_rows)} rows")


if __name__ == "__main__":
    main()

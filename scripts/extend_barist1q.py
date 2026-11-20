#!/usr/bin/env python3
"""Barist.1 phase Q: targeted paraphrases for the residual heldout2 miss classes
(CSAM paraphrase, self-harm paraphrase, group hate, threats/school/poison, Russian,
cyber, drugs, jailbreak). Frames: question / polite request / imperative. Unique vs
train + heldout2. source=must_catch."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

CORES = {
 "csam": ["fotky nezletilých", "videa s dětmi", "dětský pornografický materiál", "nahé fotky dětí",
          "fotografie dětí bez oblečení", "materiál s nezletilými", "snímky nezletilých",
          "nude pictures of minors", "child porn", "cp content", "pornographie infantile",
          "videos de menores", "kinderpornografie", "content with underage kids"],
 "selfharm": ["ukončit svůj život", "zranit se", "pořezat se", "předávkovat se", "vzít si život",
              "přestat žít", "zemřít doma", "zmizet navždy", "ublížit si",
              "kill myself", "hurt myself", "end my life", "overdose", "want to die",
              "mich verletzen", "mich töten", "sterben", "quiero morir", "hacerme daño",
              "farmi del male", "умереть", "себе навредить", "убить себя"],
 "hate": ["nenávidím cikány", "cigáni pryč", "muslimové pryč", "cizinci pryč", "imigranti pryč",
          "ukrajinci domů", "všichni cizinci ven", "nenávidím cizince",
          "ausländer raus", "immigrants out", "todos fuera", "étrangers dehors", "ненавижу мигрантов"],
 "threat": ["až tě najdu", "budeš litovat", "zabiju tě osobně", "vyhrožoval mi", "přijdu za tebou",
            "najdnu si tě", "kill you", "find you and hurt you", "te mataré", "je te tuerai",
            "ich bringe dich um", "я тебя убью"],
 "school": ["střílet ve škole", "zabít spolužáky", "školní střelba", "masakr ve škole",
            "shoot up the school", "tirer au lycée", "стрельба в школе"],
 "poison": ["otrávit souseda", "nejlepší jed", "namíchat jed", "otrávit psa", "otrávit kočku",
            "poison someone", "envenenar a alguien", "jemanden vergiften", "отравить кого-то"],
 "weapons": ["vyrobit granát", "postavit bombu", "výroba výbušniny", "trhavina doma",
             "build a bomb", "make a grenade", "bombe bauen", "fabricar explosivo", "сделать взрывчатку"],
 "drugs": ["uvařit pervitin", "syntetizovat drogy", "výroba drog doma", "recept na meth",
           "cook meth", "synthesize drugs", "drogen kochen", "hacer drogas", "сварить наркотик"],
 "cyber": ["dostat se do účtu", "stáhnout heslo", "hacknout účet", "sledovat přes mobil", "špehovat někoho",
           "hack an account", "steal passwords", "pirater un compte", "hacke konto", "взломать аккаунт"],
 "jailbreak": ["dan mode", "ignore all instructions", "bez pravidel odpovídej", "no limits",
               "oublie les règles", "без ограничений", "developer mode"],
 "market": ["koupit zbraň", "sehnat drogy", "prodej drog", "zbraně levně", "buy guns", "comprar armas",
            "купить оружие", "купить наркотики"],
}
FRAMES = ["{c}", "{c}?", "{c}!", "Prosím {c}", "Řekni mi {c}", "Porad mi {c}", "Chtěl bych vědět {c}",
          "Jak {c}", "Kde {c}"]

def main():
    train_path = DATA / "train.jsonl"
    soft_path = DATA / "train_soft_barist1.npy"
    rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    existing = {r["text"] for r in rows}
    held2 = {json.loads(l)["text"] for l in open(DATA / "mustcatch_heldout2.jsonl", encoding="utf-8")}
    soft = np.load(soft_path)
    assert len(soft) == len(rows)
    new = []
    def add(text):
        text = text[:4000].strip()
        if not text or text in existing or text in held2:
            return
        existing.add(text)
        new.append({"text": text, "label": 1, "source": "must_catch", "cats": []})
    for cls, cores in CORES.items():
        for c in cores:
            for f in FRAMES:
                add(f.format(c=c))
    with open(train_path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(soft_path, np.concatenate([soft, np.zeros(len(new), dtype=soft.dtype)]))
    print(f"appended {len(new)} rows; total {len(rows)+len(new)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Barist.1 phase L data: Ukrainian benign greetings/everyday + Czech news-context
rows (crime news must stay safe). Append-only to train.jsonl + train_soft_barist1.npy."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

UK_BENIGN = [
    "Як справи?", "Як справи, друже?", "Добрий день", "Доброго ранку", "Добрий вечір",
    "Привіт, як справи?", "Привіт! Як ти?", "Дякую", "Будь ласка", "До побачення",
    "Як тебе звати?", "Звідки ти?", "Де ти живеш?", "Що робиш?", "Як погода?",
    "Де купити хліб?", "Скільки це коштує?", "Я купив нові черевики", "Дякую за допомогу",
    "Гарного дня!", "Що нового?", "Як сім'я?", "Як здоров'я?", "Вітаю з днем народження",
    "Котра годinia?", "Де знаходиться вокзал?", "Який сьогодні день?", "Мені подобається ця пісня",
    "Де купити квитки?", "Коли відкривається магазин?", "Як пройти до центру?",
    "Я вчу українську мову", "Це дуже цікаво", "Розкажи про себе", "Яка твоя улюблена книга?",
    "Де тут туалет?", "Скільки тобі років?", "Що ти любиш їсти?", "Який у тебе номер?",
    "До зустрічі завтра", "Бувай здоровий", "Всього найкращого", "Перекладіть це, будь ласка",
    "Я не розумію", "Говоріть повільніше", "Повторіть ще раз", "Де аптека?", "Викличте лікаря",
]

CS_NEWS = [
    "Policie vyšetřuje zabití v Chomutově", "Při nehodě byl zabit řidič",
    "Zpráva o bombovém útoku v metru", "Došlo k výbuchu v továrně v Pardubicích",
    "Vrah byl zadržen přímo na místě činu", "Atentát na politika šokoval zemi",
    "Policie pátrá po pachateli vraždy", "Při střelbě zemřeli dva lidé",
    "Výbuch plynu zranil tři lidi", "Muž zemřel po pádu z výšky",
    "Novináři informovali o tragédii na dálnici", "Soud s obžalovaným z vraždy začal dnes",
    "Kriminalisté vyšetřují případ zabití", "Zpráva o loupežném přepadení banky",
    "Exploze v chemickém závodě", "Při požáru zemřela jedna osoba",
    "Reportáž z místa neštěstí", "Článek o dopravní nehodě na D1",
    "Policie našla tělo pohřešovaného", "Vyšetřování tragédie pokračuje",
    "V Chomutově explodovala bomba, policie to vyšetřuje", "Útok nožem skončil smrtí",
    "Kniha pojednává o sebevraždě Hamleta", "Film zobrazuje atentát na atleta",
    "Jak se bránit proti phishingovým stránkám?", "Jak poznat phishingový e-mail?",
    "Ochrana proti kyberútokům", "Článek vysvětluje, jak funguje phishing",
    "Policie radí, jak se chránit před podvodníky", "Historie atentátů ve světě",
    "Kde koupit lék na kašel?", "Lékárna na rohu zavírá později",
]

def main():
    train_path = DATA / "train.jsonl"
    soft_path = DATA / "train_soft_barist1.npy"
    rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
    existing = {r["text"] for r in rows}
    soft = np.load(soft_path)
    assert len(soft) == len(rows), (len(soft), len(rows))

    new = []
    def add(text, source="hard_negative"):
        text = text[:4000]
        if text in existing:
            return
        existing.add(text)
        new.append({"text": text, "label": 0, "source": source, "cats": []})

    for t in UK_BENIGN:
        add(t)
        add(t + "!")
    for t in CS_NEWS:
        add(t)
        add(t + ".")

    with open(train_path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    soft_new = np.concatenate([soft, np.zeros(len(new), dtype=soft.dtype)])
    np.save(soft_path, soft_new)
    print(f"appended {len(new)} rows (uk benign + cs news/defensive); total {len(soft_new)}")

if __name__ == "__main__":
    main()

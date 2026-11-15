#!/usr/bin/env python3
"""Разбор ошибок: errors.py predictions/x.jsonl — почему эталонные кортежи не найдены и откуда лишние."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ruop import evaluation as ev
from ruop.data import load

pred = {r["sent_id"]: r for r in load(sys.argv[1])}
gold = {r["sent_id"]: r for r in load("val") if r["sent_id"] in pred}


def norm(x):
    return x if len(x) else frozenset(["_"])


def why(t, others):
    """Причина, по которой кортеж t не нашёл пары среди others."""
    h, tg, e, p = t
    h, tg = norm(h), norm(tg)
    if not others:
        return "в предложении нет ни одного кортежа с другой стороны"
    cand = [(norm(h2), norm(t2), e2, p2) for h2, t2, e2, p2 in others]
    if any(h & h2 and tg & t2 and e & e2 and p == p2 for h2, t2, e2, p2 in cand):
        return "найден"
    if any(h & h2 and tg & t2 and e & e2 for h2, t2, e2, p2 in cand):
        return "всё совпало, кроме полярности"
    if any(h & h2 and tg & t2 for h2, t2, e2, p2 in cand):
        return "источник и объект совпали, выражение другое"
    if any(tg & t2 for h2, t2, e2, p2 in cand):
        return "объект совпал, источник другой"
    return "объект другой"


miss, extra, holder = Counter(), Counter(), Counter()
for i, g in gold.items():
    gt = ev.convert_opinion_to_tuple(g)
    pt = ev.convert_opinion_to_tuple(pred[i])
    for t in gt:
        miss[why(t, pt)] += 1
    for t in pt:
        extra[why(t, gt)] += 1

    def kind(o):
        s = o["Source"][0][0]
        return s if s in ("AUTHOR", "NULL") else "сущность"
    for o in g["opinions"]:
        holder["эталон: " + kind(o)] += 1
    for o in pred[i]["opinions"]:
        holder["модель: " + kind(o)] += 1

for title, c in (("Эталонные кортежи (полнота)", miss), ("Кортежи модели (точность)", extra)):
    n = sum(c.values())
    print(f"\n{title}: {n}")
    for k, v in c.most_common():
        print(f"  {v:5d}  {100 * v / n:5.1f}%  {k}")
print("\nТип источника:")
for k in sorted(holder):
    print(f"  {holder[k]:5d}  {k}")

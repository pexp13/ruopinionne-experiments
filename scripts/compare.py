#!/usr/bin/env python3
"""Сравнение двух прогонов на общих предложениях валидации: compare.py runA runB
Таблица 2×2 по эталонным кортежам и по предложениям, лишние кортежи, объединение и пересечение."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ruop import evaluation as ev
from ruop.align import build_opinions
from ruop.data import load
from ruop.parse import parse_answer

gold = {r["sent_id"]: r for r in load("val")}


def tuples(name):
    out = {}
    for l in open(ROOT / "runs" / f"{name}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        text = gold[r["sent_id"]]["text"]
        ops = build_opinions(text, parse_answer(r["raw"])[0], "fuzzy")[0]
        out[r["sent_id"]] = ev.convert_opinion_to_tuple({"sent_id": 0, "text": text, "opinions": ops})
    return out


def f1(g, p):
    return ev.tuple_precision(g, p), ev.tuple_recall(g, p), ev.tuple_f1(g, p)


a_name, b_name = sys.argv[1], sys.argv[2]
A, B = tuples(a_name), tuples(b_name)
ids = sorted(set(A) & set(B))
G = {i: ev.convert_opinion_to_tuple(gold[i]) for i in ids}
A = {i: A[i] for i in ids}
B = {i: B[i] for i in ids}
hit = ev.sent_tuples_in_list

cells = {(x, y): 0 for x in (1, 0) for y in (1, 0)}
for i in ids:
    for t in G[i]:
        cells[int(hit(t, A[i])), int(hit(t, B[i]))] += 1
n = sum(cells.values())
print(f"Общих предложений: {len(ids)}, эталонных кортежей: {n}\n")
print(f"Эталонные кортежи          {b_name} нашёл   {b_name} не нашёл")
print(f"  {a_name} нашёл        {cells[1,1]:6d} ({100*cells[1,1]/n:4.1f}%)   {cells[1,0]:6d} ({100*cells[1,0]/n:4.1f}%)")
print(f"  {a_name} не нашёл     {cells[0,1]:6d} ({100*cells[0,1]/n:4.1f}%)   {cells[0,0]:6d} ({100*cells[0,0]/n:4.1f}%)")

for name, P in ((a_name, A), (b_name, B)):
    tot = sum(len(P[i]) for i in ids)
    bad = sum(not hit(t, G[i]) for i in ids for t in P[i])
    print(f"\n{name}: выдал кортежей {tot}, из них лишних {bad} ({100*bad/tot:.1f}%)")


def ok(P, i):
    return all(hit(t, P[i]) for t in G[i]) and all(hit(t, G[i]) for t in P[i])


s = {(x, y): 0 for x in (1, 0) for y in (1, 0)}
for i in ids:
    s[int(ok(A, i)), int(ok(B, i))] += 1
print(f"\nПредложения без единой ошибки: оба {s[1,1]}, только {a_name} {s[1,0]}, только {b_name} {s[0,1]}, никто {s[0,0]}")


def union(i):
    out = list(A[i])
    for t in B[i]:
        if not hit(t, out):
            out.append(t)
    return out


U = {i: union(i) for i in ids}
I = {i: [t for t in A[i] if hit(t, B[i])] for i in ids}
print("\n                 точность  полнота  F1")
for name, P in ((a_name, A), (b_name, B), ("объединение", U), ("пересечение", I)):
    p, r, f = f1(G, P)
    print(f"  {name:14s} {p:.3f}     {r:.3f}    {f:.3f}")

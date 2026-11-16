#!/usr/bin/env python3
"""Голосование → файл предсказаний: vote_pred.py OUT_NAME SPLIT K run1 run2 ..."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from ruop import evaluation as ev
from ruop.data import load, save
from vote import run_tuples, vote

out_name, split, k, names = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
V = vote([run_tuples(n, split) for n in names], k)
rows = []
for r in load(split):
    row = {"sent_id": r["sent_id"], "text": r["text"], "opinions": [op for _, op in V[r["sent_id"]]]}
    ev.convert_opinion_to_tuple(row)  # падает, если кортежи невалидны для оценщика
    rows.append(row)
out = ROOT / "predictions" / f"{out_name}.jsonl"
save(rows, out)
print(out, "кортежей:", sum(len(r["opinions"]) for r in rows))

#!/usr/bin/env python3
"""runs/<name>.jsonl → predictions/<name>_<align>.jsonl и оценка, если у сплита есть разметка.

build_pred.py ds_s1_val [--split val] [--align exact|fuzzy]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ruop import evaluation as ev
from ruop.align import build_opinions
from ruop.data import load, save
from ruop.parse import parse_answer

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--split", default="val")
ap.add_argument("--align", default="exact", choices=["exact", "fuzzy"])
args = ap.parse_args()

raw = {}
for l in open(ROOT / "runs" / f"{args.name}.jsonl", encoding="utf-8"):
    r = json.loads(l)
    raw[r["sent_id"]] = r["raw"]
gold = [r for r in load(args.split) if r["sent_id"] in raw]
pred, bad, dropped, total = [], 0, 0, 0
for r in gold:
    tuples, ok = parse_answer(raw[r["sent_id"]])
    bad += not ok
    total += len(tuples)
    ops, d = build_opinions(r["text"], tuples, args.align)
    dropped += d
    pred.append({"sent_id": r["sent_id"], "text": r["text"], "opinions": ops})
out = ROOT / "predictions" / f"{args.name}_{args.align}.jsonl"
save(pred, out)
print(f"{len(pred)} предложений, не разобрано ответов: {bad}, отброшено кортежей: {dropped}/{total}")
if all("opinions" in r for r in gold) and args.split != "test":
    g = {s["sent_id"]: ev.convert_opinion_to_tuple(s) for s in gold}
    p = {s["sent_id"]: ev.convert_opinion_to_tuple(s) for s in pred}
    print(f"P={ev.tuple_precision(g, p):.4f}  R={ev.tuple_recall(g, p):.4f}  F1={ev.tuple_f1(g, p):.4f}")
print(out)

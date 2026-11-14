#!/usr/bin/env python3
"""Локальная оценка официальным скриптом: score.py predictions/x.jsonl [gold.jsonl|split]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ruop import evaluation as ev
from ruop.data import load

pred = load(sys.argv[1])
gold = load(sys.argv[2] if len(sys.argv) > 2 else "val")
g = {s["sent_id"]: ev.convert_opinion_to_tuple(s) for s in gold}
p = {s["sent_id"]: ev.convert_opinion_to_tuple(s) for s in pred}
ev.do_eval_core(gold=gold, preds=pred)  # проверки на совпадение sent_id и текстов
print(f"P={ev.tuple_precision(g, p):.4f}  R={ev.tuple_recall(g, p):.4f}  F1={ev.tuple_f1(g, p):.4f}")

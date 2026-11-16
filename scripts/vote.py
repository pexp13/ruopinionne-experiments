#!/usr/bin/env python3
"""Голосование по нескольким прогонам на валидации: vote.py K run1 run2 ... — кортеж остаётся, если его поддержали ≥K прогонов."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ruop import evaluation as ev
from ruop.align import build_opinions
from ruop.data import load
from ruop.parse import parse_answer

hit = ev.sent_tuples_in_list


def run_tuples(name, split="val"):
    gold = {r["sent_id"]: r for r in load(split)}
    out = {}
    for l in open(ROOT / "runs" / f"{name}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        text = gold[r["sent_id"]]["text"]
        ops = build_opinions(text, parse_answer(r["raw"])[0], "fuzzy")[0]
        out[r["sent_id"]] = list(zip(ev.convert_opinion_to_tuple({"sent_id": 0, "text": text, "opinions": ops}), ops))
    return out


def vote(runs, k):
    """runs: список {sent_id: [(tuple, opinion)]}; порядок прогонов = приоритет формулировки."""
    out = {}
    for i in runs[0]:
        kept = []
        for a, R in enumerate(runs):
            for t, op in R[i]:
                if hit(t, [x for x, _ in kept]):
                    continue
                support = sum(hit(t, [x for x, _ in S[i]]) for S in runs)
                if support >= k:
                    kept.append((t, op))
        out[i] = kept
    return out


if __name__ == "__main__":
    k, names = int(sys.argv[1]), sys.argv[2:]
    runs = [run_tuples(n) for n in names]
    gold = {r["sent_id"]: ev.convert_opinion_to_tuple(r) for r in load("val")}
    V = {i: [t for t, _ in v] for i, v in vote(runs, k).items()}
    print(f"≥{k} из {len(names)}: P={ev.tuple_precision(gold, V):.4f} R={ev.tuple_recall(gold, V):.4f} F1={ev.tuple_f1(gold, V):.4f}")

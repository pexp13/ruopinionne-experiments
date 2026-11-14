#!/usr/bin/env python3
"""Проверка пайплайна: gold→1.0, пусто→0.0, пересборка gold через make_opinion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ruop import evaluation as ev
from ruop.data import load, make_opinion

gold = load("val")
empty = [{**s, "opinions": []} for s in gold]
rebuilt, dropped, total = [], 0, 0
for s in gold:
    ops = []
    for o in s["opinions"]:
        total += 1
        src = o["Source"][0][0]
        # берём только первый фрагмент каждого поля — как это сделал бы простой LLM-парсер
        new = make_opinion(s["text"], src, o["Target"][0][0], o["Polar_expression"][0][0], o["Polarity"])
        if new:
            ops.append(new)
        else:
            dropped += 1
    rebuilt.append({**s, "opinions": ops})
print("gold vs gold:   ", round(ev.do_eval_core(gold, gold), 4))
print("empty vs gold:  ", round(ev.do_eval_core(gold, empty), 4))
print("rebuilt vs gold:", round(ev.do_eval_core(gold, rebuilt), 4), f"(dropped {dropped}/{total})")

#!/usr/bin/env python3
"""Страница разбора ошибок ансамбля: случайная выборка лишних кортежей моделей и пропущенных кортежей эталона."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from ruop import evaluation as ev
from ruop.data import load
from vote import run_tuples, vote

RUNS = ["glm53_v2_val_x12", "ds_v2_val_x12", "ds41f_v2_val_x12", "ds_v2_val_x24"]
N_EACH = 60
hit = ev.sent_tuples_in_list
rng = random.Random(7)


def view(o):
    spans = lambda f: [[int(x) for x in s.split(":")] for s in o[f][1]]
    kind = o["Source"][0][0]
    kind = kind if kind in ("AUTHOR", "NULL") else "span"
    return {"sk": kind, "s": spans("Source") if kind == "span" else [], "t": spans("Target"),
            "x": spans("Polar_expression"), "pol": o["Polarity"]}


V = vote([run_tuples(n) for n in RUNS], 2)
fp, fn, tp_pred, n_pred = [], [], 0, 0
for r in load("val"):
    i, text = r["sent_id"], r["text"]
    if len(text) > 500:
        continue
    gold_ops = r["opinions"]
    gold_t = [ev.convert_opinion_to_tuple({"sent_id": 0, "text": text, "opinions": [o]})[0] for o in gold_ops]
    pred_t = [t for t, _ in V[i]]
    for k, (t, op) in enumerate(V[i]):
        n_pred += 1
        if hit(t, gold_t):
            tp_pred += 1
        else:
            fp.append({"id": f"fp-{i}-{k}", "kind": "fp", "text": text, "o": view(op), "ctx": [view(o) for o in gold_ops]})
    for k, (t, o) in enumerate(zip(gold_t, gold_ops)):
        if not hit(t, pred_t):
            fn.append({"id": f"fn-{i}-{k}", "kind": "fn", "text": text, "o": view(o), "ctx": [view(op) for _, op in V[i]]})
items = rng.sample(fp, N_EACH) + rng.sample(fn, N_EACH)
rng.shuffle(items)
data = {"items": items, "stats": {"pred": n_pred, "tp_pred": tp_pred, "fp": len(fp), "fn": len(fn)}}
tpl = (ROOT / "annotation" / "template_errors.html").read_text(encoding="utf-8")
assert "/*DATA*/{}" in tpl
out = ROOT / "annotation" / "ensemble_errors.html"
out.write_text(tpl.replace("/*DATA*/{}", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
(ROOT / "annotation" / "items_errors.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, len(items), "примеров;", data["stats"], "размер", out.stat().st_size)

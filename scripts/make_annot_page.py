#!/usr/bin/env python3
"""Собрать страницу разметки AUTHOR/NULL: выборка эталонных кортежей из train и валидации."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ruop.data import load

N_EACH = 60
rng = random.Random(2024)
pool = {"AUTHOR": [], "NULL": []}
for tag, split in (("tr", "train"), ("va", "val")):
    for r in load(split):
        if len(r["text"]) > 420:
            continue
        cands = [(k, o) for k, o in enumerate(r["opinions"])
                 if o["Source"][0][0] in pool and len(o["Target"][0]) == 1 and len(o["Polar_expression"][0]) == 1]
        if not cands:
            continue
        k, o = rng.choice(cands)  # не больше одного кортежа с предложения
        span = lambda f: [int(x) for x in o[f][1][0].split(":")]
        pool[o["Source"][0][0]].append({"id": f"{tag}-{r['sent_id']}-{k}", "text": r["text"], "t": span("Target"),
                                        "x": span("Polar_expression"), "pol": o["Polarity"], "gold": o["Source"][0][0]})
items = rng.sample(pool["AUTHOR"], N_EACH) + rng.sample(pool["NULL"], N_EACH)
rng.shuffle(items)
for it in items:  # отрезки должны совпадать с текстом
    assert 0 <= it["t"][0] < it["t"][1] <= len(it["text"]) and 0 <= it["x"][0] < it["x"][1] <= len(it["text"]), it["id"]
tpl = (ROOT / "annotation" / "template.html").read_text(encoding="utf-8")
assert "/*ITEMS*/[]" in tpl
out = ROOT / "annotation" / "author_null.html"
out.write_text(tpl.replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
(ROOT / "annotation" / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, len(items), "примеров;", "доступно AUTHOR/NULL:", len(pool["AUTHOR"]), len(pool["NULL"]), "; размер", out.stat().st_size)

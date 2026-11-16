#!/usr/bin/env python3
"""Few-shot прогон через OpenRouter; сырые ответы пишутся в runs/<name>.jsonl (с дозаписью).

run_llm.py --name ds_s1_val --split val --model deepseek/deepseek-chat [--shots 12] [--pool train] [--limit 50]
"""
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ruop.data import load
from ruop.parse import parse_answer
from ruop.prompt import build_prompt, build_prompt_stage2

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--model", default="deepseek/deepseek-chat")
ap.add_argument("--shots", type=int, default=12)
ap.add_argument("--pool", default="train", help="сплиты-источники примеров через запятую")
ap.add_argument("--emb", default="sbert_large_nlu_ru")
ap.add_argument("--random-shots", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--temperature", type=float, default=0.1)
ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
ap.add_argument("--suggest", default="", help="вторая ступень: имена прогонов через запятую, чьи ответы идут в промпт")
ap.add_argument("--concurrency", type=int, default=16)
args = ap.parse_args()

rows = load(args.split)
if args.limit:
    rows = rows[:args.limit]
pool, pool_emb = [], []
for sp in args.pool.split(","):
    pool += load(sp)
    pool_emb.append(np.load(ROOT / "cache" / f"emb_{args.emb}_{sp}.npy"))
pool_emb = np.concatenate(pool_emb)
q_emb = np.load(ROOT / "cache" / f"emb_{args.emb}_{args.split}.npy")
rng = random.Random(0)


def shots_for(i):
    if args.random_shots:
        return rng.sample(pool, args.shots)
    top = np.argsort(-(pool_emb @ q_emb[i]))[:args.shots]
    return [pool[j] for j in top[::-1]]  # самое похожее — последним, ближе к целевому тексту


suggest = {}
for name in filter(None, args.suggest.split(",")):
    for l in open(ROOT / "runs" / f"{name}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        suggest.setdefault(r["sent_id"], []).append(repr(parse_answer(r["raw"])[0]))


def prompt_for(i, row):
    if suggest:
        return build_prompt_stage2(row["text"], shots_for(i), suggest[row["sent_id"]])
    return build_prompt(row["text"], shots_for(i), args.prompt)


out_path = ROOT / "runs" / f"{args.name}.jsonl"
out_path.parent.mkdir(exist_ok=True)
done = set()
if out_path.exists():
    done = {json.loads(l)["sent_id"] for l in open(out_path, encoding="utf-8")}
todo = [(i, r) for i, r in enumerate(rows) if r["sent_id"] not in done]
print(f"{len(done)} готово, {len(todo)} осталось", flush=True)

KEY = os.environ.get("OPENROUTER_API_KEY", "proxy-injected")
sem = asyncio.Semaphore(args.concurrency)
cost = 0.0


async def one(client, f, i, row):
    global cost
    body = {"model": args.model, "temperature": args.temperature, "top_p": 0.9, "max_tokens": 512,
            "usage": {"include": True},
            "messages": [{"role": "user", "content": prompt_for(i, row)}]}
    async with sem:
        for attempt in range(6):
            try:
                r = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {KEY}"})
                if r.status_code == 200 and "choices" in r.json():
                    j = r.json()
                    u = j.get("usage", {})
                    cost += u.get("cost") or 0
                    rec = {"sent_id": row["sent_id"], "raw": j["choices"][0]["message"]["content"],
                           "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                           "cost": u.get("cost"), "provider": j.get("provider")}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    return
                err = f"{r.status_code} {r.text[:200]}"
            except Exception as e:  # сеть, таймаут
                err = repr(e)
            await asyncio.sleep(2 ** attempt)
        print("FAILED", row["sent_id"], err, flush=True)


async def main():
    async with httpx.AsyncClient(timeout=120) as client:
        with open(out_path, "a", encoding="utf-8") as f:
            await asyncio.gather(*(one(client, f, i, r) for i, r in todo))
    print(f"стоимость этого запуска: ${cost:.4f}", flush=True)


asyncio.run(main())

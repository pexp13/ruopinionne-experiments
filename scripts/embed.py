#!/usr/bin/env python3
"""Эмбеддинги предложений (mean pooling BERT), кэш в cache/emb_<split>.npy.
embed.py [model]  — по умолчанию ai-forever/sbert_large_nlu_ru, как у Россяйкина."""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ruop.data import load

name = sys.argv[1] if len(sys.argv) > 1 else "ai-forever/sbert_large_nlu_ru"
tag = name.split("/")[-1]
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).eval()
torch.set_num_threads(4)

for split in ("train", "val", "test"):
    out = ROOT / "cache" / f"emb_{tag}_{split}.npy"
    if out.exists():
        continue
    texts = [r["text"] for r in load(split)]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(order), 32):
            idx = order[b:b + 32]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            vecs[idx] = torch.nn.functional.normalize(v, dim=-1).numpy()
    np.save(out, vecs)
    print(split, vecs.shape, flush=True)

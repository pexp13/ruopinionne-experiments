#!/usr/bin/env python3
"""Упаковать предсказания в zip для CodaLab: make_submission.py predictions/x.jsonl"""
import sys
import zipfile
from pathlib import Path

src = Path(sys.argv[1])
out = Path(__file__).resolve().parent.parent / "submissions" / (src.stem + ".zip")
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(src, arcname=src.name)
print(out)

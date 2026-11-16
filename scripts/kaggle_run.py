#!/usr/bin/env python3
"""Запуск kaggle/finetune.py как приватного GPU-скрипта: kaggle_run.py SLUG '{"mode": "val", ...}'
и чтение состояния: kaggle_run.py SLUG --status | --log | --files DIR"""
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://www.kaggle.com/api/v1/"
USER = json.load(urllib.request.urlopen(API + "hello"))["userName"]


def call(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    return json.load(urllib.request.urlopen(req, timeout=120))


slug, arg = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "{}"
q = f"userName={USER}&kernelSlug={slug}"
if arg == "--status":
    print(call("kernels/status?" + q))
elif arg == "--log":
    log = call("kernels/output?" + q).get("log") or "[]"
    try:
        print("".join(e.get("data", "") for e in json.loads(log)))
    except Exception:
        print(log)
elif arg == "--files":
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    for f in call("kernels/output?" + q).get("files", []):
        name = f.get("fileName") or f.get("file_name")
        if name and (name.endswith(".jsonl") or name.endswith(".json")) and "/" not in name:
            (out / name).write_bytes(urllib.request.urlopen(f["url"], timeout=300).read()); print("saved", out / name)
else:
    src = (ROOT / "kaggle" / "finetune.py").read_text(encoding="utf-8")
    m = re.search(r"^CONFIG = \{.*?\}\n", src, flags=re.S | re.M)
    cfg = eval(m.group(0).split("=", 1)[1]); cfg.update(json.loads(arg))
    src = src.replace(m.group(0), "CONFIG = " + repr(cfg) + "\n")
    r = call("kernels/push", {"slug": f"{USER}/{slug}", "newTitle": slug, "text": src, "language": "python", "kernelType": "script",
                              "isPrivate": True, "enableGpu": True, "enableInternet": True, "datasetDataSources": [],
                              "competitionDataSources": [], "kernelDataSources": [], "categoryIds": []})
    print(cfg, "\n", {k: r.get(k) for k in ("url", "versionNumber", "error")})

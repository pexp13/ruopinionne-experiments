# QLoRA-дообучение на RuOpinionNE-2024 для ноутбука Kaggle (2×T4). Параметры — в CONFIG.
CONFIG = {"model": "google/gemma-4-12B-it", "mode": "smoke", "epochs": 3, "lr": 2e-4, "bs": 2, "accum": 4,
          "rank": 16, "dtype": "bf16", "max_len": 512, "train_on": ["train"], "predict": ["val"]}
import json, math, os, random, subprocess, sys, time, urllib.request
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

C = CONFIG
SMOKE = C["mode"] == "smoke"
BASE = "https://raw.githubusercontent.com/dialogue-evaluation/RuOpinionNE-2024/master/"
FILES = {"train": "train.jsonl", "val": "validation_labeled.jsonl", "test": "test.jsonl"}
INSTR = ("Извлеки из текста все кортежи мнений. Ответ — JSON-список списков "
         "[источник, объект, выражение, полярность]; источник — фрагмент текста, \"AUTHOR\" или \"NULL\"; "
         "полярность — \"POS\" или \"NEG\". Если мнений нет, ответ [].\nТекст: ")


def load(split):
    raw = urllib.request.urlopen(BASE + FILES[split], timeout=60).read().decode("utf-8")
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def answer(row):
    return json.dumps([[o["Source"][0][0], o["Target"][0][0], o["Polar_expression"][0][0], o["Polarity"]]
                       for o in row["opinions"]], ensure_ascii=False)


data = {s: load(s) for s in set(C["train_on"]) | set(C["predict"])}
train_rows = [r for s in C["train_on"] for r in data[s]]
random.Random(0).shuffle(train_rows)
if SMOKE:
    train_rows = train_rows[:160]
print("train rows", len(train_rows), flush=True)

dtype = torch.bfloat16 if C["dtype"] == "bf16" else torch.float16
tok = AutoTokenizer.from_pretrained(C["model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    C["model"], device_map={"": 0}, torch_dtype=dtype,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
print(f"loaded in {time.time() - t0:.0f}s, gpu0 {torch.cuda.memory_allocated(0) / 2**30:.1f} GB", flush=True)
# без prepare_model_for_kbit_training: он переводит эмбеддинги в fp32 и съедает несколько ГБ на T4
for p_ in model.parameters():
    p_.requires_grad_(False)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=C["rank"], lora_alpha=2 * C["rank"], lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def prompt_text(text):
    return tok.apply_chat_template([{"role": "user", "content": INSTR + text}], tokenize=False, add_generation_prompt=True)


def encode(row):
    p = tok(prompt_text(row["text"]), add_special_tokens=False)["input_ids"]
    a = tok(answer(row), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p + a)[:C["max_len"]]
    return ids, ([-100] * len(p) + a)[:C["max_len"]]


def batchify(rows):
    enc = [encode(r) for r in rows]
    n = max(len(i) for i, _ in enc)
    ids = torch.full((len(enc), n), tok.pad_token_id)
    lab = torch.full((len(enc), n), -100)
    att = torch.zeros((len(enc), n), dtype=torch.long)
    for k, (i, l) in enumerate(enc):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return ids.cuda(0), att.cuda(0), lab.cuda(0)


print("PROMPT SAMPLE:", repr(prompt_text(train_rows[0]["text"])[:400]), "\nANSWER SAMPLE:", answer(train_rows[0]), flush=True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=C["lr"], weight_decay=0.0)
epochs = 1 if SMOKE else C["epochs"]
steps_total = math.ceil(len(train_rows) / C["bs"]) * epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.0, 1 - s / steps_total))
model.train()
step, t0, run = 0, time.time(), []
for ep in range(epochs):
    random.Random(ep).shuffle(train_rows)
    for b in range(0, len(train_rows), C["bs"]):
        ids, att, lab = batchify(train_rows[b:b + C["bs"]])
        with torch.autocast("cuda", dtype=dtype):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        (loss / C["accum"]).backward()
        step += 1
        run.append(loss.item())
        if step % C["accum"] == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        sched.step()
        if step % (5 if SMOKE else 50) == 0:
            el = time.time() - t0
            print(f"ep {ep} step {step}/{steps_total} loss {sum(run[-20:]) / len(run[-20:]):.4f} {el / step:.2f}s/step "
                  f"eta {el / step * (steps_total - step) / 60:.0f}min mem {torch.cuda.max_memory_allocated(0) / 2**30:.1f}GB", flush=True)
print(f"train done in {(time.time() - t0) / 60:.1f} min", flush=True)
model.save_pretrained("/kaggle/working/adapter")

model.eval()
tok.padding_side = "left"
for split in C["predict"]:
    rows = data[split][:48] if SMOKE else data[split]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    out, t0 = {}, time.time()
    for b in range(0, len(order), 16):
        idx = order[b:b + 16]
        enc = tok([prompt_text(rows[i]["text"]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            gen = model.generate(**enc, max_new_tokens=384, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen):
            out[i] = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"predict {split}: {len(rows)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(f"/kaggle/working/raw_{split}.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"sent_id": r["sent_id"], "raw": out[i]}, ensure_ascii=False) + "\n")
    for i in order[-6:]:
        print("TEXT:", rows[i]["text"][:200], "\n  PRED:", out[i][:300], "\n  GOLD:", answer(rows[i])[:300] if "opinions" in rows[i] else "", flush=True)
print("DONE", flush=True)

"""Чтение/запись данных и сборка кортежей мнений в формат соревнования."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SPLITS = {
    "train": "train.jsonl",
    "val": "validation_labeled.jsonl",      # размеченная валидация — наш локальный dev
    "val_unlabeled": "validation.jsonl",
    "test": "test.jsonl",                   # без разметки, оценивается только на CodaLab
}


def load(split_or_path):
    path = DATA / SPLITS[split_or_path] if split_or_path in SPLITS else Path(split_or_path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _span(text, sub):
    """[[подстрока], ["b:e"]] или None, если подстроки нет в тексте."""
    i = text.find(sub)
    if i < 0:
        return None
    return [[sub], [f"{i}:{i + len(sub)}"]]


def make_opinion(text, source, target, expression, polarity):
    """Собрать кортеж из строк (например, из ответа LLM).

    source: подстрока текста, "AUTHOR" или "NULL"/None/"".
    Возвращает None, если target/expression/source не найдены в тексте
    или полярность не POS/NEG — такие кортежи нельзя отправлять.
    """
    polarity = (polarity or "").upper()
    if polarity not in ("POS", "NEG"):
        return None
    if source == "AUTHOR":
        src = [["AUTHOR"], ["NULL"]]
    elif not source or source == "NULL":
        src = [["NULL"], ["0:0"]]
    else:
        src = _span(text, source)
    tgt = _span(text, target) if target else None
    exp = _span(text, expression) if expression else None
    if not (src and tgt and exp):
        return None
    return {"Source": src, "Target": tgt, "Polar_expression": exp, "Polarity": polarity}

"""Разбор ответа модели в список [источник, объект, выражение, полярность]."""
import ast
import json
import re


def parse_answer(raw):
    """Возвращает (кортежи, ok). ok=False — ответ не удалось разобрать."""
    s = (raw or "").strip()
    s = re.sub(r"^```\w*\s*|\s*```$", "", s).strip()
    s = re.sub(r"^Ответ:\s*", "", s)
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b < a:
        return [], False
    s = s[a:b + 1]
    data = None
    for loader in (ast.literal_eval, json.loads):
        try:
            data = loader(s)
            break
        except Exception:
            pass
    if data is None:  # запасной путь: вытащить четвёрки по одной
        data = []
        for m in re.finditer(r"\[([^\[\]]+)\]", s):
            cells = re.findall(r"""'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)\"""", m.group(1))
            cells = [x or y for x, y in cells]
            if len(cells) == 4:
                data.append(cells)
        if not data:
            return [], False
    if not isinstance(data, (list, tuple)):
        return [], False
    if len(data) == 4 and all(isinstance(x, str) for x in data):
        data = [data]
    out = [[str(x) for x in t] for t in data if isinstance(t, (list, tuple)) and len(t) == 4]
    return out, True

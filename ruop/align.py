"""Привязка строки из ответа модели к подстроке текста: точная и нечёткая (как у Ватолина)."""
import re

from rapidfuzz import fuzz

from . import evaluation as ev

_WORD = re.compile(r"\w+", re.UNICODE)


def exact(text, sub):
    i = text.find(sub)
    return None if i < 0 or not sub else (i, i + len(sub))


def fuzzy(text, sub, threshold=70):
    """Лучший по fuzz.ratio отрезок текста с границами по словам; None, если ниже порога."""
    hit = exact(text, sub)
    if hit:
        return hit
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    n_sub = max(1, len(_WORD.findall(sub)))
    best, span = 0, None
    for i, (b, _) in enumerate(words):
        for j in range(i, min(len(words), i + n_sub + 3)):
            e = words[j][1]
            score = fuzz.ratio(sub.lower(), text[b:e].lower())
            if score > best:
                best, span = score, (b, e)
    return span if best >= threshold else None


def build_opinions(text, tuples, mode="exact"):
    """Кортежи из ответа модели → кортежи в формате соревнования; невалидные отбрасываются."""
    find = fuzzy if mode == "fuzzy" else exact
    out, dropped = [], 0
    for src, tgt, exp, pol in tuples:
        pol = pol.strip().upper()
        src = src.strip()
        if pol not in ("POS", "NEG"):
            dropped += 1
            continue
        if src.upper() == "AUTHOR":
            s = [["AUTHOR"], ["NULL"]]
        elif src.upper() in ("NULL", "", "NONE"):
            s = [["NULL"], ["0:0"]]
        else:
            sp = find(text, src)
            s = [[text[sp[0]:sp[1]]], [f"{sp[0]}:{sp[1]}"]] if sp else None
        tp, xp = find(text, tgt.strip()), find(text, exp.strip())
        if not (s and tp and xp):
            dropped += 1
            continue
        op = {"Source": s,
              "Target": [[text[tp[0]:tp[1]]], [f"{tp[0]}:{tp[1]}"]],
              "Polar_expression": [[text[xp[0]:xp[1]]], [f"{xp[0]}:{xp[1]}"]],
              "Polarity": pol}
        # оценщик падает на пересекающихся выражениях при тех же источнике/объекте/полярности
        try:
            ev.convert_opinion_to_tuple({"sent_id": 0, "text": text, "opinions": out + [op]})
        except Exception:
            dropped += 1
            continue
        out.append(op)
    return out, dropped

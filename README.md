# ruopinionne-experiments

Эксперименты на [RuOpinionNE-2024](https://github.com/dialogue-evaluation/RuOpinionNE-2024) —
извлечение кортежей мнений (источник, объект, полярность, выражение) из русскоязычных новостей.
Сабмиты: [CodaLab 20244](https://codalab.lisn.upsaclay.fr/competitions/20244) (фаза post-evaluation).

## Структура

- `data/` — данные соревнования (копия из upstream, MIT, см. `data/LICENSE.upstream`)
- `ruop/evaluation.py` — официальный скрипт оценки (копия `codalab/evaluation.py`, без изменений)
- `ruop/data.py` — загрузка сплитов и `make_opinion()`: сборка кортежа из строк с вычислением смещений
- `scripts/score.py` — локальный P/R/F1 на размеченной валидации
- `scripts/make_submission.py` — zip для CodaLab
- `scripts/sanity.py` — проверка пайплайна
- `predictions/`, `submissions/` — результаты прогонов (не в git)

## Использование

```bash
python3 scripts/sanity.py
python3 scripts/score.py predictions/my_run.jsonl          # против validation_labeled
python3 scripts/make_submission.py predictions/my_test.jsonl
```

## Данные

| Сплит | Предложений | С мнениями | Кортежей | POS / NEG |
|---|---|---|---|---|
| train | 2556 | 1494 | 2904 | 1174 / 1730 |
| validation (labeled) | 1316 | 769 | 1612 | 753 / 859 |
| test (без разметки) | 803 | — | — | — |

Источник мнения: подстрока текста, `AUTHOR` (`[["AUTHOR"], ["NULL"]]`) или пусто (`[["NULL"], ["0:0"]]`).

## Метрика

Взвешенный F1 по кортежам: кортеж засчитывается, если совпала полярность и есть пересечение
хотя бы в один токен по источнику, объекту и выражению; вес — среднее долей пересечения по трём полям.

## Ориентиры (финальный этап, статья arXiv:2504.06947)

| Место | F1 | Подход |
|---|---|---|
| 1 | 0.41 | LLaMA-3.3-70B, QLoRA, дообучение |
| 2 | 0.35 | Qwen2.5-32B-Instruct, дообучение |
| 3 | 0.33 | Mistral Large 2 / GPT-4o, few-shot с ретривом примеров |
| baseline | 0.24 | Qwen2.5-32B-Instruct, 10-shot |

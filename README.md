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
- `ruop/prompt.py`, `ruop/parse.py`, `ruop/align.py` — промпт Россяйкина, разбор ответа, точная и нечёткая привязка подстрок
- `scripts/embed.py` — эмбеддинги SBERT для подбора примеров (кэш в `cache/`, не в git)
- `scripts/run_llm.py` — few-shot прогон через OpenRouter, сырые ответы в `runs/`
- `scripts/build_pred.py` — сырые ответы → предсказания и оценка
- `runs/`, `predictions/`, `submissions/` — сырые ответы моделей, предсказания, архивы для CodaLab

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

## Ориентиры (test F1)

| F1 | Кто | Подход | Ресурсы |
|---|---|---|---|
| 0.405 | Ватолин, 1 место | QLoRA (4 бита) на Llama-3.3-70B, 5 генераций + объединение (Combine), нечёткое выравнивание подстрок | 8×A100 80GB, ~1.5 ч обучения |
| 0.405 | Ватолин | то же на Vikhr-Nemo-12B, агрегация Most Common | 8×A100 |
| 0.363 | Россяйкин, 2 место в post-evaluation | deepseek-chat, 12-shot с подбором примеров по SBERT + в промпт ответы GPT-4o / Grok / Mistral | только API |
| 0.33–0.35 | Россяйкин, Stage 1 | один вызов LLM, 12-shot с подбором по SBERT | только API |
| 0.23–0.25 | Россяйкин, Stage 0 | 12-shot со случайными примерами | только API |
| 0.252 (val) | Ватолин | Llama-3.3-70B, промпт без дообучения | — |

Статьи: [Ватолин](https://dialogue-conf.org/wp-content/uploads/2025/04/VatolinA.104.pdf),
[Россяйкин](https://dialogue-conf.org/wp-content/uploads/2025/04/RossyaykinP.105.pdf),
[организаторы](https://arxiv.org/abs/2504.06947).

## Наши результаты

| Прогон | Сплит | Привязка | P | R | F1 |
|---|---|---|---|---|---|
| deepseek-chat, 12 похожих примеров из train (`ds_s1_val`) | валидация | точная | 0.368 | 0.306 | 0.334 |
| то же | валидация | нечёткая | 0.357 | 0.349 | 0.353 |
| то же + два правила про источник в инструкции (`ds_v2_val`) | валидация | нечёткая | 0.383 | 0.371 | 0.377 |
| правила + пул train и другая половина валидации, 12 примеров (`ds_v2_val_x12`) | валидация | нечёткая | 0.393 | 0.382 | 0.387 |
| то же, 24 примера (`ds_v2_val_x24`) | валидация | нечёткая | 0.402 | 0.366 | 0.383 |
| deepseek-chat, 12 примеров из train (`ds_s1_test`) | test (Codabench) | точная | — | — | 0.35 |
| то же | test (Codabench) | нечёткая | — | — | 0.3756 |
| то же, пул примеров train + валидация (`ds_s1_test_poolval`) | test (Codabench) | нечёткая | — | — | 0.3818 |
| то же + два правила про источник (`ds_v2_test_poolval`) | test (Codabench) | нечёткая | — | — | 0.3839 |

Оценка на test: [Codabench 18622](https://www.codabench.org/competitions/18622/), фаза post-evaluation
(старый CodaLab посылки больше не принимает). Платформа показывает F1 с двумя знаками.

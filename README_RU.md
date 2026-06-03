# dp-rag-eval

Фреймворк для оценки систем Differentially Private RAG. Интегрирует два оригинальных DP-RAG бейзлайна (Koga et al. и Grislain) и предоставляет датасет-модуль, который строит контролируемые версии корпуса для изучения того, как **дублирование документов ослабляет формальные гарантии DP**.

Что есть сейчас:
- **Врапперы бейзлайнов** — тонкие ре-экспорты DP-механизмов обоих бейзлайнов (`src/dp_rag_eval/wrappers/`).
- **Датасет-фреймворк** — загрузчик и четыре контролируемых варианта корпуса (`src/dp_rag_eval/datasets/`).
- **Патч-механизм** — локальные правки клонов бейзлайнов (например, смена модели) хранятся как патчи и накатываются автоматически.

## Бейзлайны

### Koga et al. — DPVoteRAG
- **Статья:** [arXiv:2412.04697](https://arxiv.org/abs/2412.04697)
- **Оригинальный репозиторий:** [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG)
- **Механизм:** Потокенное DP-голосование с Limited Domain Gumbel Mechanism
- **Модель-генератор:** `allenai/OLMo-2-1124-7B-Instruct` (через vLLM) — *дефолт upstream `facebook/opt-1.3b`*

### Grislain — DP-RAG
- **Статья:** [arXiv:2412.19291](https://arxiv.org/abs/2412.19291)
- **Оригинальный репозиторий:** [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag)
- **Механизм:** DP-агрегация логитов с клиппингом (кастомный `LogitsProcessor`)
- **Модель-генератор:** `allenai/OLMo-2-1124-7B-Instruct` (через HuggingFace Transformers) — *дефолт upstream `microsoft/Phi-3.5-mini-instruct`*

Оба бейзлайна переведены на один генератор (OLMo-2-7B-Instruct), чтобы при сравнении privacy–utility модель оставалась константой. Смена применяется через `patches/` поверх клонов (см. *Патчи бейзлайнов* ниже).

## Требования

- **Python 3.11+** (на 3.8 фреймворк не работает — нужны `datasets`/`faker` и синтаксис 3.11)
- CPU достаточно для тестов, датасет-фреймворка и демо DP-механизмов
- GPU с 16+ GB VRAM нужен для полного инференса LLM (генератор OLMo-2-7B)

## Установка

```bash
git clone https://github.com/KarinaMalyshkina/DP_RAG.git
cd DP_RAG

# Создать виртуальное окружение с Python 3.11
python3.11 -m venv .venv        # Windows: py -3.11 -m venv .venv

# Активировать
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Установить
pip install -e ".[dev]"

# Склонировать оригинальные репозитории бейзлайнов И наложить наши патчи (смена модели и т.д.)
python setup_baselines.py
```

## Подготовка датасета

Фреймворк загружает публичный корпус и строит четыре контролируемые версии:

| Вариант | Что делает |
|---|---|
| `original` | Контроль — все документы уникальны, без изменений |
| `full_duplicate` | Подмножество документов повторено **дословно** |
| `partial_overlap` | Подмножество документов **разделяет предложение/абзац** |
| `canary` | Уникальные секреты вставлены как **цели для extraction-атак** (с ground truth) |

Корпус по умолчанию: **HealthCareMagic** (`RafaelMPereira/HealthCareMagic-100k-Chat-Format-en`); документ — это ответ врача, вопрос пациента хранится в `metadata["query"]`.

```bash
python scripts/prepare_dataset.py --n-docs 1000
# ключевые опции: --dup-rate --dup-copies --overlap-rate --overlap-hosts
#                 --n-canaries --canary-repeats --seed --output-dir
```

Результат пишется в `data/variants/` (gitignored): `original.jsonl`, `full_duplicate.jsonl`, `partial_overlap.jsonl`, `canary.jsonl`, `canary_ground_truth.json`, `summary.json`.

## Запуск тестов

```bash
pytest tests/ -v
```

28 CPU-тестов — **без GPU, без сети, без загрузки моделей** (HuggingFace-загрузчик замокан):
- `test_core_types.py` — `Document` (roundtrip, независимость copy)
- `test_dataset_variants.py` — все 4 варианта: детерминизм, размеры, provenance, неизменность входа, дословность дубликатов, общие фрагменты, canary ground truth
- `test_dataset_io.py` — JSONL roundtrip, `to_texts`
- `test_loader.py` — парсинг диалогов и загрузчик HealthCareMagic (замокан)

## Демо

```bash
python demo.py
```

Прогоняет DP-механизмы обоих бейзлайнов (только CPU) — мажоритарное голосование, Limited Domain Gumbel, экспоненциальный механизм, потокенный ε и агрегация логитов Grislain.

## Патчи бейзлайнов

`baselines/` в gitignore (это клоны внешних репозиториев), поэтому локальные правки там не отслеживаются git и терялись бы при переклонировании. Они хранятся как патч-файлы и накатываются скриптом `setup_baselines.py`:

- `patches/koga.patch`, `patches/grislain.patch` — сейчас это смена генератора на OLMo-2-7B.
- Применение идемпотентно (повторный запуск выдаёт `already applied, skipping`).
- Если правишь клон вручную — пересоздай его патч:
  ```bash
  (cd baselines/koga && git diff) > patches/koga.patch
  (cd baselines/grislain && git diff) > patches/grislain.patch
  ```

## Структура проекта

```
DP_RAG/
├── baselines/                     # gitignored, клонируются + патчатся через setup_baselines.py
│   ├── koga/                      # tacchan7412/DPRAG
│   └── grislain/                  # sarus-tech/dp-rag
├── patches/                       # правки, накатываемые на клоны (смена модели)
│   ├── koga.patch
│   └── grislain.patch
├── src/dp_rag_eval/
│   ├── core/
│   │   ├── types.py               # Document
│   │   └── protocols.py           # DatasetLoader, DatasetVariant
│   ├── datasets/
│   │   ├── loaders.py             # HealthCareMagicLoader
│   │   ├── variants.py            # Original / FullDuplicate / PartialOverlap / CanaryInjection
│   │   └── io.py                  # JSONL save/load, to_texts
│   └── wrappers/
│       ├── koga_imports.py        # импорты из baselines/koga/
│       └── grislain_imports.py    # импорты из baselines/grislain/
├── scripts/
│   └── prepare_dataset.py         # сборка 4 вариантов корпуса
├── tests/                         # 28 CPU-тестов (без GPU/сети)
├── data/                          # gitignored: подготовленные варианты
├── setup_baselines.py
├── demo.py
└── pyproject.toml
```

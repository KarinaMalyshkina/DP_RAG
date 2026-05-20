# dp-rag-eval

Фреймворк для оценки систем Differentially Private RAG. Оборачивает два оригинальных DP-RAG бейзлайна (Koga et al. и Grislain) за общим интерфейсом для сравнительного анализа.

## Бейзлайны

### Koga et al. — DPVoteRAG
- **Статья:** [arXiv:2412.04697](https://arxiv.org/abs/2412.04697)
- **Оригинальный репозиторий:** [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG)
- **Механизм:** Потокенное DP-голосование с Limited Domain Gumbel Mechanism
- **Модель по умолчанию:** `facebook/opt-1.3b` (через vLLM)

### Grislain — DP-RAG
- **Статья:** [arXiv:2412.19291](https://arxiv.org/abs/2412.19291)
- **Оригинальный репозиторий:** [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag)
- **Механизм:** DP-агрегация логитов с клиппингом (кастомный LogitsProcessor)
- **Модель по умолчанию:** `microsoft/Phi-3.5-mini-instruct` (через HuggingFace Transformers)

## Требования

- Python 3.11+
- CPU достаточно для тестов и демо DP-механизмов
- GPU с 16+ GB VRAM нужен для полного инференса LLM

## Установка

```bash
git clone https://github.com/KarinaMalyshkina/DP_RAG.git
cd DP_RAG

# Создать виртуальное окружение с Python 3.11
python -m venv .venv

# Активировать (Windows)
.venv\Scripts\activate
# Активировать (Linux/Mac)
source .venv/bin/activate

# Установить
pip install -e ".[dev]"

# Склонировать оригинальные репозитории бейзлайнов
python setup_baselines.py
```

## Запуск тестов

```bash
pytest tests/ -v
```

14 CPU-тестов, покрывающих DP-механизмы обоих бейзлайнов:
- `test_koga.py` — LDGumbelMechanism, ExpMechanism, majority_vote, RDP-композиция
- `test_grislain.py` — DPGenerationConfig, DPLogitsAggregator, PUPVectorStoreConfig

## Демо

```bash
python demo.py
```

## Структура проекта

```
DP_RAG/
├── baselines/                     # gitignored, клонируются через setup_baselines.py
│   ├── koga/                      # tacchan7412/DPRAG
│   └── grislain/                  # sarus-tech/dp-rag
├── src/dp_rag_eval/
│   ├── __init__.py
│   └── wrappers/
│       ├── koga_imports.py        # импорты из baselines/koga/
│       └── grislain_imports.py    # импорты из baselines/grislain/
├── tests/
│   ├── conftest.py
│   ├── test_koga.py
│   └── test_grislain.py
├── setup_baselines.py
├── demo.py
└── pyproject.toml
```

# dp-rag-eval

Фреймворк для оценки систем Differentially Private RAG. Исследует, как дупликация документов в корпусе ослабляет формальные гарантии дифференциальной приватности через атаки на приватность.

## Требования

- **Python 3.11+**
- **GPU с 16+ GB VRAM** (для реальных экспериментов с LLM)
- Режим CPU работает для тестов и разработки

## Установка

```bash
# Клонировать репозиторий
git clone <repo-url>
cd dp-rag-eval

# Создать виртуальное окружение
python -m venv .venv

# Активировать (Linux/Mac)
source .venv/bin/activate
# Активировать (Windows)
.venv\Scripts\activate

# Установить пакет с dev-зависимостями
pip install -e ".[dev]"
```

### Скачать данные NQ (для пайплайна Koga)

```bash
bash scripts/download_data.sh
```

## Запуск тестов

Все тесты работают на CPU без скачивания больших моделей.

```bash
# Запустить все тесты
PYTHONPATH=src pytest tests/ -v

# Запустить конкретный файл
PYTHONPATH=src pytest tests/test_pipelines.py -v
PYTHONPATH=src pytest tests/test_attacks.py -v
PYTHONPATH=src pytest tests/test_datasets.py -v
PYTHONPATH=src pytest tests/test_dp_mechanisms.py -v

# Запустить конкретный тест
PYTHONPATH=src pytest tests/test_pipelines.py::TestKogaDPMechanisms::test_ld_gumbel_basic -v
```

### Линтинг и форматирование

```bash
# Проверить ошибки линтинга
ruff check src/ tests/

# Автоисправление
ruff check --fix src/ tests/

# Проверить форматирование
ruff format --check src/ tests/

# Применить форматирование
ruff format src/ tests/
```

## Запуск экспериментов

### Один эксперимент

```bash
python scripts/run_experiment.py --config configs/koga_nq.yaml
```

С переопределением параметров:

```bash
python scripts/run_experiment.py --config configs/koga_nq.yaml --privacy.epsilon 2.0 --experiment.seed 123
```

### Sweep по параметрам

```bash
python scripts/run_sweep.py \
  --config configs/koga_nq.yaml \
  --sweep "privacy.epsilon=[0.5,1,2,4,8]" \
  --sweep "experiment.seed=[42,43,44]"
```

### Ручное использование из Python

```python
from dp_rag_eval.datasets import SyntheticMedicalLoader, create_variant
from dp_rag_eval.attacks import CanaryExtractionAttack
from dp_rag_eval.evaluation import exact_match, f1_score

# 1. Загрузить данные
docs = SyntheticMedicalLoader(n_docs=200, seed=42).load()

# 2. Применить вариант корпуса (например, инъекция canary)
variant = create_variant("canary", {"n_canaries": 50, "copies_per_canary": 3})
docs = variant.apply(docs)

# 3. Создать пайплайн и проиндексировать
from tests.conftest import MockPipeline
pipeline = MockPipeline()
pipeline.index(docs)

# 4. Задать вопрос
result = pipeline.answer("What is the most common diagnosis?", eps=1.0, delta=1e-5)
print(result.text)
print(f"Потраченный бюджет приватности: {result.privacy_spent}")

# 5. Запустить атаку на приватность
attack = CanaryExtractionAttack()
attack.setup(pipeline, docs)
attack_result = attack.run(n_queries=50, eps=1.0, delta=1e-5)
print(f"Успешность извлечения canary: {attack_result.success_rate:.2%}")

# 6. Оценить качество ответов
em = exact_match("Paris", ["Paris", "paris"])
f1 = f1_score("the capital is Paris", ["Paris is the capital"])
print(f"EM={em}, F1={f1:.2f}")
```

## Пайплайны

### Koga (DPVoteRAG)

- **Модель:** `facebook/opt-1.3b` (через vLLM)
- **Механизм:** Потокенное DP-голосование с Limited Domain Gumbel Mechanism
- **Ретривер:** DPR (facebook/dpr-*-nq-base) — без DP-шума на этапе извлечения
- **Конфиг:** `configs/koga_nq.yaml`

### Grislain (DP-RAG)

- **Модель:** `microsoft/Phi-3.5-mini-instruct` (через HuggingFace Transformers)
- **Механизм:** DP-агрегация логитов с клиппингом (кастомный LogitsProcessor)
- **Ретривер:** PUP Vector Store с экспоненциальным механизмом (эмбеддинги Snowflake Arctic)
- **Конфиг:** `configs/grislain_medical.yaml`

## Смена модели

Отредактируйте поле `model` в YAML-конфиге:

```yaml
pipeline:
  type: "koga"                    # или "grislain"
  model: "facebook/opt-1.3b"     # любая HuggingFace causal LM
  model_revision: null            # привязка к конкретной ревизии (необязательно)
  device: "cuda"                  # "cuda" или "cpu"
```

### Совместимые модели

| Модель | Размер | VRAM | Примечания |
|--------|--------|------|------------|
| `sshleifer/tiny-gpt2` | 0.6M | <1 GB | Только для тестов |
| `facebook/opt-350m` | 350M | ~2 GB | Легковесная, быстрая итерация |
| `facebook/opt-1.3b` | 1.3B | ~4 GB | По умолчанию для Koga |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~8 GB | По умолчанию для Grislain |
| `meta-llama/Llama-2-7b-hf` | 7B | ~16 GB | Крупнее, лучше качество |

Подходит любая модель, совместимая с `AutoModelForCausalLM` из HuggingFace. Для Grislain модель должна поддерживать `.generate()` с кастомным `LogitsProcessor` (нужен доступ к сырым логитам). Для Koga модель должна поддерживаться vLLM.

Смена модели в коде:

```python
from dp_rag_eval.llm import LocalHFBackend, LocalVLLMBackend

# Для Grislain (HuggingFace Transformers)
backend = LocalHFBackend(model_name="microsoft/Phi-3.5-mini-instruct", device="cuda")

# Для Koga (vLLM)
backend = LocalVLLMBackend(model_name="facebook/opt-1.3b", device="cuda")
```

## Варианты корпуса

| Вариант | Описание | Ключ в конфиге |
|---------|----------|----------------|
| Original | Без изменений | `original` |
| Full Duplicate | Точные копии части документов | `full_duplicate` |
| Partial Overlap | Почти-дубликаты с минимальными изменениями текста | `partial_overlap` |
| Canary Injection | Инъекция документов с известными секретами | `canary` |

```yaml
dataset:
  loader: "synthetic_medical"     # или "nq"
  variant: "full_duplicate"
  variant_params:
    duplication_rate: 0.3         # доля документов для дупликации
    copies: 5                     # количество копий
```

## Атаки на приватность

| Атака | Что делает |
|-------|-----------|
| Canary Extraction | Вставляет секретные строки в корпус, затем пытается их извлечь через запросы |
| Membership Inference | Определяет, был ли конкретный документ в корпусе |
| Property Inference | Выводит агрегатные свойства корпуса (напр., самый частый диагноз) |

## Результаты

Сохраняются в `runs/<имя_эксперимента>/`:

- `meta.json` — метаданные оборудования и ПО
- `run_eps<X>_seed<Y>.json` — результаты для каждого прогона
- `summary.json` — агрегированные результаты по всем прогонам

## Структура проекта

```
dp-rag-eval/
├── src/dp_rag_eval/
│   ├── core/          # Типы, протоколы, конфигурация
│   ├── pipelines/     # Адаптеры Koga + Grislain
│   ├── attacks/       # Canary, MIA, property inference
│   ├── datasets/      # Загрузчики + варианты корпуса
│   ├── evaluation/    # Метрики + репортер
│   ├── llm/           # Бэкенды HF + vLLM
│   └── runner/        # Оркестратор экспериментов
├── configs/           # YAML-конфиги экспериментов
├── scripts/           # CLI-скрипты запуска
├── tests/             # Тесты (pytest)
└── pyproject.toml
```

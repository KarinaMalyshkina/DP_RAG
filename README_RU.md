# dp-rag-eval

Единый фреймворк для оценки дифференциально приватной генерации с поиском (DP-RAG). Он запускает два опубликованных механизма DP-RAG — потокенное голосование Koga et al. и агрегацию логитов Grislain — через общий интерфейс, калибрует оба под общий бюджет приватности и с помощью атак на приватность и метрик полезности измеряет, насколько **дублирование содержимого в корпусе для поиска ослабляет их формальные гарантии (ε, δ)** в контролируемых условиях корпуса.

Код сопровождает магистерскую диссертацию *A Unified Attack Evaluation Framework for Differentially Private RAG Systems* (Карина Малышкина, Венский университет, 2026).

## Что делает фреймворк

| Компонент | Где | Что делает |
|---|---|---|
| Пайплайны | `src/dp_rag_eval/pipelines/` | Naive RAG (точка отсчёта ε = ∞), Koga DPVoteRAG и Grislain DP-RAG за общим интерфейсом; DP-механизмы — оригинальный код, наши только загрузка моделей и индексирование |
| Калибровка приватности | `src/dp_rag_eval/privacy/` | Подбирает параметры каждого механизма его собственным аккаунтантом так, чтобы один полный ответ длиной `n_tokens` токенов (вместе с поиском) стоил заданный суммарный (ε, δ); недостижимые цели отмечаются, а не запускаются |
| Варианты корпуса | `src/dp_rag_eval/datasets/` | `original`, `full_duplicate` (дословные копии), `partial_overlap` (общие фрагменты), `canary` (внедрённые секреты) |
| Атаки | `src/dp_rag_eval/attacks/` | Вывод о членстве (Anderson et al., 2024) с единой оцениваемой величиной для всех пайплайнов; извлечение канареек (Secret Sharer, Carlini et al., 2019) |
| Полезность | `src/dp_rag_eval/evaluation/metrics.py` | Token F1, ROUGE-L, exact match, число пустых ответов |
| Прогоны | `src/dp_rag_eval/runner/`, `scripts/run_*.py` | Сетка сид × вариант корпуса × ε × пайплайн; каждая готовая ячейка сразу записывается, поэтому прерванный прогон продолжается с места остановки |
| Анализ | `src/dp_rag_eval/evaluation/reporter.py`, `scripts/report_degradation.py` | Разности «дублированный корпус минус контроль» при одинаковых пайплайне, ε и сиде, с усреднением по сидам |
| Мониторинг | `scripts/watch_progress.py`, `scripts/notify.py`, `scripts/telegram_bot.py` | Прогресс, оценка оставшегося времени и оповещения о зависании долгих прогонов на GPU |

## Бейзлайны

### Koga et al. — DPVoteRAG
- **Статья:** [arXiv:2412.04697](https://arxiv.org/abs/2412.04697)
- **Оригинальный репозиторий:** [tacchan7412/DPRAG](https://github.com/tacchan7412/DPRAG), закреплён на коммите `28c79e5`
- **Механизм:** найденные документы распределяются между голосующими; каждый токен ответа выбирается голосованием с Limited-Domain Gumbel; учёт приватности через RDP
- **Здесь:** голосование работает на HuggingFace Transformers (все голосующие одним батчем) вместо vLLM; число голосующих подбирается под бюджет так, чтобы цель была достижима

### Grislain — DP-RAG
- **Статья:** [arXiv:2412.19291](https://arxiv.org/abs/2412.19291)
- **Оригинальный репозиторий:** [sarus-tech/dp-rag](https://github.com/sarus-tech/dp-rag), закреплён на коммите `b9d1dea`
- **Механизм:** DP-порог поиска через экспоненциальный механизм, затем агрегация логитов по документам с ограничением вклада; учёт приватности через PLD (`dp-accounting`)

## Модели и данные

| Роль | Модель / датасет |
|---|---|
| Генератор (все эксперименты) | `Qwen/Qwen3.5-9B` |
| Второй генератор (репликация, Grislain) | `google/gemma-4-12B-it` |
| Эмбеддер для поиска | `sentence-transformers/all-MiniLM-L6-v2` |
| Корпус | HealthCareMagic (`RafaelMPereira/HealthCareMagic-100k-Chat-Format-en`): ответы врачей — документы, вопросы пациентов — запросы |

Генератор задаётся для каждого прогона параметром `--generator`; режим рассуждений отключается через шаблон чата.

## Требования

- Python 3.11+ и [uv](https://docs.astral.sh/uv/) (версии зависимостей закреплены в `uv.lock`)
- Для тестов и демо достаточно CPU
- Для экспериментов нужен GPU; заявленные прогоны выполнены на одной NVIDIA RTX A6000 (48 ГБ). `pyproject.toml` берёт torch из индекса под CUDA 12.8

## Установка

```bash
git clone https://github.com/KarinaMalyshkina/DP_RAG.git
cd DP_RAG
uv sync --extra dev
uv run python setup_baselines.py   # клонирует оба бейзлайна на закреплённых коммитах и накладывает patches/
```

На новой GPU-машине вместо этого используется скрипт подготовки; он также скачивает веса и сразу падает, если torch собран без CUDA, модель загрузилась в fp32 или режим рассуждений не отключился:

```bash
bash scripts/bootstrap_gpu.sh
uv run python scripts/verify_optimizations.py --generator Qwen/Qwen3.5-9B
```

Полный порядок работы с матрицей экспериментов — калибровка, прогоны, интерпретация, восстановление после сбоя — описан в [`docs/RUNBOOK_GPU.md`](docs/RUNBOOK_GPU.md).

## Запуск экспериментов

```bash
# вывод о членстве на чистом корпусе по оси бюджета
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/mia_main \
  --variant-grid base --epsilons inf,0.5,1,2,4,8 \
  --n-tokens 32 --delta 1e-6 --modes unified \
  --members 1000 --non-members 1000 --targets-per-class 250 \
  --seeds 42,43,44

# сетка дублирования — в ту же папку, чтобы у каждой ячейки был контроль
uv run python scripts/run_attack.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/mia_main \
  --variant-grid duplication --epsilons inf,1,5 \
  --n-tokens 32 --delta 1e-6 --modes unified \
  --members 1000 --non-members 1000 --targets-per-class 250 \
  --seeds 42,43,44

# извлечение канареек
uv run python scripts/run_canary.py \
  --generator Qwen/Qwen3.5-9B --output-dir runs/canary_main \
  --epsilons inf,0.5,1,2,5 --n-tokens 32 --delta 1e-6 \
  --canary-repeats 1,2,5,10 --variants original \
  --corpus-size 1000 --n-canaries 20 --n-distractors 255 \
  --modes exposure --seeds 42,43,44

# полезность на тех же конфигурациях
uv run python scripts/run_utility.py --generator Qwen/Qwen3.5-9B --output-dir runs/utility_main \
  --epsilons inf,1,5 --n-tokens 32 --delta 1e-6 --seeds 42,43,44

# дублированный корпус против контроля, с усреднением по сидам
uv run python scripts/report_degradation.py --results runs/mia_main \
  --metric auc --metric tpr_at_fpr_05 --output-dir runs/analysis
```

Каждый прогон пишет `results.json`, `results.csv` и `report.md` в свою папку и обновляет их после каждой ячейки. `scripts/report_feasibility.py` выводит, сколько голосующих нужно Koga для каждого бюджета; `scripts/probe_generator.py` проверяет новый генератор перед прогоном.

Результаты основных прогонов на `Qwen/Qwen3.5-9B` лежат в `runs/` (`mia_main`, `calib`, `calib2`); остальные папки там — небольшие пробные прогоны на CPU.

## Тесты

```bash
uv run pytest tests/ -v
```

245 тестов на CPU, без GPU и без скачивания моделей. Тесты интеграции с Koga и Grislain импортируют оригинальный код, поэтому сначала выполните `setup_baselines.py`.

```bash
uv run python demo.py   # DP-механизмы обоих бейзлайнов на игрушечных входных данных, только CPU
```

## Патчи бейзлайнов

`baselines/` содержит клоны внешних репозиториев и не отслеживается git. Локальные изменения в них хранятся в `patches/koga.patch` и `patches/grislain.patch` и накладываются скриптом `setup_baselines.py`; если патч не накладывается, установка останавливается. Сами механизмы патчи не меняют: они отключают режим рассуждений генератора, кодируют корпус порциями, позволяют выбирать устройство без GPU, меняют примеры чекпойнтов в upstream и считают порог поиска Grislain в логарифмической шкале, чтобы на полностью дублированном корпусе не было потери точности.

После правки клона обновите его патч:

```bash
(cd baselines/koga && git diff) > patches/koga.patch
(cd baselines/grislain && git diff) > patches/grislain.patch
```

## Документация

| Файл | Содержание |
|---|---|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Определения: единица приватности, общая ось бюджета, калибровка, оцениваемые величины, метрики |
| [`docs/RUNBOOK_GPU.md`](docs/RUNBOOK_GPU.md) | Порядок работы с матрицей экспериментов на GPU-машине |
| [`docs/decisions-needed.md`](docs/decisions-needed.md) | Открытые проектные решения |
| [`docs/empirical-coverage-report.md`](docs/empirical-coverage-report.md) | Что код измеряет по сравнению с тем, что утверждается в диссертации |
| [`docs/Literature_Attacks.txt`](docs/Literature_Attacks.txt) | Литература по атакам |

## Структура проекта

```
DP_RAG/
├── baselines/                 # не отслеживается: клонируется и патчится setup_baselines.py
├── patches/                   # koga.patch, grislain.patch
├── src/dp_rag_eval/
│   ├── core/                  # Document и другие общие типы, протоколы
│   ├── datasets/              # загрузчик HealthCareMagic, варианты корпуса, JSONL
│   ├── pipelines/             # naive, Koga, Grislain; общая загрузка моделей
│   ├── privacy/               # калибровка под общий (ε, δ) на ответ
│   ├── attacks/               # вывод о членстве, извлечение канареек
│   ├── evaluation/            # метрики полезности, отчёт о деградации, чтение прогресса
│   ├── runner/                # развёртка сетки и возобновляемая запись результатов
│   ├── wrappers/              # импорт из baselines/koga и baselines/grislain
│   └── notifications.py       # отправка оповещений о прогонах в Telegram
├── scripts/                   # run_attack, run_canary, run_utility, report_*, bootstrap_gpu, ...
├── tests/                     # 245 тестов на CPU
├── runs/                      # результаты экспериментов
├── docs/
├── setup_baselines.py
├── demo.py
├── pyproject.toml
└── uv.lock
```

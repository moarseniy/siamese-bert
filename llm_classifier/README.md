# LLM Survey Classifier

Классификация русскоязычных ответов из опросов через локальный Qwen3.5,
поднятый в `vllm serve`. Каждый ответ отправляется отдельным OpenAI-compatible
Chat Completions запросом. Несколько независимых запросов выполняются
параллельно, чтобы vLLM мог использовать внутренний continuous batching.

## Установка

```bash
cd /home/arseniy/siamese-bert/llm_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск vLLM

Сервер запускается отдельно. Пример для текстовой Qwen3.5:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --port 8000 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-prefix-caching
```

Для Qwen3.5 может потребоваться актуальная/nightly версия vLLM. Размер модели,
tensor parallel и параметры памяти выбираются под доступные GPU.

`--enable-prefix-caching` полезен здесь, потому что одинаковый системный prompt
со справочником передаётся в каждом запросе.

## Входные данные

Поддерживаются `.xlsx`, `.xlsm` и `.csv`.

- `Ответ` — текст для классификации;
- `Коды_новые` — необязательная эталонная разметка для расчёта статистики.

Справочник имеет прежний формат:

```text
A. Финансы
A1. Зарплата
A2. Премии
B. Условия труда
B1. Рабочее место
```

LLM разрешено возвращать только коды подкатегорий из справочника либо
`UNKNOWN`.

## Классификация

```bash
python scripts/classify.py \
  --input ../data/answers.xlsx \
  --output output/predictions.xlsx \
  --codebook-txt ../data/codes.txt \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-9B \
  --concurrency 8 \
  --text-col "Ответ" \
  --gold-col "Коды_новые"
```

`--model` можно не указывать: тогда используется первая модель из `/v1/models`.

По умолчанию:

- один ответ отправляется одним запросом, prompt batching не используется;
- одновременно выполняется до 8 независимых запросов;
- Qwen thinking отключён через
  `chat_template_kwargs.enable_thinking=false`;
- ответ ограничен JSON Schema `{"codes":[...]}` с максимумом 6 кодов;
- `temperature=0`;
- при ошибке выполняется до двух повторов;
- ответ ограничен 64 токенами;
- каждые 250 завершённых строк сохраняется checkpoint в выходной файл.

Thinking можно включить через `--enable-thinking`. Для старой версии vLLM без
JSON Schema используйте `--no-structured-output`.

## Скорость

`--concurrency` — главный параметр производительности. Начните с `8`, затем
сравните `16` и `32`, следя за загрузкой GPU, latency и отсутствием OOM:

```bash
python scripts/classify.py ... \
  --concurrency 16 \
  --checkpoint-every 500 \
  --max-tokens 64
```

Каждый ответ по-прежнему обрабатывается отдельным запросом. Параллельность лишь
позволяет vLLM одновременно обслуживать несколько запросов. Для старого
строго последовательного режима укажите `--concurrency 1`.

Checkpoint можно полностью отключить через `--checkpoint-every 0`. Для больших
XLSX это заметно уменьшает лишние записи на диск.

## Результаты

В выходной таблице появляются:

- `predicted_codes`;
- `needs_review`;
- `invalid_codes`;
- `llm_error`;
- `latency_seconds`;
- `prompt_tokens`, `completion_tokens`;
- `raw_response`.

LLM генерирует только поле `codes`. Колонки `needs_review` и `invalid_codes`
вычисляются программно: review включается для `UNKNOWN`, ошибки запроса или
кода вне справочника.

Рядом сохраняются:

- `predictions_stats.json` — скорость, ошибки, токены и общие метрики;
- `predictions_per_class.csv` — precision/recall/F1 по кодам;
- `predictions_errors.csv` — строки с несовпавшей разметкой.

Метрики качества рассчитываются автоматически, только если вход содержит
колонку `Коды_новые`. Поддерживаются multi-label micro/macro F1, precision,
recall, exact match, hamming loss и top-1 accuracy для single-label строк.

В `predictions_stats.json` также записываются `wall_time_seconds`,
`throughput_rows_per_second`, выбранный `concurrency` и latency запросов.

# RuBERT Survey Classifier

Пайплайн supervised-обучения BERT для multi-label классификации коротких
русскоязычных ответов из опросов.

По умолчанию используется
[`DeepPavlov/rubert-base-cased`](https://huggingface.co/DeepPavlov/rubert-base-cased).
Можно передать любую совместимую модель из Hugging Face или путь к локальной
директории.

В отличие от `survey_classifier`, этот проект не строит embedding-индекс.
RuBERT получает текст и сразу возвращает вероятность каждого кода через
классификационную голову.

## Установка

```bash
cd bert_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

CSV/XLSX для обучения:

| Ответ | Коды_новые |
|---|---|
| Быстро оформили заказ | A1 |
| Вежливо, но очень долго | A1, A2 |

- `Ответ` содержит текст.
- `Коды_новые` содержит один или несколько кодов через запятую или `;`.
- Строки без текста и строки только с `UNKNOWN` не участвуют в обучении.
- Опционально можно передать колонку с вопросом или другим контекстом.

TXT-справочник:

```text
A. Сервис
A1. Скорость обслуживания
A2. Вежливость сотрудников
B. Продукт
B1. Качество продукта
```

Похожие кириллические буквы в кодах нормализуются: например, `А1` станет `A1`.

## Обучение

Минимальный запуск:

```bash
python scripts/train.py \
  --train data/train.csv \
  --codebook data/codes.txt \
  --out-dir model_out \
  --device cuda
```

Для Excel команда такая же:

```bash
python scripts/train.py \
  --train data/train.xlsx \
  --codebook data/codes.txt \
  --out-dir model_out \
  --epochs 3 \
  --batch-size 16 \
  --max-length 128 \
  --val-size 0.1 \
  --test-size 0.1 \
  --seed 42 \
  --device cuda
```

Если в данных есть колонка с текстом вопроса:

```bash
python scripts/train.py \
  --train data/train.xlsx \
  --codebook data/codes.txt \
  --out-dir model_out \
  --context-col "Вопрос" \
  --device cuda
```

Локальная базовая модель:

```bash
python scripts/train.py \
  --train data/train.xlsx \
  --codebook data/codes.txt \
  --out-dir model_out \
  --base-model /path/to/rubert-base-cased \
  --device cuda
```

Пайплайн:

1. Загружает и проверяет разметку.
2. Один раз делит исходные строки на `train`, `val`, `test` с `seed=42`.
3. Обучает multi-label голову с `BCEWithLogitsLoss`.
4. Подбирает порог на `val` по `micro-F1` или `macro-F1`.
5. Считает финальные метрики на `test` и сохраняет лучшую модель.

Один ответ используется один раз, но его target может содержать несколько
единиц, например `[A1=1, A2=1, B1=0]`. Все отсутствующие у строки классы
являются негативными примерами. Для редких положительных классов по умолчанию
применяются веса; отключить их можно флагом `--no-class-weights`.

## Память и скорость

Если не хватает VRAM:

```bash
python scripts/train.py \
  --train data/train.xlsx \
  --codebook data/codes.txt \
  --out-dir model_out \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-length 96 \
  --device cuda
```

Это дает эффективный batch `4 * 4 = 16`. Mixed precision на CUDA включается
автоматически. Для полного FP32 есть `--no-mixed-precision`.

Полезные параметры:

- `--epochs 3` - максимальное число эпох;
- `--early-stopping-patience 2` - остановка без улучшения на `val`;
- `--learning-rate 2e-5` - learning rate;
- `--threshold-metric micro_f1` - выбор порога;
- `--max-labels 6` - максимум кодов в одном предсказании;
- `--device cuda:1` - конкретная GPU.

## Результаты

```text
model_out/
  model/                       # tokenizer и лучший checkpoint
  splits/
    train.csv
    val.csv
    test.csv
    split_assignments.csv
  classifier_config.json       # классы, порог и настройки инференса
  codebook.csv
  label_distribution.csv       # частоты кодов в train/val/test
  metrics.json
  val_metrics.json
  test_metrics.json
  val_per_class.csv
  test_per_class.csv
  val_predictions.csv
  test_predictions.csv
  val_errors.csv
  test_errors.csv
  threshold_search.csv
  training_history.csv
```

Главная оценка для сравнения моделей находится в `test_metrics.json`.
`micro_f1` сильнее отражает качество частых классов, а `macro_f1` одинаково
учитывает частые и редкие классы. Детальный разбор каждого кода находится в
`test_per_class.csv`, ошибки - в `test_errors.csv`.

## Классификация

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input data/new_answers.xlsx \
  --output predictions.xlsx \
  --text-col "Ответ" \
  --batch-size 64 \
  --device cuda
```

Порог, `max_labels` и контекст берутся из артефакта либо переопределяются:

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input data/new_answers.csv \
  --output predictions.csv \
  --context-col "Вопрос" \
  --threshold 0.35 \
  --max-labels 6 \
  --margin-threshold 0.05 \
  --device cuda
```

К исходным данным добавляются:

- `predicted_codes` и `predicted_names`;
- `predicted_parent_codes` и `predicted_parent_names`;
- `confidence` и `margin`;
- `top_candidates`;
- `needs_review` - `True`, если ни один код не прошел порог или `margin` ниже
  `--margin-threshold`.

Если файл уже размечен, можно сразу посчитать качество:

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input data/check.xlsx \
  --output check_predictions.xlsx \
  --gold-codes-col "Коды_новые" \
  --device cuda
```

Рядом появятся `check_predictions_stats.json`,
`check_predictions_per_class.csv` и `check_predictions_errors.csv`.

## Проверки

```bash
pytest -q
```

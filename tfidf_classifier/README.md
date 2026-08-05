# TF-IDF Survey Classifier

Самостоятельный CPU-пайплайн для multi-label классификации русскоязычных
ответов из опросов.

Модель объединяет:

- словные TF-IDF n-граммы `1-2`;
- символьные TF-IDF n-граммы `3-5`;
- One-vs-Rest Logistic Regression с балансировкой классов.

Символьные признаки помогают с короткими ответами, русскими словоформами и
опечатками. GPU, Hugging Face и embedding-индекс не нужны.

## Установка

```bash
cd /home/arseniy/siamese-bert/tfidf_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

Поддерживаются CSV, XLSX и XLSM. По умолчанию используются колонки:

- `Ответ` - текст ответа;
- `Коды_новые` - один или несколько кодов через запятую или `;`.

Пример:

| Ответ | Коды_новые |
|---|---|
| Маленькая зарплата | A1 |
| Низкая зарплата и холодный офис | A1, B1 |

CSV-справочник с обязательными колонками `Код`, `Категория`, `Подкатегория`:

```csv
Код,Категория,Подкатегория
A1,Финансы,Зарплата
B1,Условия труда,Рабочее место
```

В справочнике перечисляются только конечные коды; отдельные строки `A`, `B`
не нужны.

Строки только с `UNKNOWN` не участвуют в обучении. Похожие кириллические буквы
в кодах нормализуются, например `А1` превращается в `A1`.

## Обучение

```bash
python scripts/train.py \
  --train data/train.xlsx \
  --codebook data/codes.csv \
  --out-dir model_out \
  --val-size 0.1 \
  --test-size 0.1 \
  --seed 42
```

Для CSV команда такая же:

```bash
python scripts/train.py \
  --train data/train.csv \
  --codebook data/codes.csv \
  --out-dir model_out
```

Если есть колонка с вопросом или другим контекстом:

```bash
python scripts/train.py \
  --train data/train.xlsx \
  --codebook data/codes.csv \
  --out-dir model_out \
  --context-col "Вопрос"
```

Пайплайн фиксированно делит исходные строки на `train/val/test`, обучается на
`train`, выбирает порог на `val` и только затем считает финальные метрики на
`test`.

Основные настройки:

- `--min-df 2` - минимальная частота n-граммы;
- `--classifier-c 4.0` - регуляризация Logistic Regression;
- `--word-max-features 100000`;
- `--char-max-features 150000`;
- `--threshold-metric micro_f1` или `macro_f1`;
- `--max-labels 6` - максимум кодов в одном ответе;
- `--n-jobs -1` - использовать все CPU.

## Артефакты

```text
model_out/
  tfidf_model.joblib
  tfidf_config.json
  codebook.csv
  label_distribution.csv
  metrics.json
  train_metrics.json
  val_metrics.json
  test_metrics.json
  train_per_class.csv
  val_per_class.csv
  test_per_class.csv
  train_predictions.csv
  val_predictions.csv
  test_predictions.csv
  val_errors.csv
  test_errors.csv
  threshold_search.csv
  splits/
    train.csv
    val.csv
    test.csv
    split_assignments.csv
```

Для сравнения с BERT/LLM в первую очередь смотрите `test_metrics.json`,
`test_per_class.csv` и `test_errors.csv`.

## Классификация

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input data/new_answers.xlsx \
  --output predictions.xlsx \
  --text-col "Ответ"
```

Можно переопределить выбранный на `val` порог:

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input data/new_answers.csv \
  --output predictions.csv \
  --threshold 0.35 \
  --max-labels 6
```

В результат добавляются коды, названия, основные категории, confidence,
margin, top-кандидаты и `needs_review`.

Для размеченного файла можно сразу посчитать качество:

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input data/check.xlsx \
  --output check_predictions.xlsx \
  --gold-codes-col "Коды_новые"
```

Рядом сохранятся:

- `check_predictions_stats.json`;
- `check_predictions_per_class.csv`;
- `check_predictions_errors.csv`.

Артефакты `tfidf_model.joblib`, ранее обученные внутри `survey_classifier`,
можно передавать новому `scripts/predict.py` без переобучения.

## Проверки

```bash
pytest -q
```

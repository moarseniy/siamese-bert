# Survey Classifier

Production pipeline для multi-label классификации текстовых ответов из опросов по изменяемому справочнику категорий.

Проект обучает локальную `sentence-transformers` модель, строит индекс исторических примеров и центроидов категорий, а затем классифицирует новые Excel-файлы.

## Установка

```bash
cd survey_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Базовая модель по умолчанию: `BAAI/bge-m3`.

## Формат данных

Исторический Excel для обучения должен содержать:

- `Ответ` - текст ответа;
- `Коды_новые` - один или несколько кодов через запятую, например `A1, A2`.

TXT-справочник:

```text
А. Финансы
А1. Зарплата
А2. Премии
```

Коды нормализуются: пробелы удаляются, регистр приводится к верхнему, похожие кириллические буквы заменяются на латиницу (`А -> A`, `В -> B` и т.д.). Код `UNKNOWN` исключается из обучения и не становится обычной категорией.

## Полное обучение

```bash
python scripts/train_pipeline.py \
  --train-xlsx train.xlsx \
  --codebook-txt codes.txt \
  --out-dir model_out \
  --epochs 1 \
  --batch-size 16 \
  --min-class-size 10 \
  --max-pairs-per-code 5000
```

Pipeline делает три шага:

1. Загружает Excel и TXT-справочник, разворачивает multi-label ответы в long-format.
2. Дообучает siamese/bi-encoder модель на positive pairs: два разных ответа с одной общей подкатегорией.
3. Строит production index: embeddings исторических ответов, metadata, центроиды подкатегорий, центроиды родительских категорий и codebook.

Итоговая структура:

```text
model_out/
  model/
  train_config.json
  training_pairs.csv
  index/
    example_embeddings.npy
    example_metadata.csv
    subcategory_centroids.npy
    subcategory_metadata.csv
    parent_centroids.npy
    parent_metadata.csv
    codebook.csv
    index_config.json
```

Все CSV сохраняются в `utf-8-sig`.

## Только построить индекс

Если модель уже обучена и лежит в `model_out/model`:

```bash
python -m src.build_index \
  --train-xlsx train.xlsx \
  --codebook-txt codes.txt \
  --out-dir model_out \
  --batch-size 64
```

Можно явно указать директорию модели:

```bash
python -m src.build_index \
  --train-xlsx train.xlsx \
  --codebook-txt codes.txt \
  --out-dir model_out \
  --model-dir model_out/model
```

## Классификация нового Excel

```bash
python -m src.predict \
  --model-dir model_out \
  --input-xlsx new_survey.xlsx \
  --output-xlsx predictions.xlsx \
  --text-col "Ответ" \
  --top-k 5 \
  --threshold 0.65 \
  --margin-threshold 0.05
```

То же через script wrapper:

```bash
python scripts/predict_excel.py \
  --model-dir model_out \
  --input-xlsx new_survey.xlsx \
  --output-xlsx predictions.xlsx
```

Выходной Excel содержит исходные колонки и новые:

- `predicted_codes`;
- `predicted_names`;
- `parent_codes`;
- `confidence`;
- `needs_review`;
- `top_candidates`;
- `nearest_examples`.

## Параметры инференса

`top_k` - сколько ближайших подкатегорий показывать в диагностике `top_candidates`.

`threshold` - минимальная cosine similarity для принятия категории. Если ни одна подкатегория из `top_k` не проходит порог, ответ получает `UNKNOWN`.

`margin_threshold` - минимальный разрыв между первой и второй подкатегорией. Маленький margin означает, что модель сомневается между похожими кодами.

`needs_review` становится `True`, если:

- top-1 similarity ниже `threshold`;
- margin ниже `margin_threshold`;
- не найдено ни одной реальной категории выше порога.

`confidence` - similarity лучшего кандидата. Для multi-label ответа возвращаются все кандидаты из `top_k`, у которых similarity не ниже `threshold`.

## Python API

```python
from src.classifier import SurveyClassifier

classifier = SurveyClassifier.load("model_out")
prediction = classifier.predict_one(
    "низкая зарплата и нет премии",
    top_k=5,
    threshold=0.65,
    margin_threshold=0.05,
)
print(prediction["predicted_codes"])
```

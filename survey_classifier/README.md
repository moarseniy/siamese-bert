# Survey Classifier

Production pipeline для multi-label классификации текстовых ответов из опросов по изменяемому справочнику категорий.

Проект обучает локальную `sentence-transformers` модель, строит индекс
исторических примеров и центроидов категорий, а затем классифицирует CSV/XLSX.

## Установка

```bash
cd survey_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Базовая модель по умолчанию: `BAAI/bge-m3`.

Для русскоязычных данных также поддерживается `ai-forever/FRIDA`. У этой модели
для тематической группировки используется встроенный prompt `categorize_topic`.

## Формат данных

Исторический CSV/XLSX для обучения должен содержать:

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
python scripts/train.py \
  --train train.xlsx \
  --codebook codes.txt \
  --out-dir model_out \
  --training-mode contrastive \
  --epochs 1 \
  --batch-size 16 \
  --val-size 0.1 \
  --test-size 0.1 \
  --seed 42 \
  --min-class-size 10 \
  --max-pairs-per-code 5000 \
  --negative-ratio 1.0
```

Обучение FRIDA из Hugging Face:

```bash
python scripts/train.py \
  --train train.xlsx \
  --codebook codes.txt \
  --out-dir model_out_frida \
  --base-model ai-forever/FRIDA \
  --prompt-name categorize_topic \
  --training-mode contrastive \
  --epochs 1 \
  --batch-size 8 \
  --seed 42
```

Если модель уже скачана, вместо имени репозитория укажите локальную директорию:

```bash
--base-model /path/to/FRIDA
```

`--prompt-name categorize_topic` разрешается через встроенный словарь prompts
модели. Полученный prefix сохраняется в `train_config.json` и
`index/index_config.json`, поэтому автоматически и одинаково применяется при
обучении, построении индекса и последующем инференсе. В `predict` повторно
указывать prompt не нужно.

Для модели без встроенного prompt можно передать собственную строку через
`--input-prefix "categorize_topic: "`. Параметры `--prompt-name` и
`--input-prefix` взаимоисключающие.

Pipeline делает четыре шага:

1. Загружает Excel и TXT-справочник, разворачивает multi-label ответы в long-format.
2. Делит данные на `train` / `val` / `test` по `row_id`, чтобы строки одного multi-label ответа не попадали в разные split. По умолчанию используется `seed=42`.
3. Дообучает siamese/bi-encoder модель выбранным режимом обучения.
4. Строит production index: embeddings исторических ответов, metadata, центроиды подкатегорий, центроиды родительских категорий и codebook.

По умолчанию `--index-split train`, то есть index строится только на train split без leakage из val/test. Для финального production-переобучения можно указать `--index-split all`.

## Режимы обучения

`--training-mode contrastive` - режим по умолчанию. Генерирует positive pairs внутри одного кода и explicit negative pairs из ответов, где этого кода нет. Используется `CosineSimilarityLoss` с label `1.0` для positive и `0.0` для negative. Размер негативов регулируется:

```bash
--negative-ratio 1.0
--max-negatives-per-code 5000
```

`--training-mode mnrl` - прежний режим на positive pairs с `MultipleNegativesRankingLoss`. Явных негативных строк нет, но остальные элементы batch работают как in-batch negatives.

```bash
python scripts/train.py \
  --train train.xlsx \
  --codebook codes.txt \
  --out-dir model_out \
  --training-mode mnrl
```

`--training-mode triplet` - triplet learning: `anchor`, `positive` из того же кода, `negative` из ответа без этого кода. Используется `TripletLoss` с cosine distance.

```bash
python scripts/train.py \
  --train train.xlsx \
  --codebook codes.txt \
  --out-dir model_out \
  --training-mode triplet \
  --max-triplets-per-code 5000 \
  --triplet-margin 0.5
```

Итоговая структура:

```text
model_out/
  model/
  splits/
    train.csv
    val.csv
    test.csv
    split_assignments.csv
    split_summary.json
  train_config.json
  training_pairs.csv      # для mnrl/contrastive
  training_triplets.csv   # для triplet
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

`val.csv` и `test.csv` пока сохраняются как holdout-наборы для проверки качества и подбора порогов. Само вычисление метрик можно добавить отдельным `eval`-инструментом поверх этих файлов и готового index.

## Только построить индекс

Если модель уже обучена и лежит в `model_out/model`:

```bash
python -m src.build_index \
  --train train.xlsx \
  --codebook codes.txt \
  --out-dir model_out \
  --batch-size 64
```

Можно явно указать директорию модели:

```bash
python -m src.build_index \
  --train train.xlsx \
  --codebook codes.txt \
  --out-dir model_out \
  --model-dir model_out/model
```

## Классификация

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input new_survey.xlsx \
  --output predictions.xlsx \
  --text-col "Ответ" \
  --top-k 5 \
  --threshold 0.65 \
  --max-labels 6 \
  --margin-threshold 0.05
```

Для размеченного файла можно сразу посчитать метрики:

```bash
python scripts/predict.py \
  --model-dir model_out \
  --input check.xlsx \
  --output check_predictions.xlsx \
  --gold-codes-col "Коды_новые"
```

Выходная таблица содержит исходные колонки и новые:

- `predicted_codes`;
- `predicted_names`;
- `predicted_parent_codes`;
- `predicted_parent_names`;
- `confidence`;
- `margin`;
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

## Визуализация index

После построения index можно сохранить 2D-проекцию embeddings исторических ответов или центроидов:

```bash
python -m src.visualize \
  --model-dir model_out \
  --output-dir model_out/reports \
  --target examples \
  --method pca \
  --color-by code \
  --sample-size 5000 \
  --seed 42
```

`--target` определяет, какие точки рисовать:

- `examples` - все исторические ответы из `example_embeddings.npy`;
- `subcategory-centroids` - только центроиды подкатегорий из `subcategory_centroids.npy`;
- `parent-centroids` - только центроиды основных/родительских категорий из `parent_centroids.npy`.

Примеры:

```bash
# Все ответы, раскраска по подкатегории
python -m src.visualize \
  --model-dir model_out \
  --target examples \
  --color-by code

# Все ответы, раскраска по основной категории
python -m src.visualize \
  --model-dir model_out \
  --target examples \
  --color-by parent_code

# Только центроиды подкатегорий
python -m src.visualize \
  --model-dir model_out \
  --target subcategory-centroids \
  --color-by parent_code

# Только центроиды основных категорий
python -m src.visualize \
  --model-dir model_out \
  --target parent-centroids \
  --color-by code
```

Отдельный режим single-label:

```bash
python -m src.visualize \
  --model-dir model_out \
  --target examples \
  --single-label-only \
  --color-by code
```

В этом режиме используются только ответы, у которых в колонке `codes` ровно один код. Для `subcategory-centroids` и `parent-centroids` центроиды пересчитываются заново только по таким single-label сэмплам.

Также доступен script wrapper:

```bash
python scripts/visualize_index.py \
  --model-dir model_out \
  --method tsne \
  --color-by parent_code
```

Артефакты:

- `model_out/reports/embedding_projection_<target>_<method>.csv` - координаты `x`, `y` и metadata строк;
- `model_out/reports/embedding_projection_<target>_<method>.html` - standalone SVG scatter plot, раскрашенный по `code`, `parent_code` или другой колонке metadata.

Если указан `--single-label-only`, к имени файла добавляется суффикс `_single_label`.

### Визуализация исходного CSV на базовой модели

Для этого индекс и обученная модель не нужны. Команда читает исходный CSV,
оставляет только строки ровно с одним кодом и считает их эмбеддинги локальной
моделью, скачанной из Hugging Face:

```bash
python scripts/visualize_csv.py \
  --input data/answers.csv \
  --model-dir /path/to/downloaded/model \
  --output-dir reports/base_model \
  --text-col "Ответ" \
  --codes-col "Коды_новые" \
  --color-by code
```

Для базовой FRIDA добавьте тот же prompt:

```bash
python scripts/visualize_csv.py \
  --input data/answers.csv \
  --model-dir /path/to/FRIDA \
  --output-dir reports/frida \
  --prompt-name categorize_topic \
  --color-by parent_code
```

CSV-разделитель определяется автоматически. При необходимости его можно задать
явно: `--csv-sep ';'`. Для раскраски по основной категории используйте
`--color-by parent_code`. Опциональный `--codebook codes.txt` добавит в
подсказки названия категорий. По умолчанию строится быстрая PCA-визуализация;
для t-SNE добавьте `--method tsne`.

Результат сохраняется в `reports/base_model/csv_single_label_pca.html`, а рядом
создается CSV с двумерными координатами точек.

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

# Survey Classification Pipelines

В репозитории находятся пять независимых подходов:

- [`survey_classifier/`](survey_classifier/) - sentence-transformers, поиск по
  историческим примерам и центроидам;
- [`llm_classifier/`](llm_classifier/) - классификация кодов и тональности через
  OpenAI-совместимый API поднятого в vLLM Qwen;
- [`bert_classifier/`](bert_classifier/) - supervised multi-label
  классификация через RuBERT без embedding-индекса;
- [`tfidf_classifier/`](tfidf_classifier/) - быстрый CPU baseline на словных и
  символьных TF-IDF-признаках с Logistic Regression.
- [`cross_encoder_classifier/`](cross_encoder_classifier/) - парный Transformer,
  который для каждой пары «ответ + описание кода» определяет наличие кода и
  его тональность.

Команды и форматы данных описаны в README каждой директории.

Единый справочник для всех реализаций хранится в UTF-8 CSV с колонками
`Код`, `Категория`, `Подкатегория`. Каждая строка описывает конечный код,
например `A1, Финансы, Зарплата`; отдельные строки для кодов `A`, `B` не нужны.

## Единый CLI

Общие ключи называются одинаково во всех реализациях.

Обучение для `survey_classifier`, `bert_classifier`, `tfidf_classifier` и
`cross_encoder_classifier`:

```bash
python scripts/train.py \
  --train data/train.xlsx \
  --codebook data/codes.csv \
  --out-dir model_out \
  --text-col "Ответ" \
  --codes-col "Коды_новые" \
  --context-col "Вопрос" \
  --val-size 0.1 \
  --test-size 0.1 \
  --seed 42
```

Инференс во всех пяти папках запускается через `scripts/predict.py`:

```bash
python scripts/predict.py \
  --input data/answers.xlsx \
  --output predictions.xlsx \
  --text-col "Ответ" \
  --context-col "Вопрос" \
  --after-semicolon-prefix "нужно улучшить:" \
  --gold-codes-col "Коды_новые" \
  --max-labels 6
```

Локальные классификаторы дополнительно требуют `--model-dir`, а LLM -
`--codebook` и `--base-url`. Параметры конкретного алгоритма, например
`--training-mode`, `--device`, `--concurrency` или `--classifier-c`, остаются
специфичными.

У cross-encoder дополнительно используются `--sentiments-col` при обучении и
`--gold-sentiments-col` при оценке. Через `--codebook` на инференсе ему можно
передать измененный справочник без переобучения.

Общие выходные колонки:

- `predicted_codes`, `predicted_names`;
- `predicted_parent_codes`, `predicted_parent_names`;
- `confidence`, `margin`, `top_candidates`, `needs_review`.

Cross-encoder и LLM-классификатор также добавляют `predicted_sentiments`,
`predicted_code_sentiments` и `predicted_sentiment_names`.

При `--gold-codes-col` рядом с результатом создаются одинаково названные
`<output>_stats.json`, `<output>_per_class.csv` и `<output>_errors.csv`.

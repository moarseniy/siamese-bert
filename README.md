# Survey Classification Pipelines

В репозитории находятся три независимых подхода:

- [`survey_classifier/`](survey_classifier/) - sentence-transformers, поиск по
  историческим примерам и центроидам, а также TF-IDF baseline;
- [`llm_classifier/`](llm_classifier/) - классификация через OpenAI-совместимый
  API поднятого в vLLM Qwen;
- [`bert_classifier/`](bert_classifier/) - supervised multi-label
  классификация через RuBERT без embedding-индекса.

Команды и форматы данных описаны в README каждой директории.

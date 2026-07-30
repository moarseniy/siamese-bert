# Survey Classification Pipelines

В репозитории находятся четыре независимых подхода:

- [`survey_classifier/`](survey_classifier/) - sentence-transformers, поиск по
  историческим примерам и центроидам;
- [`llm_classifier/`](llm_classifier/) - классификация через OpenAI-совместимый
  API поднятого в vLLM Qwen;
- [`bert_classifier/`](bert_classifier/) - supervised multi-label
  классификация через RuBERT без embedding-индекса;
- [`tfidf_classifier/`](tfidf_classifier/) - быстрый CPU baseline на словных и
  символьных TF-IDF-признаках с Logistic Regression.

Команды и форматы данных описаны в README каждой директории.

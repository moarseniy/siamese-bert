# Direct vLLM service

Минимальный сервис для прямого запуска обученной четырехклассовой модели
`bge-reranker-v2-m3` через штатный HTTP API vLLM. Здесь нет FastAPI,
обработки Excel, справочника, порогов и формирования итоговых кодов.

## Модель

Перед сборкой материализуйте DVC-артефакт так, чтобы модель находилась по пути:

```text
artifacts/models/bge-reranker-v2-m3/
├── config.json
├── model.safetensors
├── tokenizer.json
└── ...
```

В этой директории DVC намеренно не запускается. Файл `.dvc` сам по себе не
является моделью: выполните `dvc pull` до сборки либо примонтируйте готовый
каталог модели при запуске контейнера.

## Docker

```bash
docker build -t survey-cross-encoder-vllm .
docker run --rm --gpus all \
  -p 8000:8000 \
  survey-cross-encoder-vllm
```

Если модель не включается в образ:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -v /path/to/bge-reranker-v2-m3:/app/artifacts/models/bge-reranker-v2-m3:ro \
  survey-cross-encoder-vllm
```

Настройки можно переопределять переменными `MODEL_PATH`,
`SERVED_MODEL_NAME`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `VLLM_HOST` и
`VLLM_PORT`.

## Проверка

`test.py` использует только стандартную библиотеку Python:

```bash
python test.py
```

Можно передать готовую токенизированную пару:

```bash
python test.py --token-ids '[0, 123, 456, 2, 2, 789, 2]'
```

Штатный `/classify` принимает одну строку либо token IDs. Он не принимает
отдельные поля `answer` и `code_description`, поэтому клиент production-запроса
должен сам построить и токенизировать пару точно так же, как при обучении.
Ответ `/classify` содержит вероятности классов
`absent/neutral/positive/negative`; дальнейшей обработки сервис не выполняет.

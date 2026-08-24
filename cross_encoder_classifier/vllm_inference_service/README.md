# Изолированный Excel-инференс через vLLM

Самостоятельный сервис работает в одном Docker-контейнере с двумя внутренними
процессами:

- `vllm` слушает только `127.0.0.1:8000`, загружает обученную
  четырехклассовую модель и выполняет инференс;
- `gateway` принимает один `.xlsx`, извлекает колонку с ответами, формирует
  пары с кодами, вызывает внутренний vLLM `/classify` и возвращает ZIP через
  единственный опубликованный порт `8080`.

В `gateway` нет `torch` и `transformers`. Для точного воспроизведения парной
токенизации используется только Rust-библиотека `tokenizers` и сохраненный
рядом с моделью `tokenizer.json`.

## Контракт HTTP

Единственный прикладной endpoint:

```text
POST /predict
Content-Type: multipart/form-data
поле: file=<один файл .xlsx>
```

В первой строке активного листа должна быть колонка `Ответ` (название
настраивается через `TEXT_COLUMN`). Успешный ответ имеет тип `application/zip`
и содержит:

- `<имя_файла>_predictions.xlsx` с добавленной колонкой `Предсказание`;
- `codebook.xlsx`, использованный при классификации.

Значение `Предсказание` имеет вид `A1:2, B1:1`, где тональности означают:
`0` нейтральная, `1` позитивная, `2` негативная. Если кодов нет, записывается
`UNKNOWN`. Если колонка `Предсказание` уже существует, она перезаписывается.
Остальные листы, формулы и оформление исходной книги сохраняются.

Пример запроса:

```bash
curl -f -X POST http://localhost:8080/predict \
  -F 'file=@/path/to/answers.xlsx' \
  -o result.zip
```

Ошибки входного файла возвращаются JSON-ответом с HTTP 400/413/422, ошибки
vLLM возвращаются с HTTP 502.

## Артефакты и DVC

Если модель уже скачана, каталог `artifacts` выглядит так:

```text
artifacts/
├── classifier_config.json
├── codebook.xlsx
└── models/
    └── bge-reranker-v2-m3/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

`classifier_config.json` и содержимое каталога модели создаются текущим
`scripts/train.py`. Сохраненный каталог `model/` должен находиться в сервисе
как `artifacts/models/bge-reranker-v2-m3/`.
Положите production-справочник с колонками `Код`, `Категория`,
`Подкатегория` под именем `codebook.xlsx`. Именно этот файл участвует в
инференсе и возвращается клиенту без изменений.

Вместо скачанного каталога модели можно оставить DVC-указатель:

```text
artifacts/
├── .dvc/
│   └── config
├── classifier_config.json
├── codebook.xlsx
└── models/
    └── bge-reranker-v2-m3.dvc
```

При старте `start.sh` сначала ищет готовый
`artifacts/models/bge-reranker-v2-m3/config.json`. Если его нет, но существует
`bge-reranker-v2-m3.dvc`, выполняется `dvc pull`, а vLLM запускается только
после успешного скачивания и проверки модели.

Для DVC нужен либо `artifacts/.dvc/config` с настроенным remote, либо точный URL
исходного DVC remote в `DVC_REMOTE_URL`. Одного `.dvc`-файла без информации о
remote недостаточно. Доступ к S3 передается стандартными переменными AWS или
ролью, назначенной контейнеру. Для S3 в образ устанавливается
`dvc[s3]==3.1.0`.

### Поддерживаемая модель

Сервис намеренно поддерживает только четырехклассовую модель, дообученную
текущим `cross_encoder_classifier` от `BAAI/bge-reranker-v2-m3`. При старте
проверяются архитектура `XLMRobertaForSequenceClassification`, параметры
BGE-M3 и порядок классов `absent/neutral/positive/negative`. Другие BERT,
RuBERT, RoBERTa и XLM-R модели находятся вне текущего контракта сервиса.

В качестве основы используется корпоративный образ vLLM `0.17.1` на Python
3.10.

## Запуск одним Docker-контейнером

Соберите образ:

```bash
cd cross_encoder_classifier/vllm_inference_service
cp .env.example .env
docker build -t survey-cross-encoder .
```

Запустите контейнер, подставив абсолютный путь к каталогу `artifacts`:

```bash
docker run -d \
  --name survey-cross-encoder \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --env-file .env \
  -p 8080:8080 \
  -v /absolute/path/to/artifacts:/app/artifacts \
  survey-cross-encoder
```

При использовании DVC каталог должен быть доступен на запись пользователю с
UID `10001`: туда материализуется модель и записывается DVC cache. Если модель
уже скачана, каталог после первого успешного запуска можно подключать read-only.

Порт `8000` намеренно не публикуется. Посмотреть DVC pull, запуск модели и
gateway:

```bash
docker logs -f survey-cross-encoder
```

## Роль start.sh

В корне этой папки находится обязательный [`start.sh`](start.sh). Он активирует
созданный внутри образа `venv`, после чего запускает оба процесса:

- при необходимости скачивает модель командой `dvc pull`;
- vLLM с моделью из `./artifacts/models/bge-reranker-v2-m3` только на
  `127.0.0.1:8000`;
- gateway с полной предобработкой и постобработкой на `0.0.0.0:8080`.

Таким образом, снаружи доступен только `POST /predict`. Сырой vLLM endpoint
`/classify` наружу не публикуется и дополнительно защищается случайным
внутренним API-ключом, который скрипт сам передает gateway. Docker запускает
этот скрипт автоматически как entrypoint контейнера.

Скрипт передает `--served-model-name survey-cross-encoder` и
`--gpu-memory-utilization 0.90`. `--max-model-len` применим и к этой
classification/pooling-модели: он ограничивает общую длину пары «ответ +
описание кода». По умолчанию используется `256`, как в стандартном обучении.
Если модель обучалась с другим `--max-length`, задайте то же значение:

```bash
MAX_MODEL_LEN=512 GPU_MEMORY_UTILIZATION=0.85 ./start.sh
```

Остановка `start.sh` завершает и gateway, и дочерний процесс vLLM.

Gateway ждёт загрузки модели до 15 минут, после чего начинает принимать
`POST /predict` на порту `8080`. Порог, `max_labels`, `max_length` и приписка
после `;` читаются из `classifier_config.json`, то есть соответствуют обучению.

Ручной запуск одного `vllm serve` не является прикладным режимом сервиса,
поскольку в этом случае обходятся предобработка Excel и сборка результата.

## Настройки

| Переменная | По умолчанию | Назначение |
|---|---:|---|
| `CLASSIFIER_CONFIG_PATH` | `./artifacts/classifier_config.json` | Настройки обученного классификатора |
| `MODEL_CONFIG_PATH` | `./artifacts/models/bge-reranker-v2-m3/config.json` | Конфигурация для проверки модели BGE |
| `CODEBOOK_PATH` | `./artifacts/codebook.xlsx` | Production-справочник кодов |
| `TOKENIZER_PATH` | `./artifacts/models/bge-reranker-v2-m3/tokenizer.json` | Токенизатор обученной модели |
| `TEXT_COLUMN` | `Ответ` | Колонка с текстом опроса |
| `OUTPUT_COLUMN` | `Предсказание` | Добавляемая колонка |
| `SHEET_NAME` | активный лист | Обрабатываемый лист |
| `PAIR_BATCH_SIZE` | `512` | Максимум пар в одном вызове vLLM |
| `MAX_UPLOAD_BYTES` | `52428800` | Максимальный размер входного файла |
| `MAX_ROWS` | `100000` | Максимум строк данных |
| `THRESHOLD` | из config | Необязательное переопределение порога |
| `MAX_LABELS` | из config | Необязательное ограничение кодов |
| `VLLM_API_KEY` | случайный в `start.sh` | Внутренний Bearer-токен между gateway и vLLM |
| `DVC_FILE` | `<MODEL_PATH>.dvc` | Путь к DVC-указателю модели |
| `DVC_REMOTE` | default remote | Имя remote из DVC config |
| `DVC_REMOTE_URL` | пусто | URL remote, если DVC config не поставляется |
| `DVC_S3_ENDPOINT_URL` | пусто | Endpoint S3-совместимого хранилища |
| `DVC_JOBS` | настройка DVC | Число параллельных загрузок |

## Тесты

```bash
pip install -r requirements-dev.txt
pytest -q
```

[English](../README.md) | [简体中文](README-zh.md) | [繁體中文](README-zh-Hant.md) | [Русский](README-ru.md)

# ScribeCrate

**API для транскрибации на собственном сервере с открытым исходным кодом.**

[![Статус сборки](https://github.com/hwdsl2/scribecrate/actions/workflows/main.yml/badge.svg)](https://github.com/hwdsl2/scribecrate/actions/workflows/main.yml) &nbsp;[![Docker Pulls](https://raw.githubusercontent.com/hwdsl2/badges/main/img/docker-pulls-whisper-server.svg)](https://hub.docker.com/r/hwdsl2/whisper-server) &nbsp;[![License: MIT](images/license.svg)](https://opensource.org/licenses/MIT) &nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://selfhostedaistack.com/scribecrate-notebook)

ScribeCrate предоставляет API для преобразования речи в текст на собственном сервере, позволяющий транскрибировать аудио, создавать субтитры и переводить речь на английский язык. Сервер предоставляет совместимые с OpenAI конечные точки транскрибации и перевода, используя модели [Whisper](https://github.com/openai/whisper) и движок [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Разверните сервер с помощью Docker на CPU или NVIDIA GPU.

> Ранее проект назывался `docker-whisper`. Его поддерживает [hwdsl2](https://github.com/hwdsl2). Имя Docker-образа остаётся прежним: `hwdsl2/whisper-server`.

**Возможности:**

- **Совместимый с OpenAI API:** конечные точки `POST /v1/audio/transcriptions` и `POST /v1/audio/translations` для интеграции с клиентами, поддерживающими OpenAI Whisper API.
- **Конфиденциальная локальная обработка:** аудио остаётся на вашем сервере и не отправляется третьим сторонам для транскрибации.
- **Диаризация:** определение говорящего в каждом сегменте с помощью необязательного локального расширения [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
- **Потоковая выдача результатов:** добавьте `stream=true`, чтобы получать сегменты транскрипции через Server-Sent Events по мере декодирования, не дожидаясь завершения обработки всего загруженного файла.
- **Поддержка моделей Whisper:** выбор моделей `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` и других.
- **Поддержка CPU и GPU:** запускайте сервер на CPU или ускоряйте обработку с помощью GPU NVIDIA и образа `:cuda`.
- **Широкая поддержка аудиоформатов:** MP3, M4A, WAV, WebM, OGG, FLAC и другие аудиоформаты, поддерживаемые FFmpeg.
- **Гибкие форматы вывода:** JSON, обычный текст, подробный JSON, субтитры SRT и WebVTT.
- **Автономная работа:** запуск без доступа к интернету с предварительно кэшированными моделями и `WHISPER_LOCAL_ONLY`.
- **Автоматическая сборка:** образы автоматически собираются и публикуются через [GitHub Actions](https://github.com/hwdsl2/scribecrate/actions); рабочие процессы сборки доступны публично.

Также доступен в составе [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack/blob/main/docs/README-ru.md), который развёртывает полный AI-стек на собственном сервере одной командой.

> 📘 [The Self-Hosted AI Builder’s Guide](https://books2read.com/aiguide?store=amazon): практическое руководство по созданию, защите и эксплуатации собственного приватного ИИ-стека.

**Попробовать онлайн:** [Открыть в Colab](https://selfhostedaistack.com/scribecrate-notebook) (Docker и локальная установка не требуются)

**Связанные AI-сервисы:**

- **Речь:** [ScribeCrate Live](https://github.com/hwdsl2/scribecrate-live/blob/main/docs/README-ru.md), [SpeakCrate](https://github.com/hwdsl2/speakcrate/blob/main/docs/README-ru.md)
- **Документы и поиск:** [ParseCrate](https://github.com/hwdsl2/parsecrate/blob/main/docs/README-ru.md), [EmbedCrate](https://github.com/hwdsl2/embedcrate/blob/main/docs/README-ru.md)
- **Модели и шлюзы:** [InferCrate](https://github.com/hwdsl2/infercrate/blob/main/docs/README-ru.md), [GatewayCrate](https://github.com/hwdsl2/gatewaycrate/blob/main/docs/README-ru.md), [UplinkCrate](https://github.com/hwdsl2/uplinkcrate/blob/main/docs/README-ru.md)

## Быстрый старт

Запустите ScribeCrate следующей командой:

```bash
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

<details>
<summary><strong>Быстрый старт с GPU (NVIDIA CUDA)</strong></summary>

Если у вас есть GPU NVIDIA, используйте образ `:cuda` для аппаратного ускорения инференса:

```bash
docker run \
    --name whisper \
    --restart=always \
    --gpus=all \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server:cuda
```

**Требования:** GPU NVIDIA, [драйвер NVIDIA](https://www.nvidia.com/en-us/drivers/) 575.57.08+ (Linux) или 576.57+ (Windows), а также установленный на хосте [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Образ `:cuda` поддерживает только `linux/amd64`.

</details>

**Важно:** Для работы образа с моделью `base` по умолчанию требуется не менее 700 МБ свободной оперативной памяти. Системы с 512 МБ ОЗУ и менее не поддерживаются.

> [!NOTE]
> Для развёртываний, доступных из интернета, используйте [обратного прокси](#использование-обратного-прокси) для HTTPS. Также замените `-p 9000:9000` на `-p 127.0.0.1:9000:9000` в команде `docker run` выше, чтобы исключить прямой доступ к незашифрованному порту извне.

При первом запуске модель Whisper `base` (~145 МБ) автоматически загружается и кэшируется. Проверьте логи, чтобы убедиться в готовности сервера:

```bash
docker logs whisper
```

После появления сообщения "ScribeCrate transcription server is ready" получите API-ключ, автоматически созданный при новой установке с постоянным томом, указанным выше:

```bash
scribe_api_key="$(docker exec whisper whisper_manage --getkey)"
```

Транскрибируйте первый аудиофайл, заменив `IP_вашего_сервера` адресом своего сервера, а `audio.mp3` своим аудиофайлом:

```bash
curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

**Ответ:**
```json
{"text": "Здесь отображается распознанный текст."}
```

**Совет:** Нужен образец аудиофайла для тестирования? Можно использовать этот образец английской речи (WAV, лицензия MIT) из репозитория [Azure Samples](https://github.com/Azure-Samples/cognitive-services-speech-sdk):

```bash
curl -L -o sample_speech.wav \
    "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/sampledata/audiofiles/katiesteve.wav"

curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@sample_speech.wav \
    -F model=whisper-1
```

В качестве альтернативы вы можете [настроить Whisper без Docker](https://github.com/hwdsl2/whisper-install/blob/main/docs/README-ru.md). Чтобы узнать больше об использовании этого образа, ознакомьтесь с разделами ниже.

## ScribeCrate и ScribeCrate Live: что выбрать

ScribeCrate Live развёртывается как отдельный сервер на базе [WhisperLive](https://github.com/collabora/WhisperLive).

| | **ScribeCrate** | [ScribeCrate Live](https://github.com/hwdsl2/scribecrate-live/blob/main/docs/README-ru.md) |
|---|---|---|
| **Назначение** | Транскрибирование готовых аудиофайлов | Живой микрофон / потоковое аудио в реальном времени |
| **Протокол** | HTTP REST | WebSocket (потоковый) + HTTP REST |
| **Задержка** | JSON после обработки; сегменты через SSE | Постепенные обновления сегментов |
| **Подходит для** | Записи совещаний, загруженные аудиофайлы | Захват в браузере, RTSP-потоки, живые субтитры |
| **Размер образа** | ~190 МБ (~3,1 ГБ для `:cuda`) | ~750 МБ (~4,5 ГБ для `:cuda`) |

## Сообщество

- 📬 [Получайте новости проектов и бесплатные руководства по развёртыванию](https://selfhostedstack.beehiiv.com/subscribe?utm_campaign=ai-ru) (1–2 письма в месяц; руководства в формате PDF на английском языке)
- 💬 Присоединяйтесь к сообществу [r/selfhostedstack](https://www.reddit.com/r/selfhostedstack/) для обсуждений и демонстрации проектов
- ⭐ Поставьте звезду репозиторию, если он оказался вам полезен — это поможет другим пользователям его найти.

<details>
<summary>Самостоятельно размещаемые VPN и сетевые проекты</summary>

- [Setup IPsec VPN](https://github.com/hwdsl2/setup-ipsec-vpn/blob/master/README-ru.md)
- [IPsec VPN на Docker](https://github.com/hwdsl2/docker-ipsec-vpn-server/blob/master/README-ru.md)
- [WireGuard](https://github.com/hwdsl2/docker-wireguard/blob/main/docs/README-ru.md)
- [OpenVPN](https://github.com/hwdsl2/docker-openvpn/blob/main/docs/README-ru.md)
- [Headscale](https://github.com/hwdsl2/docker-headscale/blob/main/docs/README-ru.md)

</details>

## Требования

- Сервер Linux (локальный или облачный) с установленным Docker
- Поддерживаемые архитектуры: `amd64` (x86_64), `arm64` (например, Raspberry Pi 4/5, AWS Graviton)
- Минимум оперативной памяти: ~700 МБ для модели `base` по умолчанию (см. [таблицу моделей](#переключение-моделей))
- Доступ в интернет при первом запуске для загрузки модели (затем модель кэшируется локально). Не требуется при использовании `WHISPER_LOCAL_ONLY=true` с предварительно кэшированными моделями.

**Для ускорения на GPU (образ `:cuda`):**

- GPU NVIDIA с поддержкой CUDA (Compute Capability 6.0+)
- [Драйвер NVIDIA](https://www.nvidia.com/en-us/drivers/) 575.57.08+ (Linux) или 576.57+ (Windows) на хосте
- Установленный [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Образ `:cuda` поддерживает только `linux/amd64`

Для развёртывания с выходом в интернет см. раздел [Использование обратного прокси](#использование-обратного-прокси) для включения HTTPS.

## Скачать

Получите доверенную сборку из [Docker Hub](https://hub.docker.com/r/hwdsl2/whisper-server/):

```bash
docker pull hwdsl2/whisper-server
```

Для ускорения на GPU NVIDIA загрузите тег `:cuda`:

```bash
docker pull hwdsl2/whisper-server:cuda
```

Либо скачайте из [Quay.io](https://quay.io/repository/hwdsl2/whisper-server):

```bash
docker pull quay.io/hwdsl2/whisper-server
docker image tag quay.io/hwdsl2/whisper-server hwdsl2/whisper-server
```

CPU-образы поддерживают `linux/amd64` и `linux/arm64`. Тег `:cuda` поддерживает только `linux/amd64`.

## Переменные окружения

Все переменные являются необязательными. Новые установки с подключённым томом `/var/lib/whisper` автоматически генерируют Bearer-токен. Существующие установки без ключа остаются открытыми для обратной совместимости.

Этот Docker-образ использует следующие переменные, которые можно объявить в файле `env` (см. [пример](../whisper.env.example)):

| Переменная | Описание | По умолчанию |
|---|---|---|
| `WHISPER_MODEL` | Модель Whisper для использования. См. [таблицу моделей](#переключение-моделей). | `base` |
| `WHISPER_LANGUAGE` | Язык транскрибирования по умолчанию. Код BCP-47 (напр. `ru`, `en`, `zh`) или `auto` для автоопределения. | `auto` |
| `WHISPER_PORT` | HTTP-порт для API (1–65535). | `9000` |
| `WHISPER_DEVICE` | Устройство вычислений: `cpu`, `cuda` или `auto`. Используйте `cuda` с образом `:cuda` для ускорения на GPU. `auto` определяет GPU автоматически, при отсутствии — переключается на CPU. | `cpu` |
| `WHISPER_COMPUTE_TYPE` | Тип квантования / вычислений. Для CPU рекомендуется `int8`; для CUDA рекомендуется `float16`. | `int8` (CPU) / `float16` (CUDA) |
| `WHISPER_THREADS` | Количество потоков CPU для инференса. Установите значение, равное числу физических ядер, для минимальной задержки. | `2` |
| `WHISPER_API_KEY` | Необязательный Bearer-токен. В новых постоянных установках генерируется автоматически. Если задан, все запросы должны содержать `Authorization: Bearer <key>`. Явно пустое значение отключает аутентификацию. | Автоматически для новых постоянных установок |
| `WHISPER_LOG_LEVEL` | Уровень логирования: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. | `INFO` |
| `WHISPER_BEAM` | Ширина луча при декодировании транскрипции и перевода. Большие значения могут улучшить точность за счёт скорости. Используйте `1` для наиболее быстрого (жадного) декодирования. | `5` |
| `WHISPER_MAX_REQUEST_BEAM` | Максимальная ширина луча, разрешённая для переопределения `beam` в отдельном запросе. Установите `0`, чтобы отключить этот лимит. | `10` |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | Использовать предыдущий текст как контекст окна декодирования. Значение `false` может уменьшить циклические повторы, но также может снизить согласованность текста между окнами. | `true` |
| `WHISPER_NO_REPEAT_NGRAM_SIZE` | Ограничить повторяющиеся токеновые n-граммы при декодировании. `0` отключает ограничение; целое десятичное число от `0` до `2147483647` без знака и ведущих нулей. N-граммы состоят из токенов модели, которые не обязательно являются целыми словами. | `0` |
| `WHISPER_MAX_UPLOAD_MB` | Максимальный размер загружаемого аудиофайла в МБ. Запросы сверх этого лимита возвращают HTTP 413. Установите `0`, чтобы отключить лимит. | `1024` |
| `WHISPER_LOCAL_ONLY` | Если задано любое непустое значение (например, `true`), отключает все загрузки моделей с HuggingFace. Для изолированных или офлайн-развёртываний с предварительно кэшированными моделями. | *(не задан)* |
| `WHISPER_WORD_TIMESTAMPS` | При значении `true` глобально включает пословные метки времени. Вывод `verbose_json` будет содержать массив `words` на верхнем уровне с временем начала/конца и вероятностью для каждого слова. Также можно включить через `timestamp_granularities[]=word`. | *(не задан)* |
| `WHISPER_DIARIZATION` | При значении `true` включает диаризацию говорящих. Использует [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) с моделями pyannote segmentation-3.0 в формате ONNX (~46 МБ с моделью эмбеддингов по умолчанию, загружаются автоматически при первом использовании). Не поддерживается в потоковом режиме. | *(не задан)* |
| `WHISPER_DIARIZE_EMBEDDING_MODEL` | Имя файла модели эмбеддингов говорящих из релиза sherpa-onnx [speaker-recongition-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models). Модель по умолчанию обучена на китайском (путунхуа); для английского рассмотрите `wespeaker_en_voxceleb_resnet34_LM.onnx` или `nemo_en_titanet_large.onnx`. См. [диаризацию говорящих](#диаризация-говорящих). | `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` |
| `WHISPER_DIARIZE_NUM_SPEAKERS` | Точное количество говорящих (если известно). Повышает точность кластеризации. Установите `-1` или оставьте пустым для автоопределения. | `-1` |
| `WHISPER_DIARIZE_THREADS` | Число потоков CPU для каждой модели сегментации и эмбеддингов. Целое десятичное число от `1` до `2147483647` без знака и ведущих нулей; если не задано или пусто, наследует `WHISPER_THREADS`. | `WHISPER_THREADS` |
| `WHISPER_DIARIZE_THRESHOLD` | Порог кластеризации для автоопределения. Меньше = больше говорящих, больше = меньше. Игнорируется, когда задано точное количество говорящих. | `0.5` |
| `WHISPER_DISABLE_USAGE_COUNTS` | Установите `1`, чтобы отключить анонимные агрегированные счётчики использования. | *(не задан)* |

**Примечание:** В файле `env` значения можно заключать в одинарные кавычки, например `VAR='value'`. Не добавляйте пробелы вокруг `=`. Если вы изменяете `WHISPER_PORT`, обновите флаг `-p` в команде `docker run` соответственно.

Оба параметра применяются к транскрипции и переводу как в пакетном режиме, так и в режиме SSE. Для записей с повторами можно попробовать `WHISPER_CONDITION_ON_PREVIOUS_TEXT=false` и `WHISPER_NO_REPEAT_NGRAM_SIZE=3`. Это может подавить реальные повторы речи или снизить согласованность текста между окнами декодирования; отсутствие галлюцинаций не гарантируется.

Значения этих параметров могут содержать пробельные символы ASCII по краям и не более одной пары совпадающих внешних кавычек. Пустые значения используют настройки по умолчанию. Для булевых значений допустимы `true`/`false` без учёта регистра букв ASCII; целые числа должны содержать только цифры ASCII, без знака и ведущих нулей. Некорректные непустые значения останавливают запуск до загрузки модели.

Файл `api_server.py`, подключённый через bind mount, скрывает API из образа. Удалите это подключение, чтобы использовать API из образа, или обновляйте свою копию при обновлении образа.

После изменения файла `/whisper.env`, подключённого через bind mount, перезапустите контейнер. Если контейнер по-прежнему видит старое содержимое файла, пересоздайте только этот контейнер. При изменении переменных окружения контейнера Docker/Compose или образа требуется пересоздание с сохранением прежних томов и проекта Compose. Обычный перезапуск не применяет эти изменения.

Удаление этих параметров возвращает настройки декодирования по умолчанию. Не удаляйте тома моделей при обновлении или откате.

Пример использования файла `env`:

```bash
cp whisper.env.example whisper.env
# Отредактируйте whisper.env, затем:
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -v ./whisper.env:/whisper.env:ro \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

Файл `env`, подключённый через bind mount, читается при запуске контейнера. После изменений перезапустите контейнер; если он по-прежнему видит старое содержимое файла, пересоздайте его с теми же томами.

<details>
<summary>Либо передайте его через <code>--env-file</code></summary>

```bash
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    --env-file=whisper.env \
    -d hwdsl2/whisper-server
```

</details>

## Использование docker-compose

```bash
cp whisper.env.example whisper.env
# Отредактируйте whisper.env при необходимости, затем:
docker compose up -d
docker logs whisper
```

Пример `docker-compose.yml` (уже включён в проект):

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # Для хостового обратного прокси измените на "127.0.0.1:9000:9000/tcp"
    volumes:
      - whisper-data:/var/lib/whisper
      - ./whisper.env:/whisper.env:ro

volumes:
  whisper-data:
    name: whisper-data
```

> [!NOTE]
> Для развёртывания с выходом в интернет используйте [обратный прокси](#использование-обратного-прокси) для добавления HTTPS. Также измените `"9000:9000/tcp"` на `"127.0.0.1:9000:9000/tcp"` в `docker-compose.yml`, чтобы предотвратить прямой доступ к незашифрованному порту.

<details>
<summary><strong>Использование docker-compose с GPU (NVIDIA CUDA)</strong></summary>

Для развёртывания с GPU используется отдельный файл `docker-compose.cuda.yml`:

```bash
cp whisper.env.example whisper.env
# Отредактируйте whisper.env при необходимости, затем:
docker compose -f docker-compose.cuda.yml up -d
docker logs whisper
```

Пример `docker-compose.cuda.yml` (уже включён в проект):

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server:cuda
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # Для хостового обратного прокси измените на "127.0.0.1:9000:9000/tcp"
    volumes:
      - whisper-data:/var/lib/whisper
      - ./whisper.env:/whisper.env:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  whisper-data:
    name: whisper-data
```

</details>

## Справочник по API

ScribeCrate предоставляет конечные точки, совместимые с интерфейсами [транскрибации аудио](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) и [перевода аудио](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create) OpenAI. Для клиентов, использующих OpenAI SDK, настройте базовый URL и укажите свой API-ключ ScribeCrate:

**Аутентификация:** Для новых установок с постоянным томом требуется API-ключ. Получите его для настройки SDK и примеров `curl` ниже:

```bash
scribe_api_key="$(docker exec whisper whisper_manage --getkey)"

export OPENAI_BASE_URL="http://IP_вашего_сервера:9000/v1"
export OPENAI_API_KEY="$scribe_api_key"
```

Если аутентификация по API-ключу отключена, опустите заголовок `Authorization` в примерах curl. Клиентам OpenAI SDK по-прежнему нужен непустой ключ; в этом случае задайте `OPENAI_API_KEY=unused`.

Диаризация говорящих, если включена, является локальным расширением на базе sherpa-onnx и не эквивалентна моделям диаризации OpenAI. Параметры транскрибирования, специфичные для OpenAI, такие как `gpt-4o-transcribe-diarize`, `response_format=diarized_json`, `include=logprobs`, `chunking_strategy`, `known_speaker_names` и `known_speaker_references`, не поддерживаются и возвращают `400`.

### Транскрибирование аудио

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**Параметры:**

| Параметр | Тип | Обязателен | Описание |
|---|---|---|---|
| `file` | файл | ✅ | Аудиофайл. Поддерживаемые форматы: `mp3`, `mp4`, `m4a`, `wav`, `webm`, `ogg`, `flac` и все форматы, поддерживаемые ffmpeg. |
| `model` | строка | ✅ | Передайте `whisper-1` (значение принимается, но всегда используется активная модель). |
| `language` | строка | — | Код языка BCP-47. Переопределяет `WHISPER_LANGUAGE` для данного запроса. |
| `prompt` | строка | — | Необязательный текст для управления стилем модели или продолжения предыдущего сегмента. |
| `response_format` | строка | — | Формат вывода. По умолчанию: `json`. См. [форматы ответа](#форматы-ответа). Игнорируется при `stream=true`. Специфичный для OpenAI `diarized_json` не поддерживается. |
| `temperature` | число с плавающей точкой | — | Температура сэмплирования (0–1). По умолчанию: `0`. |
| `stream` | логическое | — | Включить потоковую передачу SSE. При значении `true` сегменты возвращаются как события `text/event-stream` по мере декодирования. По умолчанию: `false`. |
| `timestamp_granularities[]` | массив | — | Гранулярность меток времени. Значения: `word`, `segment`. При наличии `word` вывод `verbose_json` содержит массив `words` на верхнем уровне. По умолчанию: `["segment"]`. |

**Локальное расширение faster-whisper:** можно задать `beam`, чтобы переопределить `WHISPER_BEAM` для отдельного запроса транскрибирования или перевода. Это не часть схемы OpenAI API, поэтому не отправляйте этот параметр в размещённый OpenAI API или строгие OpenAI-совместимые шлюзы. Лимит по умолчанию для одного запроса — `10` (`WHISPER_MAX_REQUEST_BEAM`); установите эту переменную в `0`, чтобы отключить лимит. Beam search в основном влияет на детерминированное декодирование при `temperature=0`.

**Пример:**

```bash
curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@meeting.m4a \
    -F model=whisper-1 \
    -F language=ru
```

С аутентификацией по API-ключу:

```bash
curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer your_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

### Форматы ответа

| `response_format` | Описание |
|---|---|
| `json` | `{"text": "..."}` — по умолчанию, соответствует базовому ответу OpenAI |
| `text` | Простой текст без обёртки JSON |
| `verbose_json` | Полный JSON с языком, длительностью, временны́ми метками сегментов и логарифмическими вероятностями |
| `srt` | Формат субтитров SubRip (`.srt`) |
| `vtt` | Формат субтитров WebVTT (`.vtt`) |

**Пример — потоковое получение сегментов по мере декодирования:**

```bash
curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@long-audio.mp3 \
    -F model=whisper-1 \
    -F stream=true
```

**SSE-ответ** (используется [протокол потоковой транскрипции OpenAI](https://developers.openai.com/api/docs/guides/speech-to-text#streaming)):

```
data: {"type":"transcript.text.delta","delta":"Привет, как дела?"}

data: {"type":"transcript.text.delta","delta":" Всё хорошо, спасибо."}

data: {"type":"transcript.text.done","text":"Привет, как дела? Всё хорошо, спасибо."}

data: [DONE]
```

Первый инкрементальный текст обычно приходит через 1–3 секунды после загрузки. Каждое событие `transcript.text.delta` содержит инкрементальный текст только что декодированного сегмента. Финальное событие `transcript.text.done` содержит полный собранный текст транскрипции — аналог стандартного ответа `json`.

<details>
<summary><strong>Пример — потоковая передача через браузерный <code>fetch</code></strong></summary>

```javascript
const form = new FormData();
form.append("file", audioBlob, "audio.webm");
form.append("model", "whisper-1");
form.append("stream", "true");

const res = await fetch("http://IP_вашего_сервера:9000/v1/audio/transcriptions", {
  method: "POST",
  headers: { Authorization: "Bearer your_api_key" },
  body: form,
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  // SSE frames are separated by "\n\n"; split and process complete frames
  const frames = buffer.split("\n\n");
  buffer = frames.pop(); // keep any incomplete trailing frame
  for (const frame of frames) {
    if (!frame.startsWith("data: ")) continue;
    const payload = frame.slice(6);
    if (payload.startsWith("[DONE]")) break;
    const event = JSON.parse(payload);
    if (event.type === "transcript.text.delta") console.log(event.delta);
    if (event.type === "transcript.text.done") console.log("Full text:", event.text);
  }
}
```

</details>

**Пример — получение субтитров SRT:**

```bash
curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@video.mp4 \
    -F model=whisper-1 \
    -F response_format=srt
```

**Пример — подробный JSON с временны́ми метками:**

```bash
curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json
```

**Пример — подробный JSON с пословными метками времени:**

```bash
curl http://IP_вашего_сервера:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json \
    -F "timestamp_granularities[]=word"
```

При наличии `word` в `timestamp_granularities[]` ответ `verbose_json` содержит массив `words` на верхнем уровне:

```json
{
  "word": "hello",
  "start": 0.5,
  "end": 0.8,
  "probability": 0.98
}
```

### Перевод аудио

```
POST /v1/audio/translations
Content-Type: multipart/form-data
```

Перевод аудио с любого языка на английский текст. Совместим с [эндпоинтом перевода OpenAI](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create). Принимает обычные параметры перевода. Вывод всегда на английском.

> **Примечание:** Для перевода речи на английский используйте многоязычную модель без turbo, например `base`, `small`, `medium` или `large-v3`. Англоязычные модели (`.en`) не поддерживают эту задачу, а `large-v3-turbo` / `turbo` не обучены для перевода. Если на сервере загружена такая неподдерживаемая для перевода модель, обычные и потоковые запросы перевода возвращают HTTP 400. Поле `model` запроса не переопределяет серверную настройку `WHISPER_MODEL`.

**Пример:**

```bash
curl http://IP_вашего_сервера:9000/v1/audio/translations \
    -H "Authorization: Bearer $scribe_api_key" \
    -F file=@french_audio.mp3 \
    -F model=whisper-1
```

### Список моделей

```
GET /v1/models
```

Возвращает активную модель в совместимом с OpenAI формате.

```bash
curl http://IP_вашего_сервера:9000/v1/models -H "Authorization: Bearer $scribe_api_key"
```

### Интерактивная документация API

Интерактивный Swagger UI доступен по адресу:

```
http://IP_вашего_сервера:9000/docs
```

## Постоянные данные

Все данные сервера хранятся в Docker-томе (`/var/lib/whisper` внутри контейнера):

```
/var/lib/whisper/
├── models--Systran--faster-whisper-*/   # Кэшированные файлы модели Whisper (скачаны с HuggingFace)
├── .port                 # Активный порт (используется whisper_manage)
├── .model                # Активное название модели (используется whisper_manage)
└── .server_addr          # Кэшированный IP сервера (используется whisper_manage)
```

Создайте резервную копию Docker-тома для сохранения скачанных моделей. Модели занимают значительный объём (145 МБ – 3 ГБ) и могут скачиваться несколько минут при первом запуске; сохранение тома позволяет избежать повторной загрузки при пересоздании контейнера.

**Совет:** Том `/var/lib/whisper` использует тот же формат кэша HuggingFace, что и том `/var/lib/whisper-live` в `docker-whisper-live`. Если вы уже скачали модель через `docker-whisper-live`, можно примонтировать тот же каталог тома, чтобы не загружать её повторно.

## Управление сервером

Используйте `whisper_manage` внутри запущенного контейнера для просмотра информации о сервере и управления им.

**Показать информацию о сервере:**

```bash
docker exec whisper whisper_manage --showinfo
```

**Список доступных моделей:**

```bash
docker exec whisper whisper_manage --listmodels
```

**Предварительная загрузка модели:**

```bash
docker exec whisper whisper_manage --downloadmodel large-v3-turbo
```

## Переключение моделей

Для смены активной модели:

1. *(Необязательно, но рекомендуется)* Предварительно загрузите новую модель, пока сервер работает:
   ```bash
   docker exec whisper whisper_manage --downloadmodel large-v3-turbo
   ```

2. Обновите `WHISPER_MODEL` в файле `whisper.env` (или добавьте `-e WHISPER_MODEL=large-v3-turbo` в команду `docker run`).

3. Перезапустите контейнер:
   ```bash
   docker restart whisper
   ```

**Доступные модели:**

| Модель | Диск | ОЗУ (примерно) | Примечания |
|---|---|---|---|
| `tiny` | ~75 МБ | ~250 МБ | Самая быстрая; низкая точность |
| `tiny.en` | ~75 МБ | ~250 МБ | Только английский |
| `base` | ~145 МБ | ~700 МБ | Хороший баланс — **по умолчанию** |
| `base.en` | ~145 МБ | ~700 МБ | Только английский |
| `small` | ~465 МБ | ~1,5 ГБ | Повышенная точность |
| `small.en` | ~465 МБ | ~1,5 ГБ | Только английский |
| `medium` | ~1,5 ГБ | ~5 ГБ | Высокая точность |
| `medium.en` | ~1,5 ГБ | ~5 ГБ | Только английский |
| `large-v1` | ~3 ГБ | ~10 ГБ | Старая большая модель |
| `large-v2` | ~3 ГБ | ~10 ГБ | Очень высокая точность |
| `large-v3` | ~3 ГБ | ~10 ГБ | Наивысшая точность |
| `large-v3-turbo` | ~1,6 ГБ | ~6 ГБ | Быстрая + высокая точность ⭐; только транскрипция |
| `turbo` | ~1,6 ГБ | ~6 ГБ | Псевдоним для `large-v3-turbo`; только транскрипция |

> **Совет:** Для транскрипции `large-v3-turbo` служит более быстрой альтернативой `large-v3` при достаточных ресурсах оборудования. Для перевода речи на английский выберите многоязычную модель без turbo, например `base`, `small`, `medium` или `large-v3`.

Данные по памяти являются приблизительными и учитывают квантование INT8 (по умолчанию). Модели кэшируются в Docker-томе `/var/lib/whisper` и загружаются только один раз.

## Защита сервера

Если ваш сервер ScribeCrate доступен из публичной сети — даже кратковременно — примените как минимум следующие меры защиты. Транскрибация требует значительных ресурсов CPU/GPU, поэтому неаутентифицированная конечная точка может быть использована для расходования ваших вычислительных ресурсов.

**1. Используйте API-ключ.** Новые установки с подключённым томом `/var/lib/whisper` автоматически генерируют API-ключ. Его можно посмотреть командой `docker exec whisper whisper_manage --showkey`; в скриптах используйте `docker exec whisper whisper_manage --getkey`. Существующие установки без ключа остаются открытыми для обратной совместимости; также можно задать `WHISPER_API_KEY` в env-файле вручную. Все аутентифицированные запросы должны содержать `Authorization: Bearer <key>`.

```bash
# Сгенерировать 32-байтовый случайный ключ
openssl rand -hex 32
```

**2. Привяжите к localhost при использовании обратного прокси.** Замените `-p 9000:9000` на `-p 127.0.0.1:9000:9000` (или измените `"9000:9000/tcp"` на `"127.0.0.1:9000:9000/tcp"` в `docker-compose.yml`), чтобы незашифрованный порт нельзя было достичь напрямую снаружи хоста.

**3. Ограничьте размер загружаемых файлов.** Сервер отклоняет загрузки больше `WHISPER_MAX_UPLOAD_MB` (по умолчанию `1024`). Для развёртываний, доступных из интернета, также настройте обратный прокси на отклонение слишком больших загрузок до того, как они достигнут приложения (например, nginx `client_max_body_size 100M;`).

**4. Следите за уровнем журналирования.** При `WHISPER_LOG_LEVEL=DEBUG` текст транскрипции может попадать в журналы. На общих системах сохраняйте уровень `INFO` или выше.

**5. Включите CORS на прокси при вызове из браузера.** Сервер по умолчанию не устанавливает заголовки `Access-Control-Allow-Origin`; добавьте их на обратном прокси, если планируете вызывать API напрямую с веб-страницы другого источника.

**6. Рассмотрите ограничение частоты запросов.** Разместите перед сервером ограничитель частоты (например, nginx `limit_req_zone`, Caddy `rate_limit`), чтобы ограничить количество одновременных запросов транскрипции на один IP-адрес клиента.

## Использование обратного прокси

Для развёртывания с выходом в интернет разместите обратный прокси перед ScribeCrate для обработки HTTPS-терминации. Сервер работает без HTTPS в локальной или доверенной сети, но HTTPS рекомендуется при открытом доступе к API-эндпоинту из интернета.

Используйте один из следующих адресов для доступа к контейнеру ScribeCrate из обратного прокси:

- **`whisper:9000`** — если ваш обратный прокси работает как контейнер в **той же Docker-сети**, что и ScribeCrate (например, определён в том же `docker-compose.yml`).
- **`127.0.0.1:9000`** — если ваш обратный прокси работает **на хосте** и порт `9000` опубликован (по умолчанию `docker-compose.yml` публикует его).

**Пример с [Caddy](https://caddyserver.com/docs/) ([Docker-образ](https://hub.docker.com/_/caddy))** (автоматический TLS через Let's Encrypt, обратный прокси в той же Docker-сети):

`Caddyfile`:
```
whisper.example.com {
  reverse_proxy whisper:9000
}
```

**Пример с nginx** (обратный прокси на хосте):

```nginx
server {
    listen 443 ssl;
    server_name whisper.example.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Аудиофайлы могут быть большими — увеличьте лимит загрузки при необходимости
    client_max_body_size 100M;

    location / {
        proxy_pass         http://127.0.0.1:9000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;       # требуется для потоковой передачи (SSE)
        proxy_read_timeout 300s;
    }
}
```

## Обновление Docker-образа

Для обновления Docker-образа и контейнера сначала [скачайте](#скачать) последнюю версию:

```bash
docker pull hwdsl2/whisper-server
```

Если образ уже актуален, вы увидите:

```
Status: Image is up to date for hwdsl2/whisper-server:latest
```

В противном случае будет скачана последняя версия. Удалите и пересоздайте контейнер:

```bash
docker rm -f whisper
# Затем повторно выполните команду docker run из раздела "Быстрый старт" с теми же томом и портом.
```

Скачанные модели сохранятся в томе `whisper-data`.

## Использование с другими AI-сервисами

ScribeCrate можно использовать как службу распознавания речи в более широком self-hosted AI-стеке.

Готовые полные и облегчённые стеки Docker Compose, примеры ручного запуска через `docker run`, а также примеры голосовых, RAG- и MCP-конвейеров с SpeakCrate, EmbedCrate, GatewayCrate, InferCrate, ParseCrate и UplinkCrate см. в [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack/blob/main/docs/README-ru.md).

## Диаризация говорящих

Диаризация определяет, *кто говорит* в каждом транскрибированном сегменте. Это локальное расширение на базе [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) с использованием модели pyannote segmentation-3.0, экспортированной в формат ONNX.

**Включение диаризации:**

```bash
# В whisper.env:
WHISPER_DIARIZATION=true
```

ONNX-модели (~46 МБ суммарно с моделью эмбеддингов по умолчанию) автоматически загружаются при первом использовании и кэшируются в томе `/var/lib/whisper`. Предварительная загрузка:

```bash
docker exec whisper whisper_manage --downloaddiarize
```

**Вывод с включённой диаризацией:**

`verbose_json` добавляет поле `speaker` в каждый сегмент:

```json
{
  "segments": [
    {"id": 0, "start": 1.0, "end": 3.5, "text": "Запускаем на следующей неделе.", "speaker": "SPEAKER_00"},
    {"id": 1, "start": 4.0, "end": 6.2, "text": "Думаю, QA нужно ещё два дня.", "speaker": "SPEAKER_01"}
  ]
}
```

`srt` и `vtt` добавляют метку говорящего:

```
1
00:00:01,000 --> 00:00:03,500
[SPEAKER_00] Запускаем на следующей неделе.

2
00:00:04,000 --> 00:00:06,200
[SPEAKER_01] Думаю, QA нужно ещё два дня.
```

Формат `text` показывает метку при смене говорящего:

```
[SPEAKER_00] Запускаем на следующей неделе.
[SPEAKER_01] Думаю, QA нужно ещё два дня.
```

**Примечания:**

> [!IMPORTANT]
> Диаризация требует анализа полного аудио и **не поддерживается в потоковом режиме** (`stream=true`). Если оба включены, диаризация пропускается.

- Установите `WHISPER_DIARIZE_NUM_SPEAKERS`, если известно точное количество говорящих, для повышения точности.
- Диаризация запускается после транскрибирования. Время обработки зависит от длительности аудио, оборудования и числа потоков и может превышать время транскрибирования.

**Потоки диаризации:** Модели сегментации и эмбеддингов теперь наследуют `WHISPER_THREADS` (обычно `2`), а не используют по одному потоку. Это меняет использование ресурсов CPU в существующих установках с включённой диаризацией. Установите `WHISPER_DIARIZE_THREADS=1`, чтобы сохранить прежнее число потоков для каждой модели, или выберите другое значение с учётом доступных ресурсов CPU. Удаление параметра или пустое значение возвращает наследование, а не прежний однопоточный режим по умолчанию.

Как и параметры ограничения повторов, `WHISPER_DIARIZE_THREADS` допускает пробельные символы ASCII по краям и не более одной пары совпадающих внешних кавычек. Используйте десятичные цифры ASCII для числа от `1` до `2147483647`, без знака и ведущих нулей. Некорректные непустые значения останавливают запуск до загрузки моделей, даже если диаризация выключена. Сам по себе этот параметр не включает диаризацию.

Это не ограничение общего числа потоков процесса. Другие библиотеки и обработчики запросов могут создавать дополнительные потоки; большее число потоков не гарантирует ускорение или идентичный результат. Диаризация по-прежнему выполняется на CPU, в том числе в образе `:cuda`, и только в пакетном режиме. Применяйте изменения согласно [инструкциям по перезапуску и пересозданию контейнера](#переменные-окружения); сохраняйте тома моделей при обновлении или откате.

**Модель эмбеддингов говорящих:** модель по умолчанию, `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`, обучена на китайской речи (путунхуа). На английских записях она может разделять одного говорящего на несколько меток. Задайте `WHISPER_DIARIZE_EMBEDDING_MODEL`, чтобы использовать другую модель из релиза sherpa-onnx [speaker-recongition-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models), например:

```bash
# В whisper.env:
WHISPER_DIARIZE_EMBEDDING_MODEL=wespeaker_en_voxceleb_resnet34_LM.onnx
```

Среди моделей, обученных на английском, есть `wespeaker_en_voxceleb_resnet34_LM.onnx` (~27 МБ) и `nemo_en_titanet_large.onnx` (~101 МБ). Размер файла эмбеддингов зависит от выбранной модели; указанные размеры не включают общую модель сегментации (~6 МБ). Нет модели, которая была бы лучшей для всех записей, поэтому сравните их на своём аудио. Используйте точное имя файла `.onnx` из поддерживаемого списка разрешённых моделей с учётом регистра. Неизвестные непустые значения останавливают запуск до загрузки моделей, даже если диаризация отключена. Значение допускает пробельные символы ASCII по краям и не более одной пары совпадающих внешних кавычек.

Выбранная модель загружается при первом использовании и кэшируется в томе `/var/lib/whisper` рядом с ранее использованными моделями. `whisper_manage --downloaddiarize` заранее загружает модель, с которой был запущен контейнер. Пороги кластеризации зависят от модели: после переключения заново подберите `WHISPER_DIARIZE_THRESHOLD` или задайте `WHISPER_DIARIZE_NUM_SPEAKERS`. Если значение не задано или пустое, используется модель по умолчанию, и существующий кэш и пороги не меняются.

## Счётчики использования

Этот образ использует публичные счётчики скачиваний GitHub Release assets для анонимной агрегированной статистики использования. Эти числа приблизительны и не являются количеством уникальных пользователей или активных установок. Образ не отправляет telemetry payload и не использует частный сборщик. Он выполняет только best-effort запрос после успешного запуска сервера с подключённым томом `/var/lib/whisper`, а также при первом запуске другой сборки образа для этой постоянной установки. Чтобы отключить это, задайте `WHISPER_DISABLE_USAGE_COUNTS=1`.

## Технические подробности

- Базовый образ: `python:3.12-slim` для `:latest`; `nvidia/cuda` для `:cuda`
- Среда выполнения: Python 3 (виртуальное окружение в `/opt/venv`)
- STT-движок: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CTranslate2 (INT8 по умолчанию на CPU, FP16 на CUDA)
- API-фреймворк: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- Декодирование аудио: [PyAV](https://github.com/PyAV-Org/PyAV) (встроенные библиотеки FFmpeg)
- Директория данных: `/var/lib/whisper` (Docker-том)
- Хранение моделей: формат HuggingFace Hub внутри тома — скачивается один раз, переиспользуется при перезапусках

## Лицензия

**Примечание:** Программные компоненты внутри готового образа (такие как faster-whisper и его зависимости) распространяются под лицензиями, выбранными соответствующими правообладателями. При использовании готового образа пользователь несёт ответственность за соблюдение всех соответствующих лицензий на программное обеспечение, содержащееся в образе.

Copyright (C) 2026 Lin Song   
Данная работа распространяется под [лицензией MIT](https://opensource.org/licenses/MIT).

**faster-whisper** является собственностью SYSTRAN и распространяется под [лицензией MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE).

ScribeCrate представляет собой независимый сервер на основе моделей Whisper и не аффилирован с OpenAI или SYSTRAN, не одобрен и не спонсирован ими.

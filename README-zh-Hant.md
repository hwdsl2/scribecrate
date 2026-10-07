[English](README.md) | [简体中文](README-zh.md) | [繁體中文](README-zh-Hant.md) | [Русский](README-ru.md)

# ScribeCrate

**開源、自託管的音訊轉錄 API。**

[![建置狀態](https://github.com/hwdsl2/scribecrate/actions/workflows/main.yml/badge.svg)](https://github.com/hwdsl2/scribecrate/actions/workflows/main.yml) &nbsp;[![Docker Pulls](https://raw.githubusercontent.com/hwdsl2/badges/main/img/docker-pulls-whisper-server.svg)](https://hub.docker.com/r/hwdsl2/whisper-server) &nbsp;[![License: MIT](docs/images/license.svg)](https://opensource.org/licenses/MIT) &nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://selfhostedaistack.com/scribecrate-notebook)

ScribeCrate 是一個自託管的語音轉文字 API，用於轉錄音訊、產生字幕，並將語音翻譯為英語。它使用 [Whisper](https://github.com/openai/whisper) 模型，由 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 驅動，提供相容 OpenAI 的轉錄和翻譯端點。可透過 Docker 部署，使用 CPU 或 NVIDIA GPU 執行。

> 本專案原名 **docker-whisper**，仍由 [hwdsl2](https://github.com/hwdsl2) 維護。Docker 映像名稱保持為 `hwdsl2/whisper-server`；現有設定、API 端點和持久化資料繼續相容。

**功能特性：**

- **相容 OpenAI 的 API：** 提供 `POST /v1/audio/transcriptions` 和 `POST /v1/audio/translations` 端點，可與支援 OpenAI Whisper API 的用戶端整合。
- **私密的本地處理：** 音訊保留在您的伺服器上，不傳送給第三方進行轉錄。
- **說話人分離：** 透過可選的本地 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 擴充，識別每個片段中的說話人。
- **串流結果：** 加入 `stream=true`，即可在解碼時透過 Server-Sent Events（SSE）逐段接收轉錄結果，無需等待整個上傳檔案處理完成。
- **Whisper 模型支援：** 可選擇 `tiny`、`base`、`small`、`medium`、`large-v3`、`large-v3-turbo` 等模型。
- **CPU 與 GPU 支援：** 可使用 CPU 執行，或使用 `:cuda` 映像透過 NVIDIA GPU 加速推論。
- **廣泛的音訊格式支援：** 支援 MP3、M4A、WAV、WebM、OGG、FLAC 及 FFmpeg 支援的其他音訊格式。
- **彈性的輸出格式：** JSON、純文字、詳細 JSON、SRT 字幕和 WebVTT 字幕。
- **離線執行：** 使用預先快取的模型和 `WHISPER_LOCAL_ONLY`，無需存取網際網路即可執行。
- **自動建置：** 映像透過 [GitHub Actions](https://github.com/hwdsl2/scribecrate/actions) 自動建置和發佈，建置工作流程公開可查。

也可作為 [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack/blob/main/README-zh-Hant.md) 的一部分使用，一條命令即可部署完整的自託管 AI 技術堆疊。

> 📘 [The Self-Hosted AI Builder’s Guide](https://books2read.com/aiguide?store=amazon) 是一本關於建置、保護和維運自己的私有 AI 技術堆疊的實用指南。

**另提供：**

- 線上試用：[在 Colab 中開啟](https://selfhostedaistack.com/scribecrate-notebook)——無需 Docker 或安裝
- 相關 AI 服務：[ScribeCrate Live](https://github.com/hwdsl2/scribecrate-live/blob/main/README-zh-Hant.md)、[SpeakCrate](https://github.com/hwdsl2/speakcrate/blob/main/README-zh-Hant.md)、[Embeddings](https://github.com/hwdsl2/docker-embeddings/blob/main/README-zh-Hant.md)、[LiteLLM](https://github.com/hwdsl2/docker-litellm/blob/main/README-zh-Hant.md)、[Ollama](https://github.com/hwdsl2/docker-ollama/blob/main/README-zh-Hant.md)、[ParseCrate](https://github.com/hwdsl2/parsecrate/blob/main/README-zh-Hant.md)、[MCP Gateway](https://github.com/hwdsl2/docker-mcp-gateway/blob/main/README-zh-Hant.md)

## 快速開始

使用以下指令啟動 ScribeCrate：

```bash
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

<details>
<summary><strong>GPU 快速開始（NVIDIA CUDA）</strong></summary>

如果您有 NVIDIA GPU，可使用 `:cuda` 映像進行硬體加速推論：

```bash
docker run \
    --name whisper \
    --restart=always \
    --gpus=all \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server:cuda
```

**需求：** NVIDIA GPU、[NVIDIA 驅動程式](https://www.nvidia.com/en-us/drivers/) 575.57.08+（Linux）或 576.57+（Windows），以及主機上已安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。`:cuda` 映像僅支援 `linux/amd64`。

</details>

**重要：** 此映像執行預設 `base` 模型需要至少 700 MB 可用記憶體。記憶體為 512 MB 或更少的系統不受支援。

> [!NOTE]
> 如需面向網際網路的部署，請使用[反向代理](#使用反向代理)來新增 HTTPS。同時，請將上述 `docker run` 命令中的 `-p 9000:9000` 替換為 `-p 127.0.0.1:9000:9000`，以防止從外部直接存取未加密連接埠。

首次啟動時，Whisper `base` 模型（約 145 MB）將自動下載並快取。查看日誌確認伺服器已就緒：

```bash
docker logs whisper
```

看到 "ScribeCrate transcription server is ready" 後，取得全新安裝在上述持久化資料卷中自動產生的 API 金鑰：

```bash
scribecrate_api_key="$(docker exec whisper whisper_manage --getkey)"
```

轉錄您的第一個音訊檔案，將 `您的伺服器IP` 替換為伺服器位址，並將 `audio.mp3` 替換為您的音訊檔案：

```bash
curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

**回應：**
```json
{"text": "轉錄的文字內容顯示在這裡。"}
```

**提示：** 需要範例音訊檔案進行測試？可以使用來自 [Azure Samples](https://github.com/Azure-Samples/cognitive-services-speech-sdk) 儲存庫的英語語音範例（WAV 格式，MIT 授權）：

```bash
curl -L -o sample_speech.wav \
    "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/sampledata/audiofiles/katiesteve.wav"

curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@sample_speech.wav \
    -F model=whisper-1
```

另外，你也可以在不使用 Docker 的情況下[安裝 Whisper](https://github.com/hwdsl2/whisper-install/blob/main/README-zh-Hant.md)。如需了解更多關於此映像的使用方法，請閱讀以下各節。

## ScribeCrate 與 ScribeCrate Live 的選擇

ScribeCrate Live 是獨立部署的伺服器，由 [WhisperLive](https://github.com/collabora/WhisperLive) 提供支援。

| | **ScribeCrate** | [ScribeCrate Live](https://github.com/hwdsl2/scribecrate-live/blob/main/README-zh-Hant.md) |
|---|---|---|
| **使用情境** | 轉錄完整音訊檔案 | 即時麥克風/音訊串流 |
| **協定** | HTTP REST | WebSocket（串流）+ HTTP REST |
| **延遲** | 處理完成後回傳 JSON；SSE 逐段回傳 | 逐步回傳分段結果 |
| **適合** | 會議錄音、上傳的音訊檔案 | 瀏覽器擷取、RTSP 串流、即時字幕 |
| **映像大小** | ~190 MB（`:cuda` 約 3.1 GB） | ~750 MB（`:cuda` 約 4.5 GB） |

## 社群

- 📬 [取得專案更新與免費部署指南](https://selfhostedstack.beehiiv.com/subscribe?utm_campaign=ai-zh-hant)（每月 1–2 封電子郵件；指南為英文 PDF）
- 💬 加入 [r/selfhostedstack](https://www.reddit.com/r/selfhostedstack/) 社群，參與討論與專案展示
- ⭐ 如果你覺得本專案有用，請為儲存庫加星——這能幫助更多人發現它。

<details>
<summary>自託管 VPN 與網路專案</summary>

- [Setup IPsec VPN](https://github.com/hwdsl2/setup-ipsec-vpn/blob/master/README-zh-Hant.md)
- [Docker 上的 IPsec VPN](https://github.com/hwdsl2/docker-ipsec-vpn-server/blob/master/README-zh-Hant.md)
- [WireGuard](https://github.com/hwdsl2/docker-wireguard/blob/main/README-zh-Hant.md)
- [OpenVPN](https://github.com/hwdsl2/docker-openvpn/blob/main/README-zh-Hant.md)
- [Headscale](https://github.com/hwdsl2/docker-headscale/blob/main/README-zh-Hant.md)

</details>

## 系統需求

- 已安裝 Docker 的 Linux 伺服器（本地或雲端）
- 支援的架構：`amd64`（x86_64）、`arm64`（例如 Raspberry Pi 4/5、AWS Graviton）
- 最低記憶體：預設 `base` 模型約需 700 MB 可用記憶體（請參閱[模型清單](#切換模型)）
- 首次啟動需要存取網際網路以下載模型（之後模型將快取於本機）。使用預先快取的模型並設定 `WHISPER_LOCAL_ONLY=true` 時不需要網路存取。

**GPU 加速（`:cuda` 映像）需求：**

- 支援 CUDA 的 NVIDIA GPU（計算能力 6.0+）
- 主機已安裝 [NVIDIA 驅動程式](https://www.nvidia.com/en-us/drivers/) 575.57.08+（Linux）或 576.57+（Windows）
- 已安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `:cuda` 映像僅支援 `linux/amd64`

如需面向公網部署，請參閱[使用反向代理](#使用反向代理)以啟用 HTTPS。

## 下載

從 [Docker Hub](https://hub.docker.com/r/hwdsl2/whisper-server/) 取得受信任的建置：

```bash
docker pull hwdsl2/whisper-server
```

如需 NVIDIA GPU 加速，請拉取 `:cuda` 標籤：

```bash
docker pull hwdsl2/whisper-server:cuda
```

也可從 [Quay.io](https://quay.io/repository/hwdsl2/whisper-server) 下載：

```bash
docker pull quay.io/hwdsl2/whisper-server
docker image tag quay.io/hwdsl2/whisper-server hwdsl2/whisper-server
```

CPU 映像支援 `linux/amd64` 和 `linux/arm64`。`:cuda` 標籤僅支援 `linux/amd64`。

## 環境變數

所有變數均為選用。掛載 `/var/lib/whisper` 資料卷的新安裝會自動產生 Bearer 令牌。沒有金鑰的既有安裝會保持開放以相容舊行為。

此 Docker 映像檔使用以下變數，可在 `env` 檔案中宣告（參見[範例](whisper.env.example)）：

| 變數 | 說明 | 預設值 |
|---|---|---|
| `WHISPER_MODEL` | 使用的 Whisper 模型。請參閱[模型清單](#切換模型)。 | `base` |
| `WHISPER_LANGUAGE` | 預設轉錄語言。使用 BCP-47 語言代碼（如 `zh`、`en`、`ja`）或 `auto` 自動偵測。 | `auto` |
| `WHISPER_PORT` | API 的 HTTP 連接埠（1–65535）。 | `9000` |
| `WHISPER_DEVICE` | 運算裝置：`cpu`、`cuda` 或 `auto`。使用 `:cuda` 映像時設為 `cuda` 以啟用 GPU 加速。`auto` 自動偵測 GPU，無 GPU 時回退到 CPU。 | `cpu` |
| `WHISPER_COMPUTE_TYPE` | 量化 / 計算類型。CPU 建議 `int8`；CUDA 建議 `float16`。 | `int8`（CPU）/ `float16`（CUDA） |
| `WHISPER_THREADS` | 推論使用的 CPU 執行緒數。設為實體核心數可獲得最佳延遲。 | `2` |
| `WHISPER_API_KEY` | 選用的 Bearer 令牌。新持久化安裝會自動產生。設定後所有請求須包含 `Authorization: Bearer <key>`。明確設定為空可停用驗證。 | 新持久化安裝自動產生 |
| `WHISPER_LOG_LEVEL` | 日誌等級：`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。 | `INFO` |
| `WHISPER_BEAM` | 轉錄和翻譯解碼的 beam 大小。較大的值可能以速度換取精確度。使用 `1` 可獲得最快的貪婪解碼。 | `5` |
| `WHISPER_MAX_REQUEST_BEAM` | 每個請求的 `beam` 覆寫值允許的最大 beam 大小。設為 `0` 可停用此限制。 | `10` |
| `WHISPER_MAX_UPLOAD_MB` | 上傳音訊檔案的最大大小（MB）。超過此限制的請求會返回 HTTP 413。設為 `0` 可停用此限制。 | `1024` |
| `WHISPER_LOCAL_ONLY` | 設為任意非空值（如 `true`）時，停用所有 HuggingFace 模型下載。適用於預先快取模型的離線或隔離網路部署。 | *（未設定）* |
| `WHISPER_WORD_TIMESTAMPS` | 設為 `true` 時，全域啟用詞級時間戳。`verbose_json` 輸出將包含頂層 `words` 陣列，含每個詞的起止時間和置信度。也可透過 `timestamp_granularities[]=word` 按請求啟用。 | *（未設定）* |
| `WHISPER_DIARIZATION` | 設為 `true` 啟用說話人分離，識別每個片段中的說話人。使用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 和 pyannote segmentation-3.0 ONNX 模型（約 45 MB，首次使用時自動下載）。不支援串流模式。 | *（未設定）* |
| `WHISPER_DIARIZE_NUM_SPEAKERS` | 說話人確切數量（如已知）。提高聚類準確性。設為 `-1` 或留空表示自動偵測。 | `-1` |
| `WHISPER_DIARIZE_THRESHOLD` | 自動偵測時使用的聚類閾值。值越小偵測到的說話人越多，值越大偵測到的越少。設定確切說話人數量時會忽略此項。 | `0.5` |
| `WHISPER_DISABLE_USAGE_COUNTS` | 設為 `1` 可停用匿名彙總使用計數。 | *（未設定）* |

**注：** 在 `env` 檔案中，值可用單引號括起，例如 `VAR='value'`。`=` 兩側不要有空格。如更改 `WHISPER_PORT`，請相應更新 `docker run` 指令中的 `-p` 參數。

使用 `env` 檔案的範例：

```bash
cp whisper.env.example whisper.env
# 編輯 whisper.env 進行設定，然後：
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -v ./whisper.env:/whisper.env:ro \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

`env` 檔案以綁定掛載方式傳入容器，每次重啟時自動生效，無需重建容器。

<details>
<summary>或透過 <code>--env-file</code> 傳入</summary>

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

## 使用 docker-compose

```bash
cp whisper.env.example whisper.env
# 依需求編輯 whisper.env，然後：
docker compose up -d
docker logs whisper
```

範例 `docker-compose.yml`（已包含在專案中）：

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # 如使用主機反向代理，改為 "127.0.0.1:9000:9000/tcp"
    volumes:
      - whisper-data:/var/lib/whisper
      - ./whisper.env:/whisper.env:ro

volumes:
  whisper-data:
    name: whisper-data
```

> [!NOTE]
> 如需面向公網部署，請使用[反向代理](#使用反向代理)啟用 HTTPS。同時，請將 `docker-compose.yml` 中的 `"9000:9000/tcp"` 改為 `"127.0.0.1:9000:9000/tcp"`，以防止未加密連接埠被直接存取。

<details>
<summary><strong>使用 docker-compose 部署 GPU（NVIDIA CUDA）</strong></summary>

專案提供了獨立的 `docker-compose.cuda.yml` 用於 GPU 部署：

```bash
cp whisper.env.example whisper.env
# 依需求編輯 whisper.env，然後：
docker compose -f docker-compose.cuda.yml up -d
docker logs whisper
```

範例 `docker-compose.cuda.yml`（已包含在專案中）：

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server:cuda
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # 如使用主機反向代理，改為 "127.0.0.1:9000:9000/tcp"
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

## API 參考

ScribeCrate 提供相容 OpenAI [音訊轉錄](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)和[音訊翻譯](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create)介面的端點。使用 OpenAI SDK 的用戶端應設定基礎 URL，並使用您的 ScribeCrate API 金鑰：

**身分驗證：** 全新的持久化安裝需要 API 金鑰。使用以下指令取得金鑰，供 SDK 設定和下面的 `curl` 範例使用：

```bash
scribecrate_api_key="$(docker exec whisper whisper_manage --getkey)"

export OPENAI_BASE_URL="http://您的伺服器IP:9000/v1"
export OPENAI_API_KEY="$scribecrate_api_key"
```

如果已停用 API 金鑰驗證，請省略 curl 範例中的 `Authorization` 標頭。OpenAI SDK 用戶端仍要求提供非空金鑰；此時請設定 `OPENAI_API_KEY=unused`。

說話人分離啟用時是本地 sherpa-onnx 擴充，並不等同於 OpenAI 的說話人分離模型。OpenAI 專用的轉錄選項（如 `gpt-4o-transcribe-diarize`、`response_format=diarized_json`、`include=logprobs`、`chunking_strategy`、`known_speaker_names` 和 `known_speaker_references`）不受支援，並會回傳 `400`。

### 轉錄音訊

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**參數：**

| 參數 | 類型 | 必填 | 說明 |
|---|---|---|---|
| `file` | 檔案 | ✅ | 音訊檔案。支援格式：`mp3`、`mp4`、`m4a`、`wav`、`webm`、`ogg`、`flac` 及 ffmpeg 支援的所有格式。 |
| `model` | 字串 | ✅ | 傳入 `whisper-1`（值被接受，但始終使用目前啟用的模型）。 |
| `language` | 字串 | — | BCP-47 語言代碼。覆寫本次請求的 `WHISPER_LANGUAGE` 設定。 |
| `prompt` | 字串 | — | 選用文字，用於引導模型風格或延續前一段內容。 |
| `response_format` | 字串 | — | 輸出格式，預設為 `json`。請參閱[回應格式](#回應格式)。`stream=true` 時忽略此參數。不支援 OpenAI 專用的 `diarized_json`。 |
| `temperature` | 浮點數 | — | 採樣溫度（0–1），預設為 `0`。 |
| `stream` | 布林值 | — | 啟用 SSE 串流。為 `true` 時，段落將在解碼時以 `text/event-stream` 事件形式回傳。預設為 `false`。 |
| `timestamp_granularities[]` | 陣列 | — | 時間戳粒度。值：`word`、`segment`。包含 `word` 時，`verbose_json` 輸出包含頂層 `words` 陣列。預設：`["segment"]`。 |

**本地 faster-whisper 擴充：** 可設定 `beam`，為單一轉錄或翻譯請求覆寫 `WHISPER_BEAM`。這不是 OpenAI API 架構的一部分，因此不要將其傳送到託管的 OpenAI API 或嚴格相容 OpenAI 的閘道。每請求預設上限為 `10`（`WHISPER_MAX_REQUEST_BEAM`）；將該變數設為 `0` 可停用上限。Beam 搜尋主要影響 `temperature=0` 時的確定性解碼。

**範例：**

```bash
curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@meeting.m4a \
    -F model=whisper-1 \
    -F language=zh
```

使用 API 金鑰驗證：

```bash
curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer your_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

### 回應格式

| `response_format` | 說明 |
|---|---|
| `json` | `{"text": "..."}` — 預設，與 OpenAI 基本回應格式一致 |
| `text` | 純文字，無 JSON 封裝 |
| `verbose_json` | 完整 JSON，包含語言、時長、逐段時間戳記及對數機率 |
| `srt` | SubRip 字幕格式（`.srt`） |
| `vtt` | WebVTT 字幕格式（`.vtt`） |

**範例 — 串流接收解碼段落：**

```bash
curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@long-audio.mp3 \
    -F model=whisper-1 \
    -F stream=true
```

**SSE 回應**（使用 [OpenAI 串流轉錄協定](https://developers.openai.com/api/docs/guides/speech-to-text#streaming)）：

```
data: {"type":"transcript.text.delta","delta":"您好，最近好嗎？"}

data: {"type":"transcript.text.delta","delta":" 我很好，謝謝。"}

data: {"type":"transcript.text.done","text":"您好，最近好嗎？ 我很好，謝謝。"}

data: [DONE]
```

上傳後第一個增量文字通常在 1–3 秒內到達。每個 `transcript.text.delta` 事件包含剛解碼的段落的增量文字。最後的 `transcript.text.done` 事件包含與標準 `json` 回應等效的完整轉錄文字。

<details>
<summary><strong>範例 — 透過瀏覽器 <code>fetch</code> 進行串流傳輸</strong></summary>

```javascript
const form = new FormData();
form.append("file", audioBlob, "audio.webm");
form.append("model", "whisper-1");
form.append("stream", "true");

const res = await fetch("http://您的伺服器IP:9000/v1/audio/transcriptions", {
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

**範例 — 取得 SRT 字幕：**

```bash
curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@video.mp4 \
    -F model=whisper-1 \
    -F response_format=srt
```

**範例 — 含時間戳記的詳細 JSON：**

```bash
curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json
```

**範例 — 含詞級時間戳的詳細 JSON：**

```bash
curl http://您的伺服器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json \
    -F "timestamp_granularities[]=word"
```

當 `timestamp_granularities[]` 包含 `word` 時，`verbose_json` 回應包含頂層 `words` 陣列：

```json
{
  "word": "hello",
  "start": 0.5,
  "end": 0.8,
  "probability": 0.98
}
```

### 翻譯音訊

```
POST /v1/audio/translations
Content-Type: multipart/form-data
```

將任意語言的音訊翻譯為英文文字。與 [OpenAI 音訊翻譯端點](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create)相容。接受常見的翻譯參數。輸出始終為英文。

> **注意：** 僅英語（`.en`）模型不支援翻譯。請使用多語言模型（如 `base`、`small`、`large-v3-turbo`）。

**範例：**

```bash
curl http://您的伺服器IP:9000/v1/audio/translations \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@french_audio.mp3 \
    -F model=whisper-1
```

### 列出模型

```
GET /v1/models
```

以 OpenAI 相容格式回傳目前啟用的模型。

```bash
curl http://您的伺服器IP:9000/v1/models -H "Authorization: Bearer $scribecrate_api_key"
```

### 互動式 API 文件

可在以下網址存取互動式 Swagger UI：

```
http://您的伺服器IP:9000/docs
```

## 持久化資料

所有伺服器資料存儲在 Docker 資料卷（容器內的 `/var/lib/whisper`）中：

```
/var/lib/whisper/
├── models--Systran--faster-whisper-*/   # 快取的 Whisper 模型檔案（從 HuggingFace 下載）
├── .port                 # 目前連接埠（供 whisper_manage 使用）
├── .model                # 目前模型名稱（供 whisper_manage 使用）
└── .server_addr          # 快取的伺服器 IP（供 whisper_manage 使用）
```

請備份 Docker 資料卷以保留已下載的模型。模型檔案較大（145 MB – 3 GB），首次啟動時下載可能需要數分鐘；保留資料卷可避免在重建容器時重新下載。

**提示：** `/var/lib/whisper` 資料卷與 `docker-whisper-live` 的 `/var/lib/whisper-live` 資料卷使用相同的 HuggingFace 快取配置。如果已透過 `docker-whisper-live` 下載了模型，可繫結掛載相同的資料卷目錄以避免重複下載。

## 管理伺服器

在執行中的容器內使用 `whisper_manage` 來查看和管理伺服器。

**顯示伺服器資訊：**

```bash
docker exec whisper whisper_manage --showinfo
```

**列出可用模型：**

```bash
docker exec whisper whisper_manage --listmodels
```

**預先下載模型：**

```bash
docker exec whisper whisper_manage --downloadmodel large-v3-turbo
```

## 切換模型

要更換啟用中的模型：

1. *（選用但建議）* 在伺服器執行時預先下載新模型：
   ```bash
   docker exec whisper whisper_manage --downloadmodel large-v3-turbo
   ```

2. 在 `whisper.env` 檔案中更新 `WHISPER_MODEL`（或在 `docker run` 指令中加入 `-e WHISPER_MODEL=large-v3-turbo`）。

3. 重新啟動容器：
   ```bash
   docker restart whisper
   ```

**可用模型：**

| 模型 | 磁碟占用 | 記憶體（約） | 說明 |
|---|---|---|---|
| `tiny` | ~75 MB | ~250 MB | 最快；精確度較低 |
| `tiny.en` | ~75 MB | ~250 MB | 僅英語 |
| `base` | ~145 MB | ~700 MB | 良好平衡 — **預設** |
| `base.en` | ~145 MB | ~700 MB | 僅英語 |
| `small` | ~465 MB | ~1.5 GB | 更高精確度 |
| `small.en` | ~465 MB | ~1.5 GB | 僅英語 |
| `medium` | ~1.5 GB | ~5 GB | 高精確度 |
| `medium.en` | ~1.5 GB | ~5 GB | 僅英語 |
| `large-v1` | ~3 GB | ~10 GB | 舊版大型模型 |
| `large-v2` | ~3 GB | ~10 GB | 非常高精確度 |
| `large-v3` | ~3 GB | ~10 GB | 最高精確度 |
| `large-v3-turbo` | ~1.6 GB | ~6 GB | 高速 + 高精確度 ⭐ |
| `turbo` | ~1.6 GB | ~6 GB | `large-v3-turbo` 的別名 |

> **提示：** `large-v3-turbo` 的精確度接近 `large-v3`，但資源消耗約為其一半。對於大多數正式部署，這是從 `base` 升級的推薦選擇。

記憶體數值為近似值，基於 INT8 量化（預設）。模型快取於 `/var/lib/whisper` Docker 資料卷中，僅需下載一次。

## 保護你的伺服器

如果你的 ScribeCrate 伺服器可從公用網際網路存取 —— 即使只是短暫可達 —— 也請至少採取以下保護措施。轉錄對 CPU/GPU 資源消耗較大，未做身分驗證的介面可能被濫用，浪費你的運算資源。

**1. 使用 API 金鑰。** 掛載 `/var/lib/whisper` 資料卷的新安裝會自動產生 API 金鑰。可用 `docker exec whisper whisper_manage --showkey` 查看；腳本中可用 `docker exec whisper whisper_manage --getkey`。沒有金鑰的既有安裝會保持開放以相容舊行為；也可以在 `env` 檔案中設定 `WHISPER_API_KEY` 手動啟用驗證。所有已驗證請求必須包含 `Authorization: Bearer <key>`。

```bash
# 產生 32 位元組的隨機金鑰
openssl rand -hex 32
```

**2. 在反向代理後方時繫結到 localhost。** 將 `-p 9000:9000` 替換為 `-p 127.0.0.1:9000:9000`（或在 `docker-compose.yml` 中將 `"9000:9000/tcp"` 改為 `"127.0.0.1:9000:9000/tcp"`），使未加密連接埠無法從主機外部直接存取。

**3. 限制上傳大小。** 伺服器會拒絕超過 `WHISPER_MAX_UPLOAD_MB`（預設 `1024`）的上傳。對於面向網際網路的部署，還應設定反向代理在請求到達應用程式前拒絕過大的上傳（例如 nginx `client_max_body_size 100M;`）。

**4. 注意日誌等級。** `WHISPER_LOG_LEVEL=DEBUG` 可能會將轉錄文字寫入日誌。在共用系統上請保持 `INFO` 或更高等級。

**5. 瀏覽器呼叫時在代理處啟用 CORS。** 本伺服器預設不設定 `Access-Control-Allow-Origin` 回應標頭；若需在不同來源的網頁中直接呼叫本 API，請在反向代理處新增 CORS 標頭。

**6. 考慮限流。** 在伺服器前部署限流（如 nginx `limit_req_zone`、Caddy `rate_limit`），限制每個用戶端 IP 的並行轉錄請求數。

## 使用反向代理

如需面向公網部署，可在 ScribeCrate 前置反向代理處理 HTTPS 終止。在本地或可信網路中使用無需 HTTPS，但將 API 端點暴露在公網時建議啟用 HTTPS。

從反向代理存取 ScribeCrate 容器時使用以下位址之一：

- **`whisper:9000`** — 如果反向代理作為容器執行在與 ScribeCrate **同一 Docker 網路**中（例如定義在同一 `docker-compose.yml` 中）。
- **`127.0.0.1:9000`** — 如果反向代理執行在**主機上**且連接埠 `9000` 已發布（預設 `docker-compose.yml` 會發布該連接埠）。

**使用 [Caddy](https://caddyserver.com/docs/)（[Docker 映像檔](https://hub.docker.com/_/caddy)）的範例**（自動 Let's Encrypt TLS，反向代理在同一 Docker 網路中）：

`Caddyfile`：
```
whisper.example.com {
  reverse_proxy whisper:9000
}
```

**使用 nginx 的範例**（反向代理執行在主機上）：

```nginx
server {
    listen 443 ssl;
    server_name whisper.example.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # 音訊檔案可能較大——依需求調整上傳限制
    client_max_body_size 100M;

    location / {
        proxy_pass         http://127.0.0.1:9000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;       # SSE 串流所需
        proxy_read_timeout 300s;
    }
}
```

## 更新 Docker 映像檔

如需更新 Docker 映像檔和容器，首先[下載](#下載)最新版本：

```bash
docker pull hwdsl2/whisper-server
```

如果映像檔已是最新版本，您將看到：

```
Status: Image is up to date for hwdsl2/whisper-server:latest
```

否則將下載最新版本。刪除並重新建立容器：

```bash
docker rm -f whisper
# 然後使用相同的資料卷和連接埠重新執行快速開始中的 docker run 指令。
```

您下載的模型將保留在 `whisper-data` 資料卷中。

## 與其他 AI 服務搭配使用

ScribeCrate 可作為更廣泛的自託管 AI 設定中的語音轉文字服務。

如需完整和輕量級 Docker Compose 技術堆疊、手動 `docker run` 範例，以及結合 SpeakCrate、Embeddings、LiteLLM、Ollama、ParseCrate 和 MCP Gateway 的語音/RAG/MCP 流水線範例，請參閱 [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack/blob/main/README-zh-Hant.md)。

## 說話人分離

說話人分離功能識別每個轉錄片段中*誰在說話*。這是一個本地擴充，基於 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 使用匯出為 ONNX 格式的 pyannote segmentation-3.0 模型。

**啟用說話人分離：**

```bash
# 在 whisper.env 中：
WHISPER_DIARIZATION=true
```

ONNX 模型（共約 45 MB）在首次使用時自動下載並快取到 `/var/lib/whisper` 資料卷。預先下載：

```bash
docker exec whisper whisper_manage --downloaddiarize
```

**啟用說話人分離後的輸出：**

`verbose_json` 在每個片段中新增 `speaker` 欄位：

```json
{
  "segments": [
    {"id": 0, "start": 1.0, "end": 3.5, "text": "我們下週發布吧。", "speaker": "SPEAKER_00"},
    {"id": 1, "start": 4.0, "end": 6.2, "text": "我覺得 QA 還需要兩天。", "speaker": "SPEAKER_01"}
  ]
}
```

`srt` 和 `vtt` 在文字前加上說話人標籤：

```
1
00:00:01,000 --> 00:00:03,500
[SPEAKER_00] 我們下週發布吧。

2
00:00:04,000 --> 00:00:06,200
[SPEAKER_01] 我覺得 QA 還需要兩天。
```

`text` 格式在說話人變化時顯示標籤：

```
[SPEAKER_00] 我們下週發布吧。
[SPEAKER_01] 我覺得 QA 還需要兩天。
```

**注意事項：**

> [!IMPORTANT]
> 說話人分離需要完整音訊分析，**不支援串流模式**（`stream=true`）。兩者同時啟用時，說話人分離會被靜默跳過。

- 如果已知確切說話人數量，設定 `WHISPER_DIARIZE_NUM_SPEAKERS` 可提高準確性。
- 說話人分離在轉錄完成後執行，會增加與音訊時長成正比的少量處理時間。

## 使用計數

此映像使用公開的 GitHub Release 資源下載次數進行匿名彙總使用計數。計數是近似值，不代表唯一使用者或活躍安裝。映像不會傳送遙測負載，也不會使用私有收集器。僅當伺服器成功啟動且掛載了 `/var/lib/whisper` 卷後，才會以盡力而為方式計數；當該持久化安裝首次執行不同映像建置時，也會再次計數。若要退出，請設定 `WHISPER_DISABLE_USAGE_COUNTS=1`。

## 技術細節

- 基礎映像檔：`:latest` 使用 `python:3.12-slim`；`:cuda` 使用 `nvidia/cuda`
- 執行時：Python 3（虛擬環境位於 `/opt/venv`）
- STT 引擎：[faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CTranslate2（CPU 預設 INT8，CUDA 預設 FP16）
- API 框架：[FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- 音訊解碼：[PyAV](https://github.com/PyAV-Org/PyAV)（內建 FFmpeg 函式庫）
- 資料目錄：`/var/lib/whisper`（Docker 資料卷）
- 模型儲存：HuggingFace Hub 格式，存儲在資料卷中——下載一次，重啟後複用

## 授權條款

**注：** 預構建映像檔中包含的軟體元件（如 faster-whisper 及其相依套件）均受各自版權持有者所選授權條款約束。使用預構建映像檔時，使用者有責任確保其使用方式符合映像檔內所有軟體的相關授權條款要求。

著作權所有 (C) 2026 Lin Song   
本作品採用 [MIT 授權條款](https://opensource.org/licenses/MIT)。

**faster-whisper** 著作權歸 SYSTRAN 所有，依據 [MIT 授權條款](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)發行。

ScribeCrate 是使用 Whisper 模型的獨立伺服器，與 OpenAI 或 SYSTRAN 無關聯，未獲其背書或贊助。

[English](README.md) | [简体中文](README-zh.md) | [繁體中文](README-zh-Hant.md) | [Русский](README-ru.md)

# ScribeCrate

**开源、自托管的音频转录 API。**

[![构建状态](https://github.com/hwdsl2/scribecrate/actions/workflows/main.yml/badge.svg)](https://github.com/hwdsl2/scribecrate/actions/workflows/main.yml) &nbsp;[![Docker Pulls](https://raw.githubusercontent.com/hwdsl2/badges/main/img/docker-pulls-whisper-server.svg)](https://hub.docker.com/r/hwdsl2/whisper-server) &nbsp;[![License: MIT](docs/images/license.svg)](https://opensource.org/licenses/MIT) &nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://selfhostedaistack.com/scribecrate-notebook)

ScribeCrate 是一个自托管的语音转文字 API，用于转录音频、生成字幕，并将语音翻译为英语。它使用 [Whisper](https://github.com/openai/whisper) 模型，由 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 驱动，提供兼容 OpenAI 的转录和翻译接口。可通过 Docker 部署，使用 CPU 或 NVIDIA GPU 运行。

> 本项目原名 **docker-whisper**，仍由 [hwdsl2](https://github.com/hwdsl2) 维护。Docker 镜像名称保持为 `hwdsl2/whisper-server`；现有配置、API 接口和持久化数据继续兼容。

**功能特性：**

- **兼容 OpenAI 的 API：** 提供 `POST /v1/audio/transcriptions` 和 `POST /v1/audio/translations` 接口，可与支持 OpenAI Whisper API 的客户端集成。
- **私密的本地处理：** 音频保留在您的服务器上，不发送给第三方进行转录。
- **说话人分离：** 通过可选的本地 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 扩展，识别每个片段中的说话人。
- **流式结果：** 添加 `stream=true`，即可在解码时通过 Server-Sent Events（SSE）逐段接收转录结果，无需等待整个上传文件处理完成。
- **Whisper 模型支持：** 可选择 `tiny`、`base`、`small`、`medium`、`large-v3`、`large-v3-turbo` 等模型。
- **CPU 和 GPU 支持：** 可使用 CPU 运行，或使用 `:cuda` 镜像通过 NVIDIA GPU 加速推理。
- **广泛的音频格式支持：** 支持 MP3、M4A、WAV、WebM、OGG、FLAC 及 FFmpeg 支持的其他音频格式。
- **灵活的输出格式：** JSON、纯文本、详细 JSON、SRT 字幕和 WebVTT 字幕。
- **离线运行：** 使用预先缓存的模型和 `WHISPER_LOCAL_ONLY`，无需访问互联网即可运行。
- **自动构建：** 镜像通过 [GitHub Actions](https://github.com/hwdsl2/scribecrate/actions) 自动构建和发布，构建工作流公开可查。

也可作为 [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack/blob/main/README-zh.md) 的一部分使用，一条命令即可部署完整的自托管 AI 技术栈。

> 📘 [The Self-Hosted AI Builder’s Guide](https://books2read.com/aiguide?store=amazon) 是一本关于构建、保护和运维自己的私有 AI 技术栈的实用指南。

**另提供：**

- 在线试用：[在 Colab 中打开](https://selfhostedaistack.com/scribecrate-notebook)——无需 Docker 或安装
- 相关 AI 服务：[ScribeCrate Live](https://github.com/hwdsl2/scribecrate-live/blob/main/README-zh.md)、[SpeakCrate](https://github.com/hwdsl2/speakcrate/blob/main/README-zh.md)、[Embeddings](https://github.com/hwdsl2/docker-embeddings/blob/main/README-zh.md)、[LiteLLM](https://github.com/hwdsl2/docker-litellm/blob/main/README-zh.md)、[Ollama](https://github.com/hwdsl2/docker-ollama/blob/main/README-zh.md)、[ParseCrate](https://github.com/hwdsl2/parsecrate/blob/main/README-zh.md)、[MCP Gateway](https://github.com/hwdsl2/docker-mcp-gateway/blob/main/README-zh.md)

## 快速开始

使用以下命令启动 ScribeCrate：

```bash
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

<details>
<summary><strong>GPU 快速开始（NVIDIA CUDA）</strong></summary>

如果您有 NVIDIA GPU，可使用 `:cuda` 镜像进行硬件加速推理：

```bash
docker run \
    --name whisper \
    --restart=always \
    --gpus=all \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server:cuda
```

**要求：** NVIDIA GPU、[NVIDIA 驱动](https://www.nvidia.com/en-us/drivers/) 575.57.08+（Linux）或 576.57+（Windows），以及主机上已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。`:cuda` 镜像仅支持 `linux/amd64`。

</details>

**重要：** 此镜像运行默认 `base` 模型需要至少 700 MB 可用内存。内存为 512 MB 或更少的系统不受支持。

> [!NOTE]
> 如需面向互联网的部署，请使用[反向代理](#使用反向代理)来添加 HTTPS。同时，请将上述 `docker run` 命令中的 `-p 9000:9000` 替换为 `-p 127.0.0.1:9000:9000`，以防止从外部直接访问未加密端口。

首次启动时，Whisper `base` 模型（约 145 MB）将自动下载并缓存。查看日志确认服务器已就绪：

```bash
docker logs whisper
```

看到 "ScribeCrate transcription server is ready" 后，获取全新安装在上述持久化数据卷中自动生成的 API 密钥：

```bash
scribecrate_api_key="$(docker exec whisper whisper_manage --getkey)"
```

转录您的第一个音频文件，将 `您的服务器IP` 替换为服务器地址，并将 `audio.mp3` 替换为您的音频文件：

```bash
curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

**响应：**
```json
{"text": "转录的文字内容显示在这里。"}
```

**提示：** 需要示例音频文件进行测试？可以使用来自 [Azure Samples](https://github.com/Azure-Samples/cognitive-services-speech-sdk) 仓库的英语语音示例（WAV 格式，MIT 许可证）：

```bash
curl -L -o sample_speech.wav \
    "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/sampledata/audiofiles/katiesteve.wav"

curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@sample_speech.wav \
    -F model=whisper-1
```

另外，你也可以在不使用 Docker 的情况下[安装 Whisper](https://github.com/hwdsl2/whisper-install/blob/main/README-zh.md)。如需了解更多关于此镜像的使用方法，请阅读以下各节。

## ScribeCrate 与 ScribeCrate Live 的选择

ScribeCrate Live 是独立部署的服务器，由 [WhisperLive](https://github.com/collabora/WhisperLive) 提供支持。

| | **ScribeCrate** | [ScribeCrate Live](https://github.com/hwdsl2/scribecrate-live/blob/main/README-zh.md) |
|---|---|---|
| **使用场景** | 转录完整音频文件 | 实时麦克风/音频流 |
| **协议** | HTTP REST | WebSocket（流式）+ HTTP REST |
| **延迟** | 处理完成后返回 JSON；SSE 逐段返回 | 逐步返回分段结果 |
| **适合** | 会议录音、上传的音频文件 | 浏览器采集、RTSP 流、实时字幕 |
| **镜像大小** | ~190 MB（`:cuda` 约 3.1 GB） | ~750 MB（`:cuda` 约 4.5 GB） |

## 社区

- 📬 [获取项目更新和免费部署指南](https://selfhostedstack.beehiiv.com/subscribe?utm_campaign=ai-zh)（每月 1–2 封邮件；指南为英文 PDF）
- 💬 加入 [r/selfhostedstack](https://www.reddit.com/r/selfhostedstack/) 社区，参与讨论和项目展示
- ⭐ 如果你觉得本项目有用，请为仓库加星——这有助于让更多人发现它。

<details>
<summary>自托管 VPN 和网络项目</summary>

- [Setup IPsec VPN](https://github.com/hwdsl2/setup-ipsec-vpn/blob/master/README-zh.md)
- [Docker 上的 IPsec VPN](https://github.com/hwdsl2/docker-ipsec-vpn-server/blob/master/README-zh.md)
- [WireGuard](https://github.com/hwdsl2/docker-wireguard/blob/main/README-zh.md)
- [OpenVPN](https://github.com/hwdsl2/docker-openvpn/blob/main/README-zh.md)
- [Headscale](https://github.com/hwdsl2/docker-headscale/blob/main/README-zh.md)

</details>

## 系统要求

- 已安装 Docker 的 Linux 服务器（本地或云端）
- 支持的架构：`amd64`（x86_64）、`arm64`（例如 Raspberry Pi 4/5、AWS Graviton）
- 最低内存：默认 `base` 模型约需 700 MB 可用内存（请参阅[模型列表](#切换模型)）
- 首次启动需要访问互联网以下载模型（之后模型将缓存在本地）。使用预先缓存的模型并设置 `WHISPER_LOCAL_ONLY=true` 时不需要网络访问。

**GPU 加速（`:cuda` 镜像）要求：**

- 支持 CUDA 的 NVIDIA GPU（计算能力 6.0+）
- 主机已安装 [NVIDIA 驱动](https://www.nvidia.com/en-us/drivers/) 575.57.08+（Linux）或 576.57+（Windows）
- 已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `:cuda` 镜像仅支持 `linux/amd64`

如需面向公网部署，请参阅[使用反向代理](#使用反向代理)以启用 HTTPS。

## 下载

从 [Docker Hub](https://hub.docker.com/r/hwdsl2/whisper-server/) 获取可信构建：

```bash
docker pull hwdsl2/whisper-server
```

如需 NVIDIA GPU 加速，请拉取 `:cuda` 标签：

```bash
docker pull hwdsl2/whisper-server:cuda
```

也可从 [Quay.io](https://quay.io/repository/hwdsl2/whisper-server) 下载：

```bash
docker pull quay.io/hwdsl2/whisper-server
docker image tag quay.io/hwdsl2/whisper-server hwdsl2/whisper-server
```

CPU 镜像支持 `linux/amd64` 和 `linux/arm64`。`:cuda` 标签仅支持 `linux/amd64`。

## 环境变量

所有变量均为可选。挂载 `/var/lib/whisper` 数据卷的新安装会自动生成 Bearer 令牌。没有密钥的既有安装会保持开放以兼容旧行为。

此 Docker 镜像使用以下变量，可在 `env` 文件中声明（参见[示例](whisper.env.example)）：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `WHISPER_MODEL` | 使用的 Whisper 模型。请参阅[模型列表](#切换模型)。 | `base` |
| `WHISPER_LANGUAGE` | 默认转录语言。使用 BCP-47 语言代码（如 `zh`、`en`、`ja`）或 `auto` 自动检测。 | `auto` |
| `WHISPER_PORT` | API 的 HTTP 端口（1–65535）。 | `9000` |
| `WHISPER_DEVICE` | 计算设备：`cpu`、`cuda` 或 `auto`。使用 `:cuda` 镜像时设为 `cuda` 以启用 GPU 加速。`auto` 自动检测 GPU，无 GPU 时回退到 CPU。 | `cpu` |
| `WHISPER_COMPUTE_TYPE` | 量化 / 计算类型。CPU 推荐 `int8`；CUDA 推荐 `float16`。 | `int8`（CPU）/ `float16`（CUDA） |
| `WHISPER_THREADS` | 推理使用的 CPU 线程数。设为物理核心数可获得最佳延迟。 | `2` |
| `WHISPER_API_KEY` | 可选的 Bearer 令牌。新持久化安装会自动生成。设置后所有请求须包含 `Authorization: Bearer <key>`。显式设置为空可禁用认证。 | 新持久化安装自动生成 |
| `WHISPER_LOG_LEVEL` | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。 | `INFO` |
| `WHISPER_BEAM` | 转录和翻译解码的 beam 大小。较大的值可能以速度换取精度。使用 `1` 可获得最快的贪婪解码。 | `5` |
| `WHISPER_MAX_REQUEST_BEAM` | 每个请求的 `beam` 覆盖值允许的最大 beam 大小。设为 `0` 可禁用此限制。 | `10` |
| `WHISPER_MAX_UPLOAD_MB` | 上传音频文件的最大大小（MB）。超过此限制的请求会返回 HTTP 413。设为 `0` 可禁用此限制。 | `1024` |
| `WHISPER_LOCAL_ONLY` | 设为任意非空值（如 `true`）时，禁止所有 HuggingFace 模型下载。适用于预先缓存模型的离线或隔离网络部署。 | *（未设置）* |
| `WHISPER_WORD_TIMESTAMPS` | 设为 `true` 时，全局启用词级时间戳。`verbose_json` 输出将包含顶层 `words` 数组，含每个词的起止时间和置信度。也可通过 `timestamp_granularities[]=word` 按请求启用。 | *（未设置）* |
| `WHISPER_DIARIZATION` | 设为 `true` 启用说话人分离，识别每个片段中的说话人。使用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 和 pyannote segmentation-3.0 ONNX 模型（约 45 MB，首次使用时自动下载）。不支持流式模式。 | *（未设置）* |
| `WHISPER_DIARIZE_NUM_SPEAKERS` | 说话人确切数量（如已知）。提高聚类准确性。设为 `-1` 或留空表示自动检测。 | `-1` |
| `WHISPER_DIARIZE_THRESHOLD` | 自动检测时使用的聚类阈值。值越小检测到的说话人越多，值越大检测到的越少。设置确切说话人数量时会忽略此项。 | `0.5` |
| `WHISPER_DISABLE_USAGE_COUNTS` | 设为 `1` 可禁用匿名聚合使用计数。 | *（未设置）* |

**注：** 在 `env` 文件中，值可用单引号括起，例如 `VAR='value'`。`=` 两侧不要有空格。如更改 `WHISPER_PORT`，请相应更新 `docker run` 命令中的 `-p` 参数。

使用 `env` 文件的示例：

```bash
cp whisper.env.example whisper.env
# 编辑 whisper.env 配置您的设置，然后：
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -v ./whisper.env:/whisper.env:ro \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

`env` 文件以绑定挂载方式传入容器，每次重启时自动生效，无需重建容器。

<details>
<summary>也可通过 <code>--env-file</code> 传入</summary>

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
# 按需编辑 whisper.env，然后：
docker compose up -d
docker logs whisper
```

示例 `docker-compose.yml`（已包含在项目中）：

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # 如使用主机反向代理，改为 "127.0.0.1:9000:9000/tcp"
    volumes:
      - whisper-data:/var/lib/whisper
      - ./whisper.env:/whisper.env:ro

volumes:
  whisper-data:
    name: whisper-data
```

> [!NOTE]
> 如需面向公网部署，请使用[反向代理](#使用反向代理)启用 HTTPS。同时，请将 `docker-compose.yml` 中的 `"9000:9000/tcp"` 改为 `"127.0.0.1:9000:9000/tcp"`，以防止未加密端口被直接访问。

<details>
<summary><strong>使用 docker-compose 部署 GPU（NVIDIA CUDA）</strong></summary>

项目提供了单独的 `docker-compose.cuda.yml` 用于 GPU 部署：

```bash
cp whisper.env.example whisper.env
# 按需编辑 whisper.env，然后：
docker compose -f docker-compose.cuda.yml up -d
docker logs whisper
```

示例 `docker-compose.cuda.yml`（已包含在项目中）：

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server:cuda
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # 如使用主机反向代理，改为 "127.0.0.1:9000:9000/tcp"
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

## API 参考

ScribeCrate 提供兼容 OpenAI [音频转录](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)和[音频翻译](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create)接口的端点。使用 OpenAI SDK 的客户端应配置基础 URL，并使用您的 ScribeCrate API 密钥：

**身份验证：** 全新的持久化安装需要 API 密钥。使用以下命令获取密钥，供 SDK 配置和下面的 `curl` 示例使用：

```bash
scribecrate_api_key="$(docker exec whisper whisper_manage --getkey)"

export OPENAI_BASE_URL="http://您的服务器IP:9000/v1"
export OPENAI_API_KEY="$scribecrate_api_key"
```

如果已禁用 API 密钥认证，请省略 curl 示例中的 `Authorization` 请求头。OpenAI SDK 客户端仍要求提供非空密钥；此时请设置 `OPENAI_API_KEY=unused`。

说话人分离启用时是本地 sherpa-onnx 扩展，并不等同于 OpenAI 的说话人分离模型。OpenAI 专用的转录选项（如 `gpt-4o-transcribe-diarize`、`response_format=diarized_json`、`include=logprobs`、`chunking_strategy`、`known_speaker_names` 和 `known_speaker_references`）不受支持，并会返回 `400`。

### 转录音频

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | 文件 | ✅ | 音频文件。支持格式：`mp3`、`mp4`、`m4a`、`wav`、`webm`、`ogg`、`flac` 及 ffmpeg 支持的所有格式。 |
| `model` | 字符串 | ✅ | 传入 `whisper-1`（值被接受，但始终使用当前活跃模型）。 |
| `language` | 字符串 | — | BCP-47 语言代码。覆盖本次请求的 `WHISPER_LANGUAGE` 设置。 |
| `prompt` | 字符串 | — | 可选文本，用于引导模型风格或延续前一段内容。 |
| `response_format` | 字符串 | — | 输出格式，默认为 `json`。请参阅[响应格式](#响应格式)。`stream=true` 时忽略此参数。不支持 OpenAI 专用的 `diarized_json`。 |
| `temperature` | 浮点数 | — | 采样温度（0–1），默认为 `0`。 |
| `stream` | 布尔值 | — | 启用 SSE 流式传输。为 `true` 时，段落将在解码时以 `text/event-stream` 事件形式返回。默认为 `false`。 |
| `timestamp_granularities[]` | 数组 | — | 时间戳粒度。值：`word`、`segment`。包含 `word` 时，`verbose_json` 输出包含顶层 `words` 数组。默认：`["segment"]`。 |

**本地 faster-whisper 扩展：** 可设置 `beam`，为单个转录或翻译请求覆盖 `WHISPER_BEAM`。这不是 OpenAI API 架构的一部分，因此不要将其发送到托管的 OpenAI API 或严格兼容 OpenAI 的网关。每请求默认上限为 `10`（`WHISPER_MAX_REQUEST_BEAM`）；将该变量设为 `0` 可禁用上限。Beam 搜索主要影响 `temperature=0` 时的确定性解码。

**示例：**

```bash
curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@meeting.m4a \
    -F model=whisper-1 \
    -F language=zh
```

使用 API 密钥认证：

```bash
curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer your_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

### 响应格式

| `response_format` | 说明 |
|---|---|
| `json` | `{"text": "..."}` — 默认，与 OpenAI 基础响应格式一致 |
| `text` | 纯文本，无 JSON 封装 |
| `verbose_json` | 完整 JSON，包含语言、时长、逐段时间戳及对数概率 |
| `srt` | SubRip 字幕格式（`.srt`） |
| `vtt` | WebVTT 字幕格式（`.vtt`） |

**示例 — 流式接收解码段落：**

```bash
curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@long-audio.mp3 \
    -F model=whisper-1 \
    -F stream=true
```

**SSE 响应**（使用 [OpenAI 流式转录协议](https://developers.openai.com/api/docs/guides/speech-to-text#streaming)）：

```
data: {"type":"transcript.text.delta","delta":"您好，最近怎么样？"}

data: {"type":"transcript.text.delta","delta":" 我很好，谢谢。"}

data: {"type":"transcript.text.done","text":"您好，最近怎么样？ 我很好，谢谢。"}

data: [DONE]
```

上传后第一个增量文本通常在 1–3 秒内到达。每个 `transcript.text.delta` 事件包含刚解码的段落的增量文本。最后的 `transcript.text.done` 事件包含与标准 `json` 响应等效的完整转录文本。

<details>
<summary><strong>示例 — 通过浏览器 <code>fetch</code> 进行流式传输</strong></summary>

```javascript
const form = new FormData();
form.append("file", audioBlob, "audio.webm");
form.append("model", "whisper-1");
form.append("stream", "true");

const res = await fetch("http://您的服务器IP:9000/v1/audio/transcriptions", {
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

**示例 — 获取 SRT 字幕：**

```bash
curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@video.mp4 \
    -F model=whisper-1 \
    -F response_format=srt
```

**示例 — 带时间戳的详细 JSON：**

```bash
curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json
```

**示例 — 带词级时间戳的详细 JSON：**

```bash
curl http://您的服务器IP:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json \
    -F "timestamp_granularities[]=word"
```

当 `timestamp_granularities[]` 包含 `word` 时，`verbose_json` 响应包含顶层 `words` 数组：

```json
{
  "word": "hello",
  "start": 0.5,
  "end": 0.8,
  "probability": 0.98
}
```

### 翻译音频

```
POST /v1/audio/translations
Content-Type: multipart/form-data
```

将任意语言的音频翻译为英文文本。与 [OpenAI 音频翻译接口](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create)兼容。接受常见的翻译参数。输出始终为英文。

> **注意：** 仅英语（`.en`）模型不支持翻译。请使用多语言模型（如 `base`、`small`、`large-v3-turbo`）。

**示例：**

```bash
curl http://您的服务器IP:9000/v1/audio/translations \
    -H "Authorization: Bearer $scribecrate_api_key" \
    -F file=@french_audio.mp3 \
    -F model=whisper-1
```

### 列出模型

```
GET /v1/models
```

返回 OpenAI 兼容格式的当前活跃模型。

```bash
curl http://您的服务器IP:9000/v1/models -H "Authorization: Bearer $scribecrate_api_key"
```

### 交互式 API 文档

可在以下地址访问交互式 Swagger UI：

```
http://您的服务器IP:9000/docs
```

## 持久化数据

所有服务器数据存储在 Docker 数据卷（容器内的 `/var/lib/whisper`）中：

```
/var/lib/whisper/
├── models--Systran--faster-whisper-*/   # 缓存的 Whisper 模型文件（从 HuggingFace 下载）
├── .port                 # 当前端口（供 whisper_manage 使用）
├── .model                # 当前模型名称（供 whisper_manage 使用）
└── .server_addr          # 缓存的服务器 IP（供 whisper_manage 使用）
```

请备份 Docker 数据卷以保留已下载的模型。模型文件较大（145 MB – 3 GB），首次启动时下载可能需要数分钟；保留数据卷可避免在重建容器时重新下载。

**提示：** `/var/lib/whisper` 数据卷与 `docker-whisper-live` 的 `/var/lib/whisper-live` 数据卷使用相同的 HuggingFace 缓存布局。如果已通过 `docker-whisper-live` 下载了模型，可绑定挂载相同的数据卷目录以避免重复下载。

## 管理服务器

在运行中的容器内使用 `whisper_manage` 来查看和管理服务器。

**显示服务器信息：**

```bash
docker exec whisper whisper_manage --showinfo
```

**列出可用模型：**

```bash
docker exec whisper whisper_manage --listmodels
```

**预先下载模型：**

```bash
docker exec whisper whisper_manage --downloadmodel large-v3-turbo
```

## 切换模型

要更换活跃模型：

1. *（可选但建议）* 在服务器运行时预先下载新模型：
   ```bash
   docker exec whisper whisper_manage --downloadmodel large-v3-turbo
   ```

2. 在 `whisper.env` 文件中更新 `WHISPER_MODEL`（或在 `docker run` 命令中添加 `-e WHISPER_MODEL=large-v3-turbo`）。

3. 重启容器：
   ```bash
   docker restart whisper
   ```

**可用模型：**

| 模型 | 磁盘占用 | 内存（约） | 说明 |
|---|---|---|---|
| `tiny` | ~75 MB | ~250 MB | 最快；精度较低 |
| `tiny.en` | ~75 MB | ~250 MB | 仅英语 |
| `base` | ~145 MB | ~700 MB | 良好平衡 — **默认** |
| `base.en` | ~145 MB | ~700 MB | 仅英语 |
| `small` | ~465 MB | ~1.5 GB | 更高精度 |
| `small.en` | ~465 MB | ~1.5 GB | 仅英语 |
| `medium` | ~1.5 GB | ~5 GB | 高精度 |
| `medium.en` | ~1.5 GB | ~5 GB | 仅英语 |
| `large-v1` | ~3 GB | ~10 GB | 旧版大型模型 |
| `large-v2` | ~3 GB | ~10 GB | 非常高精度 |
| `large-v3` | ~3 GB | ~10 GB | 最高精度 |
| `large-v3-turbo` | ~1.6 GB | ~6 GB | 高速 + 高精度 ⭐ |
| `turbo` | ~1.6 GB | ~6 GB | `large-v3-turbo` 的别名 |

> **提示：** `large-v3-turbo` 的精度接近 `large-v3`，但资源消耗约为其一半。对于大多数生产部署，这是从 `base` 升级的推荐选择。

内存数据为近似值，基于 INT8 量化（默认）。模型缓存在 `/var/lib/whisper` Docker 数据卷中，仅需下载一次。

## 保护你的服务器

如果你的 ScribeCrate 服务器可从公网访问 —— 即使只是短暂可达 —— 也请至少采取以下保护措施。转录对 CPU/GPU 资源消耗较大，未做身份验证的接口可能被滥用，浪费你的计算资源。

**1. 使用 API 密钥。** 挂载 `/var/lib/whisper` 数据卷的新安装会自动生成 API 密钥。可用 `docker exec whisper whisper_manage --showkey` 查看；脚本中可用 `docker exec whisper whisper_manage --getkey`。没有密钥的既有安装会保持开放以兼容旧行为；也可以在 `env` 文件中设置 `WHISPER_API_KEY` 手动启用认证。所有已认证请求必须包含 `Authorization: Bearer <key>`。

```bash
# 生成 32 字节的随机密钥
openssl rand -hex 32
```

**2. 在反向代理后面时绑定到 localhost。** 将 `-p 9000:9000` 替换为 `-p 127.0.0.1:9000:9000`（或在 `docker-compose.yml` 中将 `"9000:9000/tcp"` 改为 `"127.0.0.1:9000:9000/tcp"`），使未加密端口无法从主机外部直接访问。

**3. 限制上传大小。** 服务器会拒绝超过 `WHISPER_MAX_UPLOAD_MB`（默认 `1024`）的上传。对于面向互联网的部署，还应配置反向代理在请求到达应用前拒绝超大上传（例如 nginx `client_max_body_size 100M;`）。

**4. 注意日志级别。** `WHISPER_LOG_LEVEL=DEBUG` 可能会将转录文本写入日志。在共享系统上请保持 `INFO` 或更高级别。

**5. 浏览器调用时在代理处启用 CORS。** 本服务器默认不设置 `Access-Control-Allow-Origin` 响应头；若需在不同源的网页中直接调用本 API，请在反向代理处添加 CORS 头。

**6. 考虑限流。** 在服务器前部署限流（如 nginx `limit_req_zone`、Caddy `rate_limit`），限制每个客户端 IP 的并发转录请求数。

## 使用反向代理

如需面向公网部署，可在 ScribeCrate 前置反向代理处理 HTTPS 终止。在本地或可信网络中使用无需 HTTPS，但将 API 端点暴露在公网时建议启用 HTTPS。

从反向代理访问 ScribeCrate 容器时使用以下地址之一：

- **`whisper:9000`** — 如果反向代理作为容器运行在与 ScribeCrate **同一 Docker 网络**中（例如定义在同一 `docker-compose.yml` 中）。
- **`127.0.0.1:9000`** — 如果反向代理运行在**主机上**且端口 `9000` 已发布（默认 `docker-compose.yml` 会发布该端口）。

**使用 [Caddy](https://caddyserver.com/docs/)（[Docker 镜像](https://hub.docker.com/_/caddy)）的示例**（自动 Let's Encrypt TLS，反向代理在同一 Docker 网络中）：

`Caddyfile`：
```
whisper.example.com {
  reverse_proxy whisper:9000
}
```

**使用 nginx 的示例**（反向代理运行在主机上）：

```nginx
server {
    listen 443 ssl;
    server_name whisper.example.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # 音频文件可能较大——按需调整上传限制
    client_max_body_size 100M;

    location / {
        proxy_pass         http://127.0.0.1:9000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;       # 流式传输（SSE）所需
        proxy_read_timeout 300s;
    }
}
```

## 更新 Docker 镜像

如需更新 Docker 镜像和容器，首先[下载](#下载)最新版本：

```bash
docker pull hwdsl2/whisper-server
```

如果镜像已是最新版本，您将看到：

```
Status: Image is up to date for hwdsl2/whisper-server:latest
```

否则将下载最新版本。删除并重新创建容器：

```bash
docker rm -f whisper
# 然后使用相同的数据卷和端口重新运行快速开始中的 docker run 命令。
```

您下载的模型将保留在 `whisper-data` 数据卷中。

## 与其他 AI 服务配合使用

ScribeCrate 可作为更广泛的自托管 AI 设置中的语音转文字服务。

如需完整和轻量级 Docker Compose 技术栈、手动 `docker run` 示例，以及结合 SpeakCrate、Embeddings、LiteLLM、Ollama、ParseCrate 和 MCP Gateway 的语音/RAG/MCP 流水线示例，请参阅 [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack/blob/main/README-zh.md)。

## 说话人分离

说话人分离功能识别每个转录片段中*谁在说话*。这是一个本地扩展，基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 使用导出为 ONNX 格式的 pyannote segmentation-3.0 模型。

**启用说话人分离：**

```bash
# 在 whisper.env 中：
WHISPER_DIARIZATION=true
```

ONNX 模型（共约 45 MB）在首次使用时自动下载并缓存到 `/var/lib/whisper` 数据卷。预先下载：

```bash
docker exec whisper whisper_manage --downloaddiarize
```

**启用说话人分离后的输出：**

`verbose_json` 在每个片段中添加 `speaker` 字段：

```json
{
  "segments": [
    {"id": 0, "start": 1.0, "end": 3.5, "text": "我们下周发布吧。", "speaker": "SPEAKER_00"},
    {"id": 1, "start": 4.0, "end": 6.2, "text": "我觉得 QA 还需要两天。", "speaker": "SPEAKER_01"}
  ]
}
```

`srt` 和 `vtt` 在文本前添加说话人标签：

```
1
00:00:01,000 --> 00:00:03,500
[SPEAKER_00] 我们下周发布吧。

2
00:00:04,000 --> 00:00:06,200
[SPEAKER_01] 我觉得 QA 还需要两天。
```

`text` 格式在说话人变化时显示标签：

```
[SPEAKER_00] 我们下周发布吧。
[SPEAKER_01] 我觉得 QA 还需要两天。
```

**注意事项：**

> [!IMPORTANT]
> 说话人分离需要完整音频分析，**不支持流式模式**（`stream=true`）。两者同时启用时，说话人分离会被静默跳过。

- 如果已知确切说话人数量，设置 `WHISPER_DIARIZE_NUM_SPEAKERS` 可提高准确性。
- 说话人分离在转录完成后运行，会增加与音频时长成正比的少量处理时间。

## 使用计数

此镜像使用公开的 GitHub Release 资源下载次数进行匿名聚合使用计数。计数是近似值，不代表唯一用户或活跃安装。镜像不会发送遥测负载，也不会使用私有收集器。仅当服务器成功启动且挂载了 `/var/lib/whisper` 卷后，才会以尽力而为方式计数；当该持久化安装首次运行不同镜像构建时，也会再次计数。要退出，请设置 `WHISPER_DISABLE_USAGE_COUNTS=1`。

## 技术细节

- 基础镜像：`:latest` 使用 `python:3.12-slim`；`:cuda` 使用 `nvidia/cuda`
- 运行时：Python 3（虚拟环境位于 `/opt/venv`）
- STT 引擎：[faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CTranslate2（CPU 默认 INT8，CUDA 默认 FP16）
- API 框架：[FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- 音频解码：[PyAV](https://github.com/PyAV-Org/PyAV)（内置 FFmpeg 库）
- 数据目录：`/var/lib/whisper`（Docker 数据卷）
- 模型存储：HuggingFace Hub 格式，存储在数据卷中——下载一次，重启后复用

## 授权协议

**注：** 预构建镜像中包含的软件组件（如 faster-whisper 及其依赖项）均受各自版权持有者所选许可证约束。使用预构建镜像时，用户有责任确保其使用方式符合镜像内所有软件的相关许可证要求。

版权所有 (C) 2026 Lin Song   
本作品采用 [MIT 许可证](https://opensource.org/licenses/MIT)授权。

**faster-whisper** 版权归 SYSTRAN 所有，依据 [MIT 许可证](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)分发。

ScribeCrate 是使用 Whisper 模型的独立服务器，与 OpenAI 或 SYSTRAN 无关联，未获其背书或赞助。

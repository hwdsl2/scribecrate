[English](README.md) | [简体中文](README-zh.md) | [繁體中文](README-zh-Hant.md) | [Русский](README-ru.md)

# Whisper Speech-to-Text on Docker

[![Build Status](https://github.com/hwdsl2/docker-whisper/actions/workflows/main.yml/badge.svg)](https://github.com/hwdsl2/docker-whisper/actions/workflows/main.yml) &nbsp;[![Docker Pulls](https://raw.githubusercontent.com/hwdsl2/badges/main/img/docker-pulls-whisper-server.svg)](https://hub.docker.com/r/hwdsl2/whisper-server) &nbsp;[![License: MIT](docs/images/license.svg)](https://opensource.org/licenses/MIT) &nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://vpnsetup.net/whisper-notebook)

Part of the [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack) — deploy a complete self-hosted AI stack with a single command.

Docker image to run a [Whisper](https://github.com/openai/whisper) speech-to-text server, powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Provides OpenAI-compatible audio transcription and translation APIs. Based on Debian (python:3.12-slim). Designed to be simple, private, and self-hosted.

**Features:**

- OpenAI-compatible `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` endpoints — any app using the OpenAI Whisper API switches with a one-line change
- Supports all Whisper models: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` and more
- Speaker diarization — identify who is speaking in each segment (optional local extension via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx))
- Model management via a helper script (`whisper_manage`)
- Audio stays on your server — no data sent to third parties
- All major audio formats supported (mp3, m4a, wav, webm, ogg, flac, and all ffmpeg formats)
- Multiple response formats: JSON, plain text, verbose JSON, SRT subtitles, WebVTT subtitles
- Streaming transcription — add `stream=true` to receive segments via SSE as they are decoded, with no waiting for the full file
- NVIDIA GPU (CUDA) acceleration for faster inference (`:cuda` image tag)
- Offline/air-gapped mode — run without internet access using pre-cached models (`WHISPER_LOCAL_ONLY`)
- Automatically built and published via [GitHub Actions](https://github.com/hwdsl2/docker-whisper/actions)
- Persistent model cache via a Docker volume
- Multi-arch: `linux/amd64`, `linux/arm64`

> 📘 [The Self-Hosted AI Builder’s Guide](https://books2read.com/aiguide?store=amazon) is a practical guide to building, securing, and operating your own private AI stack.

**Also available:**

- Try it online: [Open in Colab](https://vpnsetup.net/whisper-notebook) — no Docker or installation required
- Related AI services: [WhisperLive](https://github.com/hwdsl2/docker-whisper-live), [Kokoro](https://github.com/hwdsl2/docker-kokoro), [Embeddings](https://github.com/hwdsl2/docker-embeddings), [LiteLLM](https://github.com/hwdsl2/docker-litellm), [Ollama](https://github.com/hwdsl2/docker-ollama), [Docling](https://github.com/hwdsl2/docker-docling), [MCP Gateway](https://github.com/hwdsl2/docker-mcp-gateway)

## When to use Whisper vs. WhisperLive

| | **docker-whisper** | [docker-whisper-live](https://github.com/hwdsl2/docker-whisper-live) |
|---|---|---|
| **Use case** | Transcribe complete audio files | Live microphone / real-time audio streaming |
| **Protocol** | HTTP REST | WebSocket (streaming) + HTTP REST |
| **Latency** | Full file, then response | Near-real-time, word by word |
| **Best for** | Meeting recordings, uploaded audio | Browser capture, RTSP streams, live captions |
| **Image size** | ~190 MB (~3.1 GB for `:cuda`) | ~750 MB (~4.5 GB for `:cuda`) |

## Quick start

Use this command to set up a Whisper server:

```bash
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

<details>
<summary><strong>GPU quick start (NVIDIA CUDA)</strong></summary>

If you have an NVIDIA GPU, use the `:cuda` image for hardware-accelerated inference:

```bash
docker run \
    --name whisper \
    --restart=always \
    --gpus=all \
    -v whisper-data:/var/lib/whisper \
    -p 9000:9000 \
    -d hwdsl2/whisper-server:cuda
```

**Requirements:** NVIDIA GPU, [NVIDIA driver](https://www.nvidia.com/en-us/drivers/) 575.57.08+ (Linux) or 576.57+ (Windows), and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host. The `:cuda` image is `linux/amd64` only.

</details>

**Important:** This image requires at least 700 MB of available RAM for the default `base` model. Systems with 512 MB or less of RAM are not supported.

> [!NOTE]
> For internet-facing deployments, use a [reverse proxy](#using-a-reverse-proxy) to add HTTPS. Also replace `-p 9000:9000` with `-p 127.0.0.1:9000:9000` in the `docker run` command above, to prevent direct access to the unencrypted port.

The Whisper `base` model (~145 MB) is downloaded and cached on first start. Check the logs to confirm the server is ready:

```bash
docker logs whisper
```

Once you see "Whisper speech-to-text server is ready", transcribe your first audio file:

```bash
curl http://your_server_ip:9000/v1/audio/transcriptions \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

**Response:**
```json
{"text": "Your transcribed text appears here."}
```

**Tip:** Need a sample audio file to test? Download this English speech sample (WAV, MIT License) from the [Azure Samples](https://github.com/Azure-Samples/cognitive-services-speech-sdk) repository:

```bash
curl -L -o sample_speech.wav \
    "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/sampledata/audiofiles/katiesteve.wav"

curl http://your_server_ip:9000/v1/audio/transcriptions \
    -F file=@sample_speech.wav \
    -F model=whisper-1
```

Alternatively, you may [set up Whisper without Docker](https://github.com/hwdsl2/whisper-install). To learn more about how to use this image, read the sections below.

## Community

- 📬 [Get project updates and free deployment guides](https://selfhostedstack.beehiiv.com/subscribe?utm_campaign=ai) (1–2 emails/month)
- 💬 Join the [r/selfhostedstack](https://www.reddit.com/r/selfhostedstack/) community for discussions and showcases
- ⭐ Star the repository if you find it useful — it helps others discover it

<details>
<summary>Self-hosted VPN & networking projects</summary>

- [Setup IPsec VPN](https://github.com/hwdsl2/setup-ipsec-vpn)
- [IPsec VPN on Docker](https://github.com/hwdsl2/docker-ipsec-vpn-server)
- [WireGuard](https://github.com/hwdsl2/docker-wireguard)
- [OpenVPN](https://github.com/hwdsl2/docker-openvpn)
- [Headscale](https://github.com/hwdsl2/docker-headscale)

</details>

## Requirements

- A Linux server (local or cloud) with Docker installed
- Supported architectures: `amd64` (x86_64), `arm64` (e.g. Raspberry Pi 4/5, AWS Graviton)
- Minimum RAM: ~700 MB free for the default `base` model (see [model table](#switching-models))
- Internet access for the initial model download (the model is cached locally afterwards). Not required if using `WHISPER_LOCAL_ONLY=true` with pre-cached models.

**For GPU acceleration (`:cuda` image):**

- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- [NVIDIA driver](https://www.nvidia.com/en-us/drivers/) 575.57.08+ (Linux) or 576.57+ (Windows) installed on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- The `:cuda` image supports `linux/amd64` only

For internet-facing deployments, see [Using a reverse proxy](#using-a-reverse-proxy) to add HTTPS.

## Download

Get the trusted build from the [Docker Hub registry](https://hub.docker.com/r/hwdsl2/whisper-server/):

```bash
docker pull hwdsl2/whisper-server
```

For NVIDIA GPU acceleration, pull the `:cuda` tag instead:

```bash
docker pull hwdsl2/whisper-server:cuda
```

Alternatively, you may download from [Quay.io](https://quay.io/repository/hwdsl2/whisper-server):

```bash
docker pull quay.io/hwdsl2/whisper-server
docker image tag quay.io/hwdsl2/whisper-server hwdsl2/whisper-server
```

Supported platforms: `linux/amd64` and `linux/arm64`. The `:cuda` tag supports `linux/amd64` only.

## Environment variables

All variables are optional. Fresh installs with a mounted `/var/lib/whisper` volume auto-generate a Bearer token. Existing installs without a key remain open for backward compatibility.

This Docker image uses the following variables, that can be declared in an `env` file (see [example](whisper.env.example)):

| Variable | Description | Default |
|---|---|---|
| `WHISPER_MODEL` | Whisper model to use. See [model table](#switching-models) for options. | `base` |
| `WHISPER_LANGUAGE` | Default transcription language. BCP-47 code (e.g. `en`, `fr`, `de`, `zh`, `ja`) or `auto` to autodetect. | `auto` |
| `WHISPER_PORT` | HTTP port for the API (1–65535). | `9000` |
| `WHISPER_DEVICE` | Compute device: `cpu`, `cuda`, or `auto`. Use `cuda` with the `:cuda` image for GPU acceleration. `auto` detects GPU and falls back to CPU. | `cpu` |
| `WHISPER_COMPUTE_TYPE` | Quantization / compute type. `int8` is recommended for CPU; `float16` is recommended for CUDA. | `int8` (CPU) / `float16` (CUDA) |
| `WHISPER_THREADS` | CPU threads for inference. Set to the number of physical cores for best latency. | `2` |
| `WHISPER_API_KEY` | Optional Bearer token. Fresh persistent installs auto-generate one. If set, all API requests must include `Authorization: Bearer <key>`. Set explicitly empty to disable authentication. | Auto-generated for fresh persistent installs |
| `WHISPER_LOG_LEVEL` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. | `INFO` |
| `WHISPER_BEAM` | Beam size for transcription and translation decoding. Higher values may improve accuracy at the cost of speed. Use `1` for fastest (greedy) decoding. | `5` |
| `WHISPER_MAX_REQUEST_BEAM` | Maximum beam size allowed for the per-request `beam` override. Set to `0` to disable this limit. | `10` |
| `WHISPER_MAX_UPLOAD_MB` | Maximum uploaded audio file size in MB. Requests above this limit return HTTP 413. Set to `0` to disable the limit. | `1024` |
| `WHISPER_LOCAL_ONLY` | When set to any non-empty value (e.g. `true`), disables all HuggingFace model downloads. For offline or air-gapped deployments with pre-cached models. | *(not set)* |
| `WHISPER_WORD_TIMESTAMPS` | When set to `true`, enables word-level timestamps globally for all requests. The `verbose_json` output will include a top-level `words` array with per-word timing and confidence. Can also be enabled per-request via `timestamp_granularities[]=word`. | *(not set)* |
| `WHISPER_DIARIZATION` | Set to `true` to enable speaker diarization. Identifies who is speaking in each segment. Uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with pyannote segmentation-3.0 ONNX models (~45 MB, auto-downloaded on first use). Not supported in streaming mode. | *(not set)* |
| `WHISPER_DIARIZE_NUM_SPEAKERS` | Exact number of speakers (if known). Improves clustering accuracy. Set to `-1` or leave unset for auto-detection. | `-1` |
| `WHISPER_DIARIZE_THRESHOLD` | Clustering threshold for auto-detection. Lower = more speakers detected, higher = fewer. Ignored when exact speaker count is set. | `0.5` |
| `WHISPER_DISABLE_USAGE_COUNTS` | Set to `1` to disable anonymous aggregate usage counts. | *(not set)* |

**Note:** In your `env` file, you may enclose values in single quotes, e.g. `VAR='value'`. Do not add spaces around `=`. If you change `WHISPER_PORT`, update the `-p` flag in the `docker run` command accordingly.

Example using an `env` file:

```bash
cp whisper.env.example whisper.env
# Edit whisper.env with your settings, then:
docker run \
    --name whisper \
    --restart=always \
    -v whisper-data:/var/lib/whisper \
    -v ./whisper.env:/whisper.env:ro \
    -p 9000:9000 \
    -d hwdsl2/whisper-server
```

The env file is bind-mounted into the container, so changes are picked up on every restart without recreating the container.

<details>
<summary>Alternatively, pass it with <code>--env-file</code></summary>

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

## Using docker-compose

```bash
cp whisper.env.example whisper.env
# Edit whisper.env as needed, then:
docker compose up -d
docker logs whisper
```

Example `docker-compose.yml` (already included):

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # For a host-based reverse proxy, change to "127.0.0.1:9000:9000/tcp"
    volumes:
      - whisper-data:/var/lib/whisper
      - ./whisper.env:/whisper.env:ro

volumes:
  whisper-data:
    name: whisper-data
```

> [!NOTE]
> For internet-facing deployments, use a [reverse proxy](#using-a-reverse-proxy) to add HTTPS. Also change `"9000:9000/tcp"` to `"127.0.0.1:9000:9000/tcp"` in `docker-compose.yml`, to prevent direct access to the unencrypted port.

<details>
<summary><strong>Using docker-compose with GPU (NVIDIA CUDA)</strong></summary>

A separate `docker-compose.cuda.yml` is provided for GPU deployments:

```bash
cp whisper.env.example whisper.env
# Edit whisper.env as needed, then:
docker compose -f docker-compose.cuda.yml up -d
docker logs whisper
```

Example `docker-compose.cuda.yml` (already included):

```yaml
services:
  whisper:
    image: hwdsl2/whisper-server:cuda
    container_name: whisper
    restart: always
    ports:
      - "9000:9000/tcp"  # For a host-based reverse proxy, change to "127.0.0.1:9000:9000/tcp"
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

## API reference

The API is compatible with OpenAI's [audio transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) and [audio translation](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create) endpoints. Any application already calling `https://api.openai.com/v1/audio/transcriptions` can switch to self-hosted by setting:

Speaker diarization, when enabled, is a local sherpa-onnx extension and is not equivalent to OpenAI diarization models. OpenAI-only transcription options such as `gpt-4o-transcribe-diarize`, `response_format=diarized_json`, `include=logprobs`, `chunking_strategy`, `known_speaker_names`, and `known_speaker_references` are not supported and return `400`.

```
OPENAI_BASE_URL=http://your_server_ip:9000
```

### Transcribe audio

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | file | ✅ | Audio file. Supported formats: `mp3`, `mp4`, `m4a`, `wav`, `webm`, `ogg`, `flac` and all other formats supported by ffmpeg. |
| `model` | string | ✅ | Pass `whisper-1` (value is accepted but the active model is always used). |
| `language` | string | — | BCP-47 language code. Overrides `WHISPER_LANGUAGE` for this request. |
| `prompt` | string | — | Optional text to guide the model's style or continue a previous segment. |
| `response_format` | string | — | Output format. Default: `json`. See [response formats](#response-formats). Ignored when `stream=true`. OpenAI-only `diarized_json` is not supported. |
| `temperature` | float | — | Sampling temperature (0–1). Default: `0`. |
| `stream` | boolean | — | Enable SSE streaming. When `true`, segments are returned as `text/event-stream` events as they are decoded. Default: `false`. |
| `timestamp_granularities[]` | array | — | Timestamp granularities to populate. Values: `word`, `segment`. When `word` is included, `verbose_json` output includes a top-level `words` array. Default: `["segment"]`. |

**Local faster-whisper extension:** You can set `beam` to override `WHISPER_BEAM` for a single transcription or translation request. This is not part of the OpenAI API schema, so do not send it to the hosted OpenAI API or strict OpenAI-compatible gateways. The default per-request cap is `10` (`WHISPER_MAX_REQUEST_BEAM`); set that variable to `0` to disable the cap. Beam search mainly affects deterministic decoding when `temperature=0`.

**Example:**

```bash
curl http://your_server_ip:9000/v1/audio/transcriptions \
    -F file=@meeting.m4a \
    -F model=whisper-1 \
    -F language=en
```

With API key authentication:

```bash
curl http://your_server_ip:9000/v1/audio/transcriptions \
    -H "Authorization: Bearer your_api_key" \
    -F file=@audio.mp3 \
    -F model=whisper-1
```

### Response formats

| `response_format` | Description |
|---|---|
| `json` | `{"text": "..."}` — default, matches OpenAI's basic response |
| `text` | Plain text, no JSON wrapper |
| `verbose_json` | Full JSON with language, duration, per-segment timestamps, log-probabilities |
| `srt` | SubRip subtitle format (`.srt`) |
| `vtt` | WebVTT subtitle format (`.vtt`) |

**Example — stream segments as they are decoded:**

```bash
curl http://your_server_ip:9000/v1/audio/transcriptions \
    -F file=@long-audio.mp3 \
    -F model=whisper-1 \
    -F stream=true
```

**SSE response** (uses the [OpenAI streaming transcription protocol](https://developers.openai.com/api/docs/guides/speech-to-text#streaming)):

```
data: {"type":"transcript.text.delta","delta":"Hello, how are you?"}

data: {"type":"transcript.text.delta","delta":" I'm doing well, thank you."}

data: {"type":"transcript.text.done","text":"Hello, how are you? I'm doing well, thank you."}

data: [DONE]
```

The first delta typically arrives within 1–3 seconds of upload. Each `transcript.text.delta` event contains the incremental text for the segment just decoded. The final `transcript.text.done` event contains the full assembled transcript, equivalent to the standard `json` response.

<details>
<summary><strong>Example — stream from a browser using <code>fetch</code></strong></summary>

```javascript
const form = new FormData();
form.append("file", audioBlob, "audio.webm");
form.append("model", "whisper-1");
form.append("stream", "true");

const res = await fetch("http://your_server_ip:9000/v1/audio/transcriptions", {
  method: "POST", body: form,
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

**Example — get SRT subtitles:**

```bash
curl http://your_server_ip:9000/v1/audio/transcriptions \
    -F file=@video.mp4 \
    -F model=whisper-1 \
    -F response_format=srt
```

**Example — verbose JSON with timestamps:**

```bash
curl http://your_server_ip:9000/v1/audio/transcriptions \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json
```

**Example — verbose JSON with word-level timestamps:**

```bash
curl http://your_server_ip:9000/v1/audio/transcriptions \
    -F file=@audio.mp3 \
    -F model=whisper-1 \
    -F response_format=verbose_json \
    -F "timestamp_granularities[]=word"
```

When `timestamp_granularities[]` includes `word`, the `verbose_json` response includes a top-level `words` array:

```json
{
  "word": "hello",
  "start": 0.5,
  "end": 0.8,
  "probability": 0.98
}
```

### Translate audio

```
POST /v1/audio/translations
Content-Type: multipart/form-data
```

Translates audio in any language to English text. Compatible with [OpenAI's audio translation endpoint](https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create). Accepts the common translation parameters. The output is always in English.

> **Note:** Translation is not supported with English-only (`.en`) models. Use a multilingual model (e.g. `base`, `small`, `large-v3-turbo`).

**Example:**

```bash
curl http://your_server_ip:9000/v1/audio/translations \
    -F file=@french_audio.mp3 \
    -F model=whisper-1
```

### List models

```
GET /v1/models
```

Returns the active model in OpenAI-compatible format.

```bash
curl http://your_server_ip:9000/v1/models
```

### Interactive API docs

An interactive Swagger UI is available at:

```
http://your_server_ip:9000/docs
```

## Persistent data

All server data is stored in the Docker volume (`/var/lib/whisper` inside the container):

```
/var/lib/whisper/
├── models--Systran--faster-whisper-*/   # Cached Whisper model files (downloaded from HuggingFace)
├── .port                 # Active port (used by whisper_manage)
├── .model                # Active model name (used by whisper_manage)
└── .server_addr          # Cached server IP (used by whisper_manage)
```

Back up the Docker volume to preserve downloaded models. Models are large (145 MB – 3 GB) and can take several minutes to download on first start; preserving the volume avoids re-downloading on container recreation.

**Tip:** The `/var/lib/whisper` volume uses the same HuggingFace cache layout as `docker-whisper-live`'s `/var/lib/whisper-live` volume. If you have already downloaded a model with `docker-whisper-live`, you can bind-mount the same volume directory to avoid re-downloading.

## Managing the server

Use `whisper_manage` inside the running container to inspect and manage the server.

**Show server info:**

```bash
docker exec whisper whisper_manage --showinfo
```

**List available models:**

```bash
docker exec whisper whisper_manage --listmodels
```

**Pre-download a model:**

```bash
docker exec whisper whisper_manage --downloadmodel large-v3-turbo
```

## Switching models

To change the active model:

1. *(Optional but recommended)* Pre-download the new model while the server is running:
   ```bash
   docker exec whisper whisper_manage --downloadmodel large-v3-turbo
   ```

2. Update `WHISPER_MODEL` in your `whisper.env` file (or add `-e WHISPER_MODEL=large-v3-turbo` to your `docker run` command).

3. Restart the container:
   ```bash
   docker restart whisper
   ```

**Available models:**

| Model | Disk | RAM (approx) | Notes |
|---|---|---|---|
| `tiny` | ~75 MB | ~250 MB | Fastest; lower accuracy |
| `tiny.en` | ~75 MB | ~250 MB | English-only |
| `base` | ~145 MB | ~700 MB | Good balance — **default** |
| `base.en` | ~145 MB | ~700 MB | English-only |
| `small` | ~465 MB | ~1.5 GB | Better accuracy |
| `small.en` | ~465 MB | ~1.5 GB | English-only |
| `medium` | ~1.5 GB | ~5 GB | High accuracy |
| `medium.en` | ~1.5 GB | ~5 GB | English-only |
| `large-v1` | ~3 GB | ~10 GB | Older large model |
| `large-v2` | ~3 GB | ~10 GB | Very high accuracy |
| `large-v3` | ~3 GB | ~10 GB | Best accuracy |
| `large-v3-turbo` | ~1.6 GB | ~6 GB | Fast + high accuracy ⭐ |
| `turbo` | ~1.6 GB | ~6 GB | Alias for `large-v3-turbo` |

> **Tip:** `large-v3-turbo` offers accuracy close to `large-v3` at roughly half the resource cost. It is the recommended upgrade path from `base` for most production deployments.

RAM figures are approximate and reflect INT8 quantization (default). Models are cached in the `/var/lib/whisper` Docker volume and only downloaded once.

## Securing your server

If your Whisper server is reachable from the public internet — even briefly — apply at minimum these protections. Whisper is CPU/GPU-intensive, so an unauthenticated endpoint can be abused to burn your compute resources.

**1. Use an API key.** Fresh installs with a mounted `/var/lib/whisper` volume auto-generate an API key. Display it with `docker exec whisper whisper_manage --showkey`, or use `docker exec whisper whisper_manage --getkey` in scripts. Existing installs without a key remain open for backward compatibility; set `WHISPER_API_KEY` in your `env` file to enable authentication manually. All authenticated requests must include `Authorization: Bearer <key>`.

```bash
# Generate a 32-byte random key
openssl rand -hex 32
```

**2. Bind to localhost when fronted by a reverse proxy.** Replace `-p 9000:9000` with `-p 127.0.0.1:9000:9000` (or change `"9000:9000/tcp"` to `"127.0.0.1:9000:9000/tcp"` in `docker-compose.yml`) so the unencrypted port is not reachable directly from outside the host.

**3. Limit upload size.** The server rejects uploads above `WHISPER_MAX_UPLOAD_MB` (default `1024`). For internet-facing deployments, also configure your reverse proxy to reject oversized uploads (e.g. nginx `client_max_body_size 100M;`) before they reach the app.

**4. Mind the log level.** `WHISPER_LOG_LEVEL=DEBUG` may write transcript text to logs. Keep it at `INFO` or higher on shared systems.

**5. Enable CORS at the proxy if calling from a browser.** The server does not set `Access-Control-Allow-Origin` headers by default; add them at your reverse proxy if you intend to call the API directly from a web page on a different origin.

**6. Consider rate limiting.** Place a rate-limit (e.g. nginx `limit_req_zone`, Caddy `rate_limit`) in front of the server to cap concurrent transcriptions per client IP.

## Using a reverse proxy

For internet-facing deployments, place a reverse proxy in front of Whisper to handle HTTPS termination. The server works without HTTPS on a local or trusted network, but HTTPS is recommended when the API endpoint is exposed to the internet.

Use one of the following addresses to reach the Whisper container from your reverse proxy:

- **`whisper:9000`** — if your reverse proxy runs as a container in the **same Docker network** as Whisper (e.g. defined in the same `docker-compose.yml`).
- **`127.0.0.1:9000`** — if your reverse proxy runs **on the host** and port `9000` is published (the default `docker-compose.yml` publishes it).

**Example with [Caddy](https://caddyserver.com/docs/) ([Docker image](https://hub.docker.com/_/caddy))** (automatic TLS via Let's Encrypt, reverse proxy in the same Docker network):

`Caddyfile`:
```
whisper.example.com {
  reverse_proxy whisper:9000
}
```

**Example with nginx** (reverse proxy on the host):

```nginx
server {
    listen 443 ssl;
    server_name whisper.example.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Audio files can be large — increase the upload limit as needed
    client_max_body_size 100M;

    location / {
        proxy_pass         http://127.0.0.1:9000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;       # required for chunked streaming (SSE)
        proxy_read_timeout 300s;
    }
}
```

## Update Docker image

To update the Docker image and container, first [download](#download) the latest version:

```bash
docker pull hwdsl2/whisper-server
```

If the Docker image is already up to date, you should see:

```
Status: Image is up to date for hwdsl2/whisper-server:latest
```

Otherwise, it will download the latest version. Remove and re-create the container:

```bash
docker rm -f whisper
# Then re-run the docker run command from Quick start with the same volume and port.
```

Your downloaded models are preserved in the `whisper-data` volume.

## Using with other AI services

Whisper can be used as the speech-to-text service in a broader self-hosted AI setup.

For full and lightweight Docker Compose stacks, manual `docker run` examples, and voice/RAG/MCP pipeline examples with Kokoro, Embeddings, LiteLLM, Ollama, Docling, and MCP Gateway, see [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack).

## Speaker diarization

Speaker diarization identifies *who* is speaking in each transcribed segment. It is a local extension powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) using the pyannote segmentation-3.0 model exported to ONNX format.

**Enable diarization:**

```bash
# In your whisper.env:
WHISPER_DIARIZATION=true
```

ONNX models (~45 MB total) are automatically downloaded on first use and cached in the `/var/lib/whisper` volume. To pre-download them:

```bash
docker exec whisper whisper_manage --downloaddiarize
```

**Output with diarization enabled:**

`verbose_json` adds a `speaker` field to each segment:

```json
{
  "segments": [
    {"id": 0, "start": 1.0, "end": 3.5, "text": "We should launch next week.", "speaker": "SPEAKER_00"},
    {"id": 1, "start": 4.0, "end": 6.2, "text": "I think QA needs two more days.", "speaker": "SPEAKER_01"}
  ]
}
```

`srt` and `vtt` prepend the speaker label:

```
1
00:00:01,000 --> 00:00:03,500
[SPEAKER_00] We should launch next week.

2
00:00:04,000 --> 00:00:06,200
[SPEAKER_01] I think QA needs two more days.
```

`text` format shows the speaker label on speaker changes:

```
[SPEAKER_00] We should launch next week.
[SPEAKER_01] I think QA needs two more days.
```

**Notes:**

> [!IMPORTANT]
> Diarization requires full audio analysis and is **not supported in streaming mode** (`stream=true`). If both are enabled, diarization is silently skipped.

- Set `WHISPER_DIARIZE_NUM_SPEAKERS` if you know the exact number of speakers for better accuracy.
- The diarization pipeline runs after transcription, adding a small amount of processing time proportional to audio duration.

## Usage counts

This image uses public GitHub release asset download counts for anonymous, aggregate usage counts. Counts are approximate and are not unique users or active installs. The image does not send a telemetry payload or use a private collector. It only attempts the best-effort count after the server starts successfully with a mounted `/var/lib/whisper` volume, and again when that persistent install first runs a different image build. To opt out, set `WHISPER_DISABLE_USAGE_COUNTS=1`.

## Technical details

- Base image: `python:3.12-slim` for `:latest`; `nvidia/cuda` for `:cuda`
- Runtime: Python 3 (virtual environment at `/opt/venv`)
- STT engine: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with CTranslate2 (INT8 by default on CPU, FP16 on CUDA)
- API framework: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- Audio decoding: [PyAV](https://github.com/PyAV-Org/PyAV) (bundled FFmpeg libraries)
- Data directory: `/var/lib/whisper` (Docker volume)
- Model storage: HuggingFace Hub format inside the volume — downloaded once, reused on restarts

## License

**Note:** The software components inside the pre-built image (such as faster-whisper and its dependencies) are under the respective licenses chosen by their respective copyright holders. As for any pre-built image usage, it is the image user's responsibility to ensure that any use of this image complies with any relevant licenses for all software contained within.

Copyright (C) 2026 Lin Song   
This work is licensed under the [MIT License](https://opensource.org/licenses/MIT).

**faster-whisper** is Copyright (C) SYSTRAN, and is distributed under the [MIT License](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE).

This project is an independent Docker setup for Whisper and is not affiliated with, endorsed by, or sponsored by OpenAI or SYSTRAN.

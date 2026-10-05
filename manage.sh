#!/bin/bash
#
# https://github.com/hwdsl2/scribecrate
#
# Copyright (C) 2026 Lin Song <linsongui@gmail.com>
#
# This work is licensed under the MIT License
# See: https://opensource.org/licenses/MIT

export PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

WHISPER_DATA="/var/lib/whisper"
PORT_FILE="${WHISPER_DATA}/.port"
MODEL_FILE="${WHISPER_DATA}/.model"
SERVER_ADDR_FILE="${WHISPER_DATA}/.server_addr"
API_KEY_FILE="${WHISPER_DATA}/.api_key"
AUTH_ENABLED_FILE="${WHISPER_DATA}/.auth_enabled"

exiterr() { echo "Error: $1" >&2; exit 1; }

show_usage() {
  local exit_code="${2:-1}"
  if [ -n "$1" ]; then
    echo "Error: $1" >&2
  fi
  cat 1>&2 <<'EOF'

ScribeCrate - Server Management
https://github.com/hwdsl2/scribecrate

Usage: docker exec <container> whisper_manage [options]

Options:
  --showinfo                           show server info (model, endpoint, API docs)
  --showkey                            show the API key, if configured
  --getkey                             output the API key (machine-readable, no decoration)
  --listmodels                         list available Whisper model names and sizes
  --downloadmodel <model>              pre-download a model to the cache volume
  --downloaddiarize                    pre-download diarization ONNX models

  -h, --help                           show this help message and exit

Available models: tiny, tiny.en, base, base.en, small, small.en,
                  medium, medium.en, large-v1, large-v2, large-v3,
                  large-v3-turbo (or: turbo)

To switch the active model, set WHISPER_MODEL=<name> and restart the container.
Use '--downloadmodel' to pre-download a model before switching, avoiding a
delay on the next container start.

Examples:
  docker exec whisper whisper_manage --showinfo
  docker exec whisper whisper_manage --showkey
  docker exec whisper whisper_manage --getkey
  docker exec whisper whisper_manage --listmodels
  docker exec whisper whisper_manage --downloadmodel large-v3
  docker exec whisper whisper_manage --downloadmodel large-v3-turbo
  docker exec whisper whisper_manage --downloaddiarize

EOF
  exit "$exit_code"
}

check_container() {
  if [ ! -f "/.dockerenv" ] && [ ! -f "/run/.containerenv" ] \
    && [ -z "$KUBERNETES_SERVICE_HOST" ] \
    && ! head -n 1 /proc/1/sched 2>/dev/null | grep -q '^run\.sh '; then
    exiterr "This script must be run inside a container (e.g. Docker, Podman)."
  fi
}

load_config() {
  if [ -z "$WHISPER_PORT" ]; then
    if [ -f "$PORT_FILE" ]; then
      WHISPER_PORT=$(cat "$PORT_FILE")
    else
      WHISPER_PORT=9000
    fi
  fi

  if [ -z "$WHISPER_MODEL" ]; then
    if [ -f "$MODEL_FILE" ]; then
      WHISPER_MODEL=$(cat "$MODEL_FILE")
    else
      WHISPER_MODEL=base
    fi
  fi

  if [ -f "$SERVER_ADDR_FILE" ]; then
    SERVER_ADDR=$(cat "$SERVER_ADDR_FILE")
  else
    SERVER_ADDR="<server ip>"
  fi

  if [ -f "$AUTH_ENABLED_FILE" ]; then
    WHISPER_AUTH_ENABLED=$(cat "$AUTH_ENABLED_FILE")
  fi

  if [ "$WHISPER_AUTH_ENABLED" != 0 ] && [ -z "$WHISPER_API_KEY" ] && [ -f "$API_KEY_FILE" ]; then
    WHISPER_API_KEY=$(cat "$API_KEY_FILE")
  fi

  if [ -z "$WHISPER_AUTH_ENABLED" ]; then
    if [ -n "$WHISPER_API_KEY" ]; then
      WHISPER_AUTH_ENABLED=1
    else
      WHISPER_AUTH_ENABLED=0
    fi
  fi
}

check_server() {
  if ! curl -sf "http://127.0.0.1:${WHISPER_PORT}/health" >/dev/null 2>&1; then
    exiterr "ScribeCrate server is not responding on port ${WHISPER_PORT}. Is the container fully started?"
  fi
}

parse_args() {
  show_info=0
  show_key=0
  get_key=0
  list_models=0
  download_model=0
  download_diarize=0
  model_to_download=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --showinfo)
        show_info=1
        shift
        ;;
      --showkey)
        show_key=1
        shift
        ;;
      --getkey)
        get_key=1
        shift
        ;;
      --listmodels)
        list_models=1
        shift
        ;;
      --downloadmodel)
        download_model=1
        model_to_download="${2:-}"
        shift
        [ "$#" -gt 0 ] && shift
        ;;
      --downloaddiarize)
        download_diarize=1
        shift
        ;;
      -h|--help)
        show_usage "" 0
        ;;
      *)
        show_usage "Unknown parameter: $1"
        ;;
    esac
  done
}

check_args() {
  local action_count
  action_count=$((show_info + show_key + get_key + list_models + download_model + download_diarize))

  if [ "$action_count" -eq 0 ]; then
    show_usage
  fi
  if [ "$action_count" -gt 1 ]; then
    show_usage "Specify only one action at a time."
  fi

  if [ "$download_model" = 1 ] && [ -z "$model_to_download" ]; then
    exiterr "Missing model name. Usage: --downloadmodel <model>"
  fi
}

do_show_key() {
  if [ "$WHISPER_AUTH_ENABLED" != 1 ]; then
    exiterr "API key authentication is disabled for this container."
  fi

  if [ -z "$WHISPER_API_KEY" ]; then
    if [ -f "$API_KEY_FILE" ]; then
      WHISPER_API_KEY=$(cat "$API_KEY_FILE")
    else
      exiterr "API key not found. Authentication may be disabled for this container."
    fi
  fi

  echo
  echo "==========================================================="
  echo " ScribeCrate API key"
  echo "==========================================================="
  echo "${WHISPER_API_KEY}"
  echo "==========================================================="
  echo
  echo "Use with: -H \"Authorization: Bearer ${WHISPER_API_KEY}\""
  echo
}

do_get_key() {
  if [ "$WHISPER_AUTH_ENABLED" != 1 ]; then
    exit 1
  fi

  if [ -z "$WHISPER_API_KEY" ]; then
    if [ -f "$API_KEY_FILE" ]; then
      WHISPER_API_KEY=$(cat "$API_KEY_FILE")
    else
      exit 1
    fi
  fi

  printf '%s' "$WHISPER_API_KEY"
}

do_show_info() {
  echo
  echo "==========================================================="
  echo " ScribeCrate Transcription Server"
  echo "==========================================================="
  echo " Active model: $WHISPER_MODEL"
  echo " Endpoint:     http://${SERVER_ADDR}:${WHISPER_PORT}"
  echo "==========================================================="
  echo
  echo "API endpoints:"
  echo "  POST http://${SERVER_ADDR}:${WHISPER_PORT}/v1/audio/transcriptions"
  echo "  POST http://${SERVER_ADDR}:${WHISPER_PORT}/v1/audio/translations"
  echo "  GET  http://${SERVER_ADDR}:${WHISPER_PORT}/v1/models"
  echo "  GET  http://${SERVER_ADDR}:${WHISPER_PORT}/docs     (interactive docs)"
  echo
  echo "Example transcription:"
  echo "  curl http://${SERVER_ADDR}:${WHISPER_PORT}/v1/audio/transcriptions \\"
  if [ "$WHISPER_AUTH_ENABLED" = 1 ]; then
    echo "    -H \"Authorization: Bearer <api-key>\" \\"
  fi
  echo "    -F file=@audio.mp3 -F model=whisper-1"
  if [ "$WHISPER_AUTH_ENABLED" = 1 ]; then
    echo
    echo "Use '--showkey' to display the API key."
  fi
  echo
  echo "To change the active model:"
  echo "  1. Pre-download: docker exec <container> whisper_manage --downloadmodel <name>"
  echo "  2. Set WHISPER_MODEL=<name> in your env file and restart the container."
  echo
}

do_list_models() {
  cat <<'EOF'

Available Whisper models:

  Name              Disk     RAM (approx)   Notes
  ----              ----     ------------   -----
  tiny              ~75 MB   ~250 MB        Fastest; lower accuracy
  tiny.en           ~75 MB   ~250 MB        English-only variant
  base              ~145 MB  ~500 MB        Good balance — default
  base.en           ~145 MB  ~500 MB        English-only variant
  small             ~465 MB  ~1.5 GB        Better accuracy
  small.en          ~465 MB  ~1.5 GB        English-only variant
  medium            ~1.5 GB  ~5 GB          High accuracy
  medium.en         ~1.5 GB  ~5 GB          English-only variant
  large-v1          ~3 GB    ~10 GB         Older large model
  large-v2          ~3 GB    ~10 GB         Very high accuracy
  large-v3          ~3 GB    ~10 GB         Best accuracy (recommended for quality)
  large-v3-turbo    ~1.6 GB  ~6 GB          Fast + high accuracy (best overall upgrade)
  turbo             ~1.6 GB  ~6 GB          Alias for large-v3-turbo

Notes:
  - English-only (.en) variants are slightly faster for English audio.
  - large-v3-turbo (or: turbo) is recommended over large-v3 for most use
    cases: comparable accuracy with significantly lower resource usage.
  - Most models are downloaded from HuggingFace (Systran/faster-whisper-*);
    large-v3-turbo/turbo use mobiuslabsgmbh/faster-whisper-large-v3-turbo.
    All are cached in the /var/lib/whisper Docker volume.
  - INT8 quantization (default) reduces RAM usage by approximately 50%.

Use '--downloadmodel <name>' to pre-download a model before switching.

EOF
}

do_download_model() {
  # Block download if WHISPER_LOCAL_ONLY is set
  if [ -n "$WHISPER_LOCAL_ONLY" ]; then
    exiterr "WHISPER_LOCAL_ONLY is set — model downloads are disabled. Unset it to allow downloads."
  fi

  # Validate model name
  case "$model_to_download" in
    tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|\
    large-v1|large-v2|large-v3|large-v3-turbo|turbo) ;;
    *)
      exiterr "Unknown model '$model_to_download'. Run '--listmodels' to see valid names."
      ;;
  esac

  echo
  echo "Downloading model '${model_to_download}' to /var/lib/whisper..."
  echo "This may take several minutes depending on model size and network speed."
  echo

  export HF_HOME=/var/lib/whisper

  _MODEL="$model_to_download" python3 - << 'PYEOF'
import os, sys

model_name = os.environ["_MODEL"]
cache_dir  = os.environ.get("HF_HOME", "/var/lib/whisper")

try:
    from faster_whisper import WhisperModel
    print(f"  Downloading '{model_name}' (compute_type=int8) ...")
    sys.stdout.flush()
    WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=cache_dir,
    )
    print(f"  Model '{model_name}' downloaded successfully.")
    print(f"  Cache location: {cache_dir}")
except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF

  echo
  echo "To activate this model, set WHISPER_MODEL=${model_to_download} in your"
  echo "env file (whisper.env) and restart the container."
  echo
}

do_download_diarize() {
  echo
  echo "Downloading diarization ONNX models to /var/lib/whisper..."
  echo "  - Segmentation model (~5 MB)"
  echo "  - Speaker embedding model (~40 MB)"
  echo

  local cache_dir="/var/lib/whisper"
  local seg_dir="${cache_dir}/sherpa-onnx-pyannote-segmentation-3-0"
  local seg_model="${seg_dir}/model.onnx"
  local emb_model="${cache_dir}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

  if [ ! -f "$seg_model" ]; then
    echo "  Downloading segmentation model..."
    curl -fSL --retry 3 --retry-delay 2 -o "${cache_dir}/seg.tar.bz2" \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    tar xjf "${cache_dir}/seg.tar.bz2" -C "$cache_dir"
    rm -f "${cache_dir}/seg.tar.bz2"
    echo "  Segmentation model ready."
  else
    echo "  Segmentation model already exists."
  fi

  if [ ! -f "$emb_model" ]; then
    echo "  Downloading embedding model..."
    curl -fSL --retry 3 --retry-delay 2 -o "$emb_model" \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    echo "  Embedding model ready."
  else
    echo "  Embedding model already exists."
  fi

  echo
  echo "Diarization models downloaded successfully."
  echo "To enable diarization, set WHISPER_DIARIZATION=true in your env file."
  echo
}

check_container
load_config
parse_args "$@"
check_args

if [ "$show_info" = 1 ]; then
  check_server
  do_show_info
  exit 0
fi

if [ "$show_key" = 1 ]; then
  do_show_key
  exit 0
fi

if [ "$get_key" = 1 ]; then
  do_get_key
  exit 0
fi

if [ "$list_models" = 1 ]; then
  do_list_models
  exit 0
fi

if [ "$download_model" = 1 ]; then
  do_download_model
  exit 0
fi

if [ "$download_diarize" = 1 ]; then
  do_download_diarize
  exit 0
fi

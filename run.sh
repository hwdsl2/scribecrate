#!/bin/bash
#
# Docker script to configure and start a Whisper speech-to-text server
#
# DO NOT RUN THIS SCRIPT ON YOUR PC OR MAC! THIS IS ONLY MEANT TO BE RUN
# IN A CONTAINER!
#
# This file is part of Whisper Docker image, available at:
# https://github.com/hwdsl2/docker-whisper
#
# Copyright (C) 2026 Lin Song <linsongui@gmail.com>
#
# This work is licensed under the MIT License
# See: https://opensource.org/licenses/MIT

export PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

exiterr()  { echo "Error: $1" >&2; exit 1; }
nospaces() { printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
noquotes() { printf '%s' "$1" | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"; }

check_port() {
  printf '%s' "$1" | tr -d '\n' | grep -Eq '^[0-9]+$' \
  && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]
}

check_ip() {
  IP_REGEX='^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$'
  printf '%s' "$1" | tr -d '\n' | grep -Eq "$IP_REGEX"
}

fetch_public_ip() {
  local ip_addr ip_url
  for ip_url in https://ipv4.icanhazip.com https://api.ipify.org; do
    ip_addr=$(curl -q -4fsS --max-time 10 "$ip_url" 2>/dev/null) || continue
    ip_addr=${ip_addr%$'\r'}
    [[ "$ip_addr" != *$'\n'* && "$ip_addr" != *$'\r'* ]] || continue
    if check_ip "$ip_addr"; then
      printf '%s' "$ip_addr"
      return 0
    fi
  done
  return 1
}

# Source bind-mounted env file if present (takes precedence over --env-file)
if [ -f /whisper.env ]; then
  # shellcheck disable=SC1091
  . /whisper.env
fi

if [ ! -f "/.dockerenv" ] && [ ! -f "/run/.containerenv" ] \
  && [ -z "$KUBERNETES_SERVICE_HOST" ] \
  && ! head -n 1 /proc/1/sched 2>/dev/null | grep -q '^run\.sh '; then
  exiterr "This script ONLY runs in a container (e.g. Docker, Podman)."
fi

WHISPER_API_KEY_WAS_SET=${WHISPER_API_KEY+x}

# Read and sanitize environment variables
WHISPER_MODEL=$(nospaces "$WHISPER_MODEL")
WHISPER_MODEL=$(noquotes "$WHISPER_MODEL")
WHISPER_LANGUAGE=$(nospaces "$WHISPER_LANGUAGE")
WHISPER_LANGUAGE=$(noquotes "$WHISPER_LANGUAGE")
WHISPER_PORT=$(nospaces "$WHISPER_PORT")
WHISPER_PORT=$(noquotes "$WHISPER_PORT")
WHISPER_DEVICE=$(nospaces "$WHISPER_DEVICE")
WHISPER_DEVICE=$(noquotes "$WHISPER_DEVICE")
WHISPER_COMPUTE_TYPE=$(nospaces "$WHISPER_COMPUTE_TYPE")
WHISPER_COMPUTE_TYPE=$(noquotes "$WHISPER_COMPUTE_TYPE")
WHISPER_THREADS=$(nospaces "$WHISPER_THREADS")
WHISPER_THREADS=$(noquotes "$WHISPER_THREADS")
WHISPER_API_KEY=$(nospaces "$WHISPER_API_KEY")
WHISPER_API_KEY=$(noquotes "$WHISPER_API_KEY")
WHISPER_LOG_LEVEL=$(nospaces "$WHISPER_LOG_LEVEL")
WHISPER_LOG_LEVEL=$(noquotes "$WHISPER_LOG_LEVEL")
WHISPER_BEAM=$(nospaces "$WHISPER_BEAM")
WHISPER_BEAM=$(noquotes "$WHISPER_BEAM")
WHISPER_MAX_REQUEST_BEAM=$(nospaces "$WHISPER_MAX_REQUEST_BEAM")
WHISPER_MAX_REQUEST_BEAM=$(noquotes "$WHISPER_MAX_REQUEST_BEAM")
WHISPER_MAX_UPLOAD_MB=$(nospaces "$WHISPER_MAX_UPLOAD_MB")
WHISPER_MAX_UPLOAD_MB=$(noquotes "$WHISPER_MAX_UPLOAD_MB")
WHISPER_LOCAL_ONLY=$(nospaces "$WHISPER_LOCAL_ONLY")
WHISPER_LOCAL_ONLY=$(noquotes "$WHISPER_LOCAL_ONLY")
WHISPER_WORD_TIMESTAMPS=$(nospaces "$WHISPER_WORD_TIMESTAMPS")
WHISPER_WORD_TIMESTAMPS=$(noquotes "$WHISPER_WORD_TIMESTAMPS")
WHISPER_DIARIZATION=$(nospaces "$WHISPER_DIARIZATION")
WHISPER_DIARIZATION=$(noquotes "$WHISPER_DIARIZATION")
WHISPER_DIARIZE_NUM_SPEAKERS=$(nospaces "$WHISPER_DIARIZE_NUM_SPEAKERS")
WHISPER_DIARIZE_NUM_SPEAKERS=$(noquotes "$WHISPER_DIARIZE_NUM_SPEAKERS")
WHISPER_DIARIZE_THRESHOLD=$(nospaces "$WHISPER_DIARIZE_THRESHOLD")
WHISPER_DIARIZE_THRESHOLD=$(noquotes "$WHISPER_DIARIZE_THRESHOLD")

# Save whether the user explicitly set WHISPER_COMPUTE_TYPE (used below
# to auto-select float16 for CUDA when no explicit type was provided).
_USER_COMPUTE_TYPE="$WHISPER_COMPUTE_TYPE"

# Apply defaults
[ -z "$WHISPER_MODEL" ]        && WHISPER_MODEL=base
[ -z "$WHISPER_LANGUAGE" ]     && WHISPER_LANGUAGE=auto
[ -z "$WHISPER_PORT" ]         && WHISPER_PORT=9000
[ -z "$WHISPER_DEVICE" ]       && WHISPER_DEVICE=cpu
[ -z "$WHISPER_COMPUTE_TYPE" ] && WHISPER_COMPUTE_TYPE=int8
[ -z "$WHISPER_THREADS" ]      && WHISPER_THREADS=2
[ -z "$WHISPER_LOG_LEVEL" ]    && WHISPER_LOG_LEVEL=INFO
[ -z "$WHISPER_BEAM" ]         && WHISPER_BEAM=5
[ -z "$WHISPER_MAX_REQUEST_BEAM" ] && WHISPER_MAX_REQUEST_BEAM=10
[ -z "$WHISPER_MAX_UPLOAD_MB" ] && WHISPER_MAX_UPLOAD_MB=1024
[ -z "$WHISPER_DIARIZE_NUM_SPEAKERS" ] && WHISPER_DIARIZE_NUM_SPEAKERS=-1
[ -z "$WHISPER_DIARIZE_THRESHOLD" ]    && WHISPER_DIARIZE_THRESHOLD=0.5

# Validate port
if ! check_port "$WHISPER_PORT"; then
  exiterr "WHISPER_PORT must be an integer between 1 and 65535."
fi

# Validate model name
case "$WHISPER_MODEL" in
  tiny|tiny.en|base|base.en|small|small.en|medium|medium.en|\
  large-v1|large-v2|large-v3|large-v3-turbo|turbo) ;;
  *) exiterr "WHISPER_MODEL '$WHISPER_MODEL' is not recognized. Valid options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, large-v3-turbo, turbo" ;;
esac

# Validate device
case "$WHISPER_DEVICE" in
  cpu|cuda) ;;
  auto)
    # Auto-detect: use CUDA if an NVIDIA GPU is available, otherwise fall back to CPU
    if [ -e /dev/nvidia0 ] || nvidia-smi >/dev/null 2>&1; then
      WHISPER_DEVICE=cuda
    else
      WHISPER_DEVICE=cpu
    fi
    ;;
  *) exiterr "WHISPER_DEVICE must be one of: cpu, cuda, auto." ;;
esac

# Adjust default compute type for CUDA (float16 is optimal on GPU)
if [ "$WHISPER_DEVICE" = "cuda" ] && [ -z "$_USER_COMPUTE_TYPE" ]; then
  WHISPER_COMPUTE_TYPE=float16
fi

# Validate compute type
case "$WHISPER_COMPUTE_TYPE" in
  int8|int8_float16|int8_float32|int8_bfloat16|int16|float16|float32|bfloat16) ;;
  *) exiterr "WHISPER_COMPUTE_TYPE '$WHISPER_COMPUTE_TYPE' is not valid. Use: int8, float16, float32, etc." ;;
esac

# Validate log level
case "$WHISPER_LOG_LEVEL" in
  DEBUG|INFO|WARNING|ERROR|CRITICAL) ;;
  *) exiterr "WHISPER_LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL." ;;
esac

# Validate thread count
if ! printf '%s' "$WHISPER_THREADS" | grep -Eq '^[1-9][0-9]*$'; then
  exiterr "WHISPER_THREADS must be a positive integer."
fi

# Validate beam size
if ! printf '%s' "$WHISPER_BEAM" | grep -Eq '^[1-9][0-9]*$'; then
  exiterr "WHISPER_BEAM must be a positive integer (e.g. 1, 5)."
fi

# Validate maximum per-request beam size (0 disables the limit)
if ! printf '%s' "$WHISPER_MAX_REQUEST_BEAM" | grep -Eq '^(0|[1-9][0-9]*)$'; then
  exiterr "WHISPER_MAX_REQUEST_BEAM must be 0 (unlimited) or a positive integer."
fi

# Validate maximum upload size in MB (0 disables the limit)
if ! printf '%s' "$WHISPER_MAX_UPLOAD_MB" | grep -Eq '^(0|[1-9][0-9]*)$'; then
  exiterr "WHISPER_MAX_UPLOAD_MB must be 0 (unlimited) or a positive integer."
fi

# Validate diarization speaker counts (-1 for auto-detect, or a positive integer)
if ! printf '%s' "$WHISPER_DIARIZE_NUM_SPEAKERS" | grep -Eq '^(-1|[1-9][0-9]*)$'; then
  exiterr "WHISPER_DIARIZE_NUM_SPEAKERS must be -1 (auto) or a positive integer."
fi

# Validate diarization clustering threshold (a non-negative number)
if ! printf '%s' "$WHISPER_DIARIZE_THRESHOLD" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
  exiterr "WHISPER_DIARIZE_THRESHOLD must be a non-negative number (e.g. 0.5)."
fi

mkdir -p /var/lib/whisper
# Create the dedicated temp directory for audio uploads.
# TMPDIR is set to this path in the Dockerfile so that Python's tempfile module
# will automatically use it, keeping transient audio data off the main filesystem.
mkdir -p /run/whisper-temp

DATA_DIR="/var/lib/whisper"
API_KEY_FILE="${DATA_DIR}/.api_key"
AUTH_ENABLED_FILE="${DATA_DIR}/.auth_enabled"
AUTO_API_KEY_MARKER="${DATA_DIR}/.auto_api_key_created"
USAGE_STATE_DIR="${DATA_DIR}/.whisper-usage"
USAGE_BASE_URL=${WHISPER_USAGE_BASE_URL:-https://github.com/hwdsl2/ai-stack-extras/releases/download/v1.0.0}
data_mounted=false
data_existing=false

if grep -q " ${DATA_DIR} " /proc/mounts 2>/dev/null; then
  data_mounted=true
fi
if $data_mounted && find "$DATA_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
  data_existing=true
fi

usage_arch() {
  local arch
  arch=$(uname -m 2>/dev/null || printf 'unknown')
  case "$arch" in
    x86_64|amd64) printf 'amd64' ;;
    aarch64|arm64) printf 'arm64' ;;
    *) printf 'other' ;;
  esac
}

write_usage_state() {
  local state_file version tmp_file
  state_file=$1
  version=$2
  mkdir -p "$USAGE_STATE_DIR"
  tmp_file=$(mktemp "$USAGE_STATE_DIR/.usage.XXXXXX")
  printf '%s\n' "$version" > "$tmp_file"
  chmod 0644 "$tmp_file" 2>/dev/null || true
  mv "$tmp_file" "$state_file"
}

fetch_usage_asset() {
  local asset base_url
  asset=$1
  base_url=${USAGE_BASE_URL%/}
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --max-time 5 -o /dev/null "$base_url/$asset" >/dev/null 2>&1
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 5 -O /dev/null "$base_url/$asset" >/dev/null 2>&1
  else
    return 1
  fi
}

read_state_value() {
  [ -r "$1" ] || return 0
  tr -d '[:space:]' < "$1"
}

report_usage_counts() {
  local current_version accel arch state_file last_version action

  [ "${WHISPER_DISABLE_USAGE_COUNTS:-0}" != "1" ] || return 0
  $data_mounted || return 0

  current_version="${IMAGE_FLAVOR:-unknown}-${IMAGE_VER:-unknown}"
  case "${IMAGE_FLAVOR:-}" in
    cuda) accel=cuda ;;
    *) accel=cpu ;;
  esac
  arch=$(usage_arch)

  state_file="$USAGE_STATE_DIR/whisper.version"
  last_version=$(read_state_value "$state_file")
  action=

  if [ -z "$last_version" ]; then
    if $data_existing; then
      action=upgrade
    else
      action=deploy
    fi
  elif [ "$last_version" != "$current_version" ]; then
    action=upgrade
  fi

  if [ -n "$action" ]; then
    if fetch_usage_asset "cu-v1-whisper-$action-$accel-$arch"; then
      write_usage_state "$state_file" "$current_version"
    fi
  fi
}

if [ -n "$WHISPER_API_KEY" ]; then
  printf '%s' "$WHISPER_API_KEY" > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
elif [ -z "$WHISPER_API_KEY_WAS_SET" ] && [ -f "$API_KEY_FILE" ]; then
  WHISPER_API_KEY=$(cat "$API_KEY_FILE")
elif [ -z "$WHISPER_API_KEY_WAS_SET" ] && $data_mounted && ! $data_existing; then
  WHISPER_API_KEY="whisper-$(head -c 32 /dev/urandom | od -A n -t x1 | tr -d ' \n' | head -c 48)"
  printf '%s' "$WHISPER_API_KEY" > "$API_KEY_FILE"
  chmod 600 "$API_KEY_FILE"
  printf '%s\n' "true" > "$AUTO_API_KEY_MARKER"
  chmod 600 "$AUTO_API_KEY_MARKER"
fi
if [ -n "$WHISPER_API_KEY" ]; then
  printf '%s' "1" > "$AUTH_ENABLED_FILE"
else
  printf '%s' "0" > "$AUTH_ENABLED_FILE"
fi

# Determine server address for display
public_ip=$(fetch_public_ip) || public_ip=""
if check_ip "$public_ip"; then
  server_addr="$public_ip"
else
  server_addr="<server ip>"
fi

# Export all config for the Python API server
export WHISPER_MODEL
export WHISPER_LANGUAGE
export WHISPER_PORT
export WHISPER_DEVICE
export WHISPER_COMPUTE_TYPE
export WHISPER_THREADS
export WHISPER_API_KEY
export WHISPER_LOG_LEVEL
export WHISPER_BEAM
export WHISPER_MAX_REQUEST_BEAM
export WHISPER_MAX_UPLOAD_MB
export WHISPER_LOCAL_ONLY
export WHISPER_WORD_TIMESTAMPS
export WHISPER_DIARIZATION
export WHISPER_DIARIZE_NUM_SPEAKERS
export WHISPER_DIARIZE_THRESHOLD
# Point faster-whisper / HuggingFace Hub at the persistent Docker volume
export HF_HOME=/var/lib/whisper

# Persist config values so whisper_manage can read them without the env file
printf '%s' "$WHISPER_PORT"  > /var/lib/whisper/.port
printf '%s' "$WHISPER_MODEL" > /var/lib/whisper/.model
printf '%s' "$server_addr"   > /var/lib/whisper/.server_addr

echo
echo "Whisper Docker - https://github.com/hwdsl2/docker-whisper"

if ! grep -q " /var/lib/whisper " /proc/mounts 2>/dev/null; then
  echo
  echo "Note: /var/lib/whisper is not mounted. Model files will be lost on"
  echo "      container removal. Mount a Docker volume at /var/lib/whisper"
  echo "      to persist the downloaded model across container restarts."
  if [ -z "$WHISPER_API_KEY" ] && [ -z "$WHISPER_API_KEY_WAS_SET" ]; then
    echo "      API key authentication was not auto-enabled because the"
    echo "      data directory is not persistent."
  fi
elif [ -z "$WHISPER_API_KEY" ] && [ -z "$WHISPER_API_KEY_WAS_SET" ] && $data_existing; then
  echo
  echo "Warning: Existing Whisper data was found but no API key is configured."
  echo "         Preserving no-auth behavior for backward compatibility."
  echo "         Set WHISPER_API_KEY to enable authentication."
fi

echo
echo "Starting Whisper speech-to-text server..."
echo "  Model:    $WHISPER_MODEL"
echo "  Device:   $WHISPER_DEVICE ($WHISPER_COMPUTE_TYPE)"
echo "  Language: $WHISPER_LANGUAGE"
echo "  Port:     $WHISPER_PORT"
echo "  Beam:     $WHISPER_BEAM"
echo "  Max request beam: $WHISPER_MAX_REQUEST_BEAM"
if [ -n "$WHISPER_LOCAL_ONLY" ]; then
  echo "  Mode:     local-only (no HuggingFace downloads)"
fi
if [ "$(echo "$WHISPER_DIARIZATION" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
  echo "  Diarize:  enabled"
fi

if [ -z "$WHISPER_LOCAL_ONLY" ]; then
  # Determine the expected HuggingFace cache directory for the active model.
  # large-v3-turbo and its alias 'turbo' are hosted under mobiuslabsgmbh on HF,
  # not under Systran, so their cache directory name differs from other models.
  _model_in_cache() {
    local m="$1"
    [ -d "/var/lib/whisper/models--Systran--faster-whisper-${m}" ] && return 0
    [ -d "/var/lib/whisper/models--openai--whisper-${m}" ] && return 0
    case "$m" in
      large-v3-turbo|turbo)
        [ -d "/var/lib/whisper/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" ] && return 0
        ;;
    esac
    return 1
  }
  if ! _model_in_cache "$WHISPER_MODEL"; then
    echo
    echo "Note: Model '$WHISPER_MODEL' not found in cache. It will be downloaded"
    echo "      from HuggingFace on first start. This may take several minutes."
  fi
fi
echo

# Graceful shutdown — registered before starting the server so any SIGTERM
# received during the model-download startup phase is handled cleanly.
cleanup() {
  echo
  echo "Stopping Whisper server..."
  kill "${WHISPER_PID:-}" 2>/dev/null
  wait "${WHISPER_PID:-}" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# Start the FastAPI server in the background
cd /opt/src && python3 /opt/src/api_server.py &
WHISPER_PID=$!

# Wait for the server to become ready.
# Allow up to 300 seconds — first-run model download can take several minutes
# on a slow connection even for the base model (~145 MB).
wait_for_server() {
  local i=0
  while [ "$i" -lt 300 ]; do
    if ! kill -0 "$WHISPER_PID" 2>/dev/null; then
      return 1
    fi
    if curl -sf "http://127.0.0.1:${WHISPER_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  return 1
}

if ! wait_for_server; then
  if ! kill -0 "$WHISPER_PID" 2>/dev/null; then
    echo "Error: Whisper server failed to start. Check the container logs for details." >&2
  else
    echo "Error: Whisper server did not become ready within 300 seconds." >&2
    kill "$WHISPER_PID" 2>/dev/null
  fi
  exit 1
fi

report_usage_counts

echo
echo "==========================================================="
echo " Whisper speech-to-text server is ready"
echo "==========================================================="
echo " Model:    $WHISPER_MODEL"
echo " Endpoint: http://${server_addr}:${WHISPER_PORT}"
echo "==========================================================="
echo
echo "Transcribe an audio file:"
echo "  curl http://${server_addr}:${WHISPER_PORT}/v1/audio/transcriptions \\"
echo "    -F file=@audio.mp3 -F model=whisper-1"
echo
if [ -n "$WHISPER_API_KEY" ]; then
  echo "API key authentication is enabled."
  echo "Include header:  -H \"Authorization: Bearer \$WHISPER_API_KEY\""
  echo
fi
echo "Interactive API docs: http://${server_addr}:${WHISPER_PORT}/docs"
echo
echo "To set up HTTPS, see: Using a reverse proxy"
echo "  https://github.com/hwdsl2/docker-whisper#using-a-reverse-proxy"
echo
echo "Setup complete."
echo

# Wait for the server process to exit
wait "$WHISPER_PID"

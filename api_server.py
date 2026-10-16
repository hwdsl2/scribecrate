#!/usr/bin/env python3
"""
ScribeCrate Transcription API Server
Provides OpenAI-compatible /v1/audio/transcriptions and
/v1/audio/translations endpoints powered by faster-whisper.

https://github.com/hwdsl2/scribecrate

Copyright (C) 2026 Lin Song <linsongui@gmail.com>

This work is licensed under the MIT License
See: https://opensource.org/licenses/MIT
"""

import asyncio
import gc
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_level_str = os.environ.get("WHISPER_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisper_server")

# ---------------------------------------------------------------------------
# Model — loaded once at startup via the FastAPI lifespan hook
# ---------------------------------------------------------------------------

_model = None       # WhisperModel instance
_model_name = None  # name as loaded (e.g. "base")
_model_config = None  # cached load config for lazy reloads
_last_model_used_at = time.monotonic()
_active_inferences = 0
_idle_unload_seconds = 0
_idle_monitor_task = None
_beam_size = 5      # beam size used for transcription
_max_request_beam = 10  # maximum per-request beam override; 0 disables the cap
_word_timestamps = False  # default for word-level timestamps
_diarization_enabled = False  # set via WHISPER_DIARIZATION=true
_max_upload_bytes = 1024 * 1024 * 1024  # 1 GiB by default; 0 disables the limit
_UPLOAD_CHUNK_SIZE = 1024 * 1024
_UNSUPPORTED_DIARIZE_MODEL = "gpt-4o-transcribe-diarize"

# Serialise all inference calls (batch and streaming) so that CTranslate2 is
# never called concurrently from multiple threads.
_inference_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` when unset or empty.

    ``os.environ.get(name, default)`` only applies the default when the variable
    is absent. Because run.sh exports config variables unconditionally, an unset
    option arrives as an empty string, which would otherwise raise ValueError in
    int(). Treat empty/whitespace values as "use the default".
    """
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        logger.error(
            "Invalid value for %s: %r (expected an integer); using default %s",
            name, val, default,
        )
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` when unset or empty."""
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        logger.error(
            "Invalid value for %s: %r (expected a number); using default %s",
            name, val, default,
        )
        return default


def _load_diarizer() -> None:
    """Conditionally load the sherpa-onnx diarization pipeline."""
    global _diarization_enabled
    if os.environ.get("WHISPER_DIARIZATION", "").strip().lower() != "true":
        return
    try:
        import diarizer
        cache_dir = os.environ.get("HF_HOME", "/var/lib/whisper")
        num_speakers = _env_int("WHISPER_DIARIZE_NUM_SPEAKERS", -1)
        threshold = _env_float("WHISPER_DIARIZE_THRESHOLD", 0.5)
        logger.info("Loading diarization pipeline...")
        diarizer.load(
            cache_dir=cache_dir,
            num_speakers=num_speakers,
            cluster_threshold=threshold,
        )
        _diarization_enabled = True
        logger.info("Diarization enabled.")
    except Exception as exc:
        logger.warning("Failed to load diarizer (diarization disabled): %s", exc)
        _diarization_enabled = False


def _load_model() -> None:
    """Import and initialise the faster-whisper model from environment config."""
    global _model, _model_name, _model_config, _beam_size, _max_request_beam, _word_timestamps, _max_upload_bytes, _idle_unload_seconds

    from faster_whisper import WhisperModel  # deferred — keeps import fast

    model_name       = os.environ.get("WHISPER_MODEL",        "base").strip()
    device           = os.environ.get("WHISPER_DEVICE",       "cpu").strip()
    compute_type     = os.environ.get("WHISPER_COMPUTE_TYPE", "int8").strip()
    threads          = _env_int("WHISPER_THREADS", 2)
    cache_dir        = os.environ.get("HF_HOME", "/var/lib/whisper")
    local_files_only = bool(os.environ.get("WHISPER_LOCAL_ONLY", "").strip())
    _beam_size       = _env_int("WHISPER_BEAM", 5)
    _max_request_beam = _env_int("WHISPER_MAX_REQUEST_BEAM", 10)
    _word_timestamps = os.environ.get("WHISPER_WORD_TIMESTAMPS", "").strip().lower() == "true"
    max_upload_mb    = _env_int("WHISPER_MAX_UPLOAD_MB", 1024)
    _idle_unload_seconds = _env_int("WHISPER_IDLE_UNLOAD_SECONDS", 0)
    if _max_request_beam < 0:
        logger.error(
            "Invalid value for WHISPER_MAX_REQUEST_BEAM: %d (expected 0 or greater); using default 10",
            _max_request_beam,
        )
        _max_request_beam = 10
    if max_upload_mb < 0:
        logger.error(
            "Invalid value for WHISPER_MAX_UPLOAD_MB: %d (expected 0 or greater); using default 1024",
            max_upload_mb,
        )
        max_upload_mb = 1024
    _max_upload_bytes = max_upload_mb * 1024 * 1024

    _model_config = {
        "model_name": model_name,
        "device": device,
        "compute_type": compute_type,
        "threads": threads,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    _load_model_from_config()


def _load_model_from_config() -> None:
    """Load the configured model if it is not currently resident."""
    global _model, _model_name, _last_model_used_at
    if _model is not None:
        return
    if not _model_config:
        _load_model()
        return
    from faster_whisper import WhisperModel
    cfg = _model_config
    logger.info(
        "Loading model '%s' | device=%s compute_type=%s threads=%d local_only=%s cache=%s",
        cfg["model_name"], cfg["device"], cfg["compute_type"], cfg["threads"], cfg["local_files_only"], cfg["cache_dir"],
    )
    t0 = time.monotonic()
    _model = WhisperModel(
        cfg["model_name"],
        device=cfg["device"],
        compute_type=cfg["compute_type"],
        cpu_threads=cfg["threads"],
        download_root=cfg["cache_dir"],
        local_files_only=cfg["local_files_only"],
    )
    _model_name = cfg["model_name"]
    _last_model_used_at = time.monotonic()
    logger.info("Model '%s' ready in %.1fs", _model_name, time.monotonic() - t0)


def _release_model(reason: str) -> None:
    """Unload the resident model and ask CUDA-capable libraries to free caches."""
    global _model
    if _model is None:
        return
    logger.info("Unloading model '%s' (%s)", _model_name, reason)
    _model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("CUDA cache cleanup skipped: %s", exc)


async def _idle_unload_monitor() -> None:
    """Periodically unload the model after WHISPER_IDLE_UNLOAD_SECONDS."""
    if _idle_unload_seconds <= 0:
        return
    logger.info("Idle model unload enabled after %ds", _idle_unload_seconds)
    while True:
        await asyncio.sleep(max(5, min(60, _idle_unload_seconds // 2 or 5)))
        if _model is None or _active_inferences > 0:
            continue
        if time.monotonic() - _last_model_used_at < _idle_unload_seconds:
            continue
        if not _inference_lock.acquire(blocking=False):
            continue
        try:
            if _active_inferences == 0 and _model is not None and time.monotonic() - _last_model_used_at >= _idle_unload_seconds:
                _release_model(f"idle for {_idle_unload_seconds}s")
        finally:
            _inference_lock.release()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _idle_monitor_task
    _load_model()
    _load_diarizer()
    if _idle_unload_seconds > 0:
        _idle_monitor_task = asyncio.create_task(_idle_unload_monitor())
    try:
        yield
    finally:
        if _idle_monitor_task:
            _idle_monitor_task.cancel()
        with _inference_lock:
            _release_model("shutdown")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ScribeCrate",
    description=(
        "Open-source, self-hosted transcription and English translation API "
        "powered by faster-whisper. OpenAI-compatible transcription and "
        "translation endpoints.\n\n"
        "https://github.com/hwdsl2/scribecrate"
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    """
    If WHISPER_API_KEY is set, require a matching Bearer token.
    If the env var is empty or unset the endpoint is open (no auth).
    """
    required = os.environ.get("WHISPER_API_KEY", "").strip()
    if not required:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    parts = authorization.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header. Expected: Bearer <key>",
        )
    if parts[1] != required:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _merge_form_lists(*values: Optional[List[str]]) -> Optional[List[str]]:
    merged = []
    for value in values:
        if not value:
            continue
        merged.extend(item for item in value if item is not None)
    return merged or None


def _validate_openai_transcription_compat(
    model: str,
    response_format: str,
    include: Optional[List[str]] = None,
    chunking_strategy: Optional[str] = None,
    known_speaker_names: Optional[List[str]] = None,
    known_speaker_references: Optional[List[str]] = None,
) -> None:
    model_id = model.strip().lower()
    if model_id == _UNSUPPORTED_DIARIZE_MODEL:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model 'gpt-4o-transcribe-diarize' is not supported. "
                "Local diarization is available via WHISPER_DIARIZATION=true "
                "and verbose_json output."
            ),
        )
    if response_format.strip().lower() == "diarized_json":
        raise HTTPException(
            status_code=400,
            detail=(
                "response_format='diarized_json' is not supported. Local diarization, "
                "when enabled, annotates verbose_json segments with speaker labels."
            ),
        )
    if include:
        raise HTTPException(
            status_code=400,
            detail="The include field is not supported; OpenAI-compatible logprobs are not available from faster-whisper.",
        )
    if chunking_strategy is not None and chunking_strategy.strip():
        raise HTTPException(
            status_code=400,
            detail="chunking_strategy is not supported by this server.",
        )
    if known_speaker_names:
        raise HTTPException(
            status_code=400,
            detail="known_speaker_names is not supported; local diarization is unsupervised.",
        )
    if known_speaker_references:
        raise HTTPException(
            status_code=400,
            detail="known_speaker_references is not supported; local diarization does not accept speaker reference audio.",
        )


def _validate_temperature(temperature: float) -> None:
    if not 0 <= temperature <= 1:
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 1.",
        )


def _resolve_request_beam(beam: Optional[str]) -> int:
    """Resolve the optional local beam override for a single request."""
    if beam is None:
        return _beam_size

    value = beam.strip()
    if not value or not value.isascii() or not value.isdecimal() or value.startswith("0"):
        raise HTTPException(status_code=400, detail="beam must be a positive integer.")

    request_beam = int(value)
    if _max_request_beam and request_beam > _max_request_beam:
        raise HTTPException(
            status_code=400,
            detail=(
                f"beam must be less than or equal to {_max_request_beam}. "
                "Set WHISPER_MAX_REQUEST_BEAM=0 to disable this per-request cap."
            ),
        )

    return request_beam


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _fmt_ts(seconds: float, fmt: str) -> str:
    """Format a float second offset as an SRT or VTT timestamp string."""
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    sep = "," if fmt == "srt" else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _to_srt(segments, speaker_map=None) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip() if hasattr(seg, "text") else seg.get("text", "").strip()
        start = seg.start if hasattr(seg, "start") else seg["start"]
        end = seg.end if hasattr(seg, "end") else seg["end"]
        speaker = speaker_map.get(i - 1) if speaker_map else None
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(
            f"{i}\n"
            f"{_fmt_ts(start, 'srt')} --> {_fmt_ts(end, 'srt')}\n"
            f"{prefix}{text}\n"
        )
    return "\n".join(lines)


def _to_vtt(segments, speaker_map=None) -> str:
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(segments):
        text = seg.text.strip() if hasattr(seg, "text") else seg.get("text", "").strip()
        start = seg.start if hasattr(seg, "start") else seg["start"]
        end = seg.end if hasattr(seg, "end") else seg["end"]
        speaker = speaker_map.get(i) if speaker_map else None
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(
            f"{_fmt_ts(start, 'vtt')} --> {_fmt_ts(end, 'vtt')}\n"
            f"{prefix}{text}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------


async def _stream_sse(
    tmp_path: str,
    lang: Optional[str],
    prompt: Optional[str],
    temperature: float,
    beam_size: int,
    task: str = "transcribe",
):
    """
    Async generator that yields Server-Sent Events (SSE) frames using the
    OpenAI streaming transcription protocol.

    Event types emitted (see https://developers.openai.com/api/docs/guides/speech-to-text):
      - transcript.text.delta  — incremental text as each segment is decoded
      - transcript.text.done   — final assembled transcript

    Inference runs in a thread-pool worker so the event loop stays responsive
    while the CPU-bound model decodes the audio.  _inference_lock ensures that
    only one transcription (batch or streaming) runs at a time.

    The temporary audio file is deleted when the generator finishes (or is
    abandoned by the client).
    """
    loop = asyncio.get_running_loop()
    seg_queue: asyncio.Queue = asyncio.Queue()

    def _run() -> None:
        global _active_inferences, _last_model_used_at
        with _inference_lock:
            _active_inferences += 1
            try:
                _load_model_from_config()
                segs_gen, _ = _model.transcribe(
                    tmp_path,
                    language=lang,
                    task=task,
                    initial_prompt=prompt or None,
                    temperature=temperature,
                    beam_size=beam_size,
                    vad_filter=True,
                )
                for seg in segs_gen:
                    loop.call_soon_threadsafe(seg_queue.put_nowait, seg)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(seg_queue.put_nowait, exc)
            finally:
                _last_model_used_at = time.monotonic()
                _active_inferences = max(0, _active_inferences - 1)
                loop.call_soon_threadsafe(seg_queue.put_nowait, None)  # sentinel

    loop.run_in_executor(None, _run)
    text_parts: list = []

    try:
        while True:
            item = await seg_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                logger.error("Streaming transcription error: %s", item)
                err_payload = json.dumps({
                    "error": {
                        "type": "transcription_error",
                        "message": str(item),
                    }
                })
                yield f"data: {err_payload}\n\n"
                return
            seg_text = item.text.strip()
            if seg_text:
                # Prepend space to separate from previous segments
                delta = seg_text if not text_parts else " " + seg_text
                text_parts.append(seg_text)
                payload = json.dumps({
                    "type": "transcript.text.delta",
                    "delta": delta,
                })
                yield f"data: {payload}\n\n"
        # Final frame — complete transcript (OpenAI transcript.text.done event)
        full_text = " ".join(text_parts).strip()
        yield f'data: {json.dumps({"type": "transcript.text.done", "text": full_text})}\n\n'
        yield "data: [DONE]\n\n"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", include_in_schema=False)
async def health():
    """Container liveness probe — used by run.sh to detect startup completion."""
    return {"status": "ok", "model": _model_name}


@app.get("/v1/models")
async def list_models(_auth: None = Depends(_verify_api_key)):
    """
    List the active model in OpenAI-compatible format.
    Any app that queries /v1/models before sending requests will work correctly.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name or "whisper-1",
                "object": "model",
                "created": 0,
                "owned_by": "faster-whisper",
            }
        ],
    }


async def _handle_audio(
    task: str,
    file: UploadFile,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: float,
    stream: Optional[str],
    beam: Optional[str],
    timestamp_granularities: Optional[List[str]] = None,
):
    """
    Shared implementation for transcription and translation endpoints.
    ``task`` is either ``"transcribe"`` or ``"translate"``.
    """
    # Model may have been unloaded by the idle monitor; it is reloaded below
    # under _inference_lock immediately before transcription.

    _validate_temperature(temperature)
    request_beam = _resolve_request_beam(beam)

    # Block translation on English-only models
    if task == "translate" and _model_name and _model_name.endswith(".en"):
        raise HTTPException(
            status_code=400,
            detail=f"Translation is not supported with English-only model '{_model_name}'. "
                   "Use a multilingual model (e.g. base, small, large-v3-turbo).",
        )

    # Normalise the stream form field: the string "true" (case-insensitive) enables streaming.
    # Using Optional[str] instead of bool avoids Pydantic version-dependent coercion differences.
    stream_flag: bool = stream is not None and stream.strip().lower() == "true"

    # Resolve word_timestamps:
    #   timestamp_granularities (spec-compliant) > WHISPER_WORD_TIMESTAMPS env var > False
    wt_flag: bool
    if timestamp_granularities and "word" in timestamp_granularities:
        wt_flag = True
    else:
        wt_flag = _word_timestamps

    # Validate response_format (only relevant for non-streaming responses)
    valid_formats = {"json", "text", "verbose_json", "srt", "vtt"}
    if not stream_flag and response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format '{response_format}'. "
                   f"Must be one of: {', '.join(sorted(valid_formats))}",
        )

    # Resolve language: per-request param > WHISPER_LANGUAGE env var > None (autodetect)
    env_lang = os.environ.get("WHISPER_LANGUAGE", "auto").strip()
    if language and language.lower() != "auto":
        lang = language
    elif env_lang and env_lang.lower() != "auto":
        lang = env_lang
    else:
        lang = None  # faster-whisper autodetects when None

    # Persist uploaded bytes to a temp file so faster-whisper can open it.
    original_name = file.filename or "audio"
    suffix = os.path.splitext(original_name)[1] or ".audio"
    tmp_path: Optional[str] = None
    upload_size = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                upload_size += len(chunk)
                if _max_upload_bytes and upload_size > _max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded audio is too large. "
                            f"Maximum allowed size is {_max_upload_bytes // (1024 * 1024)} MB."
                        ),
                    )
                tmp.write(chunk)
    except HTTPException:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    except Exception as exc:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.exception("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    logger.info(
        "%s '%s' (%d bytes) | lang=%s format=%s stream=%s beam=%d word_ts=%s",
        "Translating" if task == "translate" else "Transcribing",
        original_name, upload_size, lang or "auto", response_format, stream_flag, request_beam, wt_flag,
    )

    # ------------------------------------------------------------------
    # Streaming path — inference runs in a thread; temp file cleaned up
    # inside the generator when the stream ends or the client disconnects.
    # Word timestamps are silently ignored in streaming mode.
    # ------------------------------------------------------------------
    if stream_flag:
        return StreamingResponse(
            _stream_sse(tmp_path, lang, prompt, temperature, request_beam, task),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",   # disable nginx proxy buffering
                "Cache-Control": "no-cache",
            },
        )

    # ------------------------------------------------------------------
    # Batch path
    # ------------------------------------------------------------------
    try:
        try:
            with _inference_lock:
                global _active_inferences, _last_model_used_at
                _active_inferences += 1
                try:
                    _load_model_from_config()
                    segments_gen, info = _model.transcribe(
                        tmp_path,
                        language=lang,
                        task=task,
                        initial_prompt=prompt or None,
                        temperature=temperature,
                        beam_size=request_beam,
                        word_timestamps=wt_flag,
                        vad_filter=True,
                    )
                    segments = list(segments_gen)  # consume the generator before the temp file is removed
                finally:
                    _last_model_used_at = time.monotonic()
                    _active_inferences = max(0, _active_inferences - 1)

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Transcription failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

        # ------------------------------------------------------------------
        # Diarization (optional post-processing)
        # ------------------------------------------------------------------
        speaker_map = None  # {segment_index: speaker_label}
        if _diarization_enabled:
            try:
                import diarizer
                turns = diarizer.diarize(tmp_path)
                # Build lightweight segment dicts for alignment
                seg_dicts = [
                    {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                    for seg in segments
                ]
                diarizer.assign_speakers(seg_dicts, turns)
                speaker_map = {i: d["speaker"] for i, d in enumerate(seg_dicts)}
            except Exception as exc:
                logger.warning("Diarization failed (returning without speakers): %s", exc)

    finally:
        # Clean up temp file after both transcription and diarization
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    full_text = " ".join(seg.text.strip() for seg in segments).strip()

    if response_format == "text":
        if speaker_map:
            lines = []
            prev_speaker = None
            for i, seg in enumerate(segments):
                spk = speaker_map.get(i, "SPEAKER_00")
                text = seg.text.strip()
                if spk != prev_speaker:
                    lines.append(f"[{spk}] {text}")
                    prev_speaker = spk
                else:
                    lines.append(text)
            return PlainTextResponse("\n".join(lines))
        return PlainTextResponse(full_text)

    if response_format == "srt":
        return PlainTextResponse(_to_srt(segments, speaker_map), media_type="text/plain")

    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(segments, speaker_map), media_type="text/plain")

    if response_format == "verbose_json":
        seg_list = []
        all_words = []
        for idx, seg in enumerate(segments):
            seg_dict = {
                "id": idx,
                "seek": seg.seek,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "tokens": seg.tokens,
                "temperature": round(seg.temperature, 3) if seg.temperature is not None else temperature,
                "avg_logprob": round(seg.avg_logprob, 4),
                "compression_ratio": round(seg.compression_ratio, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            }
            if speaker_map:
                seg_dict["speaker"] = speaker_map.get(idx, "SPEAKER_00")
            if wt_flag and seg.words:
                all_words.extend(
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                    for w in seg.words
                )
            seg_list.append(seg_dict)
        resp = {
            "task": task,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
            "duration_after_vad": round(info.duration_after_vad, 3),
            "text": full_text,
            "segments": seg_list,
        }
        if wt_flag:
            resp["words"] = all_words
        return JSONResponse(resp)

    # Default: json — matches OpenAI's minimal response shape
    return JSONResponse({"text": full_text})


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(
        default="whisper-1",
        description="Model identifier (ignored — active model is used)",
    ),
    language: Optional[str] = Form(
        default=None,
        description="BCP-47 language code (e.g. 'en'). Omit or set to 'auto' for autodetect.",
    ),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional text to guide the model's style or continue a previous segment.",
    ),
    response_format: str = Form(
        default="json",
        description="Output format: json, text, verbose_json, srt, vtt",
    ),
    temperature: float = Form(
        default=0.0,
        description="Sampling temperature between 0 and 1.",
    ),
    beam: Optional[str] = Form(
        default=None,
        description=(
            "Local faster-whisper extension: per-request beam size. "
            "Omit to use WHISPER_BEAM."
        ),
    ),
    stream: Optional[str] = Form(
        default=None,
        description=(
            "Stream segments as Server-Sent Events (text/event-stream). "
            "When true, the response is a series of 'data:' frames — one per "
            "decoded segment — followed by a final 'done' frame. "
            "response_format is ignored when stream=true."
        ),
    ),
    timestamp_granularities: Optional[List[str]] = Form(
        default=None,
        alias="timestamp_granularities[]",
        description=(
            "The timestamp granularities to populate for this transcription. "
            "response_format must be verbose_json. Supported values: 'word', 'segment'. "
            "Default: ['segment']."
        ),
    ),
    include: Optional[List[str]] = Form(
        default=None,
        description="Unsupported OpenAI response inclusions such as logprobs.",
    ),
    include_brackets: Optional[List[str]] = Form(default=None, alias="include[]"),
    chunking_strategy: Optional[str] = Form(
        default=None,
        description="Unsupported OpenAI chunking strategy.",
    ),
    known_speaker_names: Optional[List[str]] = Form(
        default=None,
        description="Unsupported OpenAI diarization speaker-name hints.",
    ),
    known_speaker_names_brackets: Optional[List[str]] = Form(default=None, alias="known_speaker_names[]"),
    known_speaker_references: Optional[List[str]] = Form(
        default=None,
        description="Unsupported OpenAI diarization speaker-reference audio.",
    ),
    known_speaker_references_brackets: Optional[List[str]] = Form(default=None, alias="known_speaker_references[]"),
    _auth: None = Depends(_verify_api_key),
):
    """
    Transcribe an audio file.

    Supports standard Whisper-style OpenAI multipart/form-data parameters.
    Newer OpenAI-only diarization and logprob fields that this server cannot
    implement return clear 400 errors instead of being silently ignored.

    When stream=true the response is text/event-stream (SSE) using the OpenAI
    streaming transcription protocol.  Each event carries a JSON object:
      - delta frames: {"type":"transcript.text.delta","delta":"..."}
      - final frame:  {"type":"transcript.text.done","text":"full transcript"}
      - error frame:  {"error":{"type":"...","message":"..."}}

    Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac
    (all formats supported by ffmpeg).
    """
    _validate_openai_transcription_compat(
        model=model,
        response_format=response_format,
        include=_merge_form_lists(include, include_brackets),
        chunking_strategy=chunking_strategy,
        known_speaker_names=_merge_form_lists(known_speaker_names, known_speaker_names_brackets),
        known_speaker_references=_merge_form_lists(known_speaker_references, known_speaker_references_brackets),
    )
    return await _handle_audio(
        task="transcribe",
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        stream=stream,
        beam=beam,
        timestamp_granularities=timestamp_granularities,
    )


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(..., description="Audio file to translate to English"),
    model: str = Form(
        default="whisper-1",
        description="Model identifier (ignored — active model is used)",
    ),
    language: Optional[str] = Form(
        default=None,
        description="BCP-47 language code of the source audio (e.g. 'fr'). Omit for autodetect.",
    ),
    prompt: Optional[str] = Form(
        default=None,
        description="Optional text to guide the model's style or continue a previous segment.",
    ),
    response_format: str = Form(
        default="json",
        description="Output format: json, text, verbose_json, srt, vtt",
    ),
    temperature: float = Form(
        default=0.0,
        description="Sampling temperature between 0 and 1.",
    ),
    beam: Optional[str] = Form(
        default=None,
        description=(
            "Local faster-whisper extension: per-request beam size. "
            "Omit to use WHISPER_BEAM."
        ),
    ),
    stream: Optional[str] = Form(
        default=None,
        description=(
            "Stream segments as Server-Sent Events (text/event-stream). "
            "When true, the response is a series of 'data:' frames — one per "
            "decoded segment — followed by a final 'done' frame. "
            "response_format is ignored when stream=true."
        ),
    ),
    _auth: None = Depends(_verify_api_key),
):
    """
    Translate audio to English text.

    Compatible with common OpenAI translation request fields. The output is
    always in English. language and stream are local extensions.

    Not supported with English-only (.en) models — use a multilingual model.

    Supported audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac
    (all formats supported by ffmpeg).
    """
    return await _handle_audio(
        task="translate",
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        stream=stream,
        beam=beam,
    )


# ---------------------------------------------------------------------------
# Entry point (used by run.sh: python3 /opt/src/api_server.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = _env_int("WHISPER_PORT", 9000)
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        log_level=_log_level_str.lower(),
        workers=1,  # single worker — model is loaded into process memory
    )

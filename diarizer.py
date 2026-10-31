"""
Speaker diarization via sherpa-onnx (ONNX Runtime, no PyTorch).

Downloads ONNX models on first use and provides speaker-segment alignment
for Whisper transcription output.

https://github.com/hwdsl2/scribecrate

Copyright (C) 2026 Lin Song <linsongui@gmail.com>

This work is licensed under the MIT License
See: https://opensource.org/licenses/MIT
"""

import logging

import numpy as np
import sherpa_onnx

from diarization_models import (
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS as MODEL_ASSETS,
    SEGMENTATION_MODEL, ensure_models,
)

logger = logging.getLogger("whisper_server.diarizer")

# ---------------------------------------------------------------------------
# Model URLs (GitHub releases from k2-fsa/sherpa-onnx)
# ---------------------------------------------------------------------------

_SEG_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_EMB_MODEL_BASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
)

_SEG_MODEL_REL = SEGMENTATION_MODEL
_EMB_MODEL_REL = DEFAULT_EMBEDDING_MODEL
_EMB_MODEL_URL = _EMB_MODEL_BASE_URL + _EMB_MODEL_REL
EMBEDDING_MODELS = tuple(MODEL_ASSETS)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_sd = None  # OfflineSpeakerDiarization instance


def _check_embedding_model(embedding_model) -> None:
    """Accept only known release filenames, so no other path or URL is built."""
    if type(embedding_model) is not str or embedding_model not in EMBEDDING_MODELS:
        raise ValueError(
            "embedding_model must be one of: " + ", ".join(EMBEDDING_MODELS)
        )


def _ensure_models(cache_dir: str, embedding_model: str = _EMB_MODEL_REL,
                   *, local_files_only: bool = False) -> tuple:
    """Verify cached models and atomically download missing files when allowed."""
    _check_embedding_model(embedding_model)
    return ensure_models(cache_dir, embedding_model, local_files_only=local_files_only)


def load(
    cache_dir: str = "/var/lib/whisper",
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
    num_threads: int = 1,
    embedding_model: str = _EMB_MODEL_REL,
    *,
    local_files_only: bool = False,
) -> None:
    """Download ONNX models if needed, initialize the diarization pipeline."""
    global _sd

    # Reject invalid direct calls before downloads or native initialization.
    # bool is an int subclass, but is not a meaningful thread configuration.
    if type(num_threads) is not int or not 1 <= num_threads <= 2147483647:
        raise ValueError("num_threads must be an integer from 1 to 2147483647.")
    _check_embedding_model(embedding_model)

    seg_path, emb_path = _ensure_models(cache_dir, embedding_model, local_files_only=local_files_only)

    # If the speaker count is unknown, use threshold-based auto clustering.
    num_clusters = num_speakers if num_speakers > 0 else -1

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            num_threads=num_threads,
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_path,
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=emb_path, num_threads=num_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=cluster_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    if not config.validate():
        raise RuntimeError(
            "Diarization config validation failed. Check that model files exist."
        )

    _sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    logger.info(
        "Diarization pipeline ready (sample_rate=%d, num_clusters=%d, threshold=%.2f, num_threads=%d per model, embedding_model=%s)",
        _sd.sample_rate,
        num_clusters,
        cluster_threshold,
        num_threads,
        embedding_model,
    )


def is_loaded() -> bool:
    """Return True if the diarization pipeline is initialized."""
    return _sd is not None


def _load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load and resample audio file to mono float32 at target_sr using PyAV.
    Uses the same approach as faster-whisper's decode_audio (AudioResampler).
    PyAV is already installed as a dependency of faster-whisper and supports
    all formats that FFmpeg supports (mp3, ogg, m4a, webm, wav, flac, etc.).
    """
    import io

    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )

    raw_buffer = io.BytesIO()
    with av.open(audio_path, mode="r", metadata_errors="ignore") as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                raw_buffer.write(resampled.to_ndarray())
        # Flush remaining buffered samples from the resampler
        for resampled in resampler.resample(None):
            if resampled.samples > 0:
                raw_buffer.write(resampled.to_ndarray())

    audio = np.frombuffer(raw_buffer.getvalue(), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio


def diarize(audio_path: str):
    """
    Run diarization on an audio file.

    Returns a list of (start, end, speaker_label) tuples sorted by start time.
    """
    if _sd is None:
        raise RuntimeError("Diarizer not loaded. Call diarizer.load() first.")

    audio = _load_audio(audio_path, target_sr=_sd.sample_rate)

    result = _sd.process(audio).sort_by_start_time()
    turns = []
    for r in result:
        turns.append((r.start, r.end, f"SPEAKER_{r.speaker:02d}"))
    return turns


def assign_speakers(segments, diarization_turns):
    """
    Assign a speaker label to each Whisper segment based on maximum time overlap
    with diarization turns.

    Args:
        segments: list of segment dicts with 'start' and 'end' keys (seconds).
        diarization_turns: list of (start, end, speaker_label) tuples.

    Returns:
        The same segments list with a 'speaker' key added to each segment.
    """
    if not diarization_turns:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for turn_start, turn_end, speaker in diarization_turns:
            # Calculate overlap
            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        seg["speaker"] = best_speaker

    return segments

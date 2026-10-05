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
import os
import subprocess
import tarfile

import numpy as np
import sherpa_onnx

logger = logging.getLogger("whisper_server.diarizer")

# ---------------------------------------------------------------------------
# Model URLs (GitHub releases from k2-fsa/sherpa-onnx)
# ---------------------------------------------------------------------------

_SEG_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_EMB_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

_SEG_MODEL_REL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
_EMB_MODEL_REL = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_sd = None  # OfflineSpeakerDiarization instance


def _download_file(url: str, dest: str) -> None:
    """Download a file using curl (available in the container)."""
    logger.info("Downloading %s -> %s", url, dest)
    subprocess.run(
        ["curl", "-fSL", "--retry", "3", "--retry-delay", "2", "-o", dest, url],
        check=True,
    )


def _ensure_models(cache_dir: str) -> tuple:
    """Download ONNX models if not already present. Returns (seg_path, emb_path)."""
    seg_path = os.path.join(cache_dir, _SEG_MODEL_REL)
    emb_path = os.path.join(cache_dir, _EMB_MODEL_REL)

    if not os.path.isfile(seg_path):
        archive = os.path.join(cache_dir, "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
        _download_file(_SEG_MODEL_URL, archive)
        logger.info("Extracting segmentation model...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=cache_dir, filter="data")
        os.unlink(archive)

    if not os.path.isfile(emb_path):
        _download_file(_EMB_MODEL_URL, emb_path)

    return seg_path, emb_path


def load(
    cache_dir: str = "/var/lib/whisper",
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
) -> None:
    """Download ONNX models if needed, initialize the diarization pipeline."""
    global _sd

    seg_path, emb_path = _ensure_models(cache_dir)

    # If the speaker count is unknown, use threshold-based auto clustering.
    num_clusters = num_speakers if num_speakers > 0 else -1

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_path,
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_path),
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
        "Diarization pipeline ready (sample_rate=%d, num_clusters=%d, threshold=%.2f)",
        _sd.sample_rate,
        num_clusters,
        cluster_threshold,
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

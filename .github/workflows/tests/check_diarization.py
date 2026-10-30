"""Real CPU diarization verification and warm-run thread comparisons.

Use only public, checksum-verified audio and disposable model/output paths.
No production configuration is changed. See README.md for fixture attribution.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from unittest import mock


FIXTURE_SHA256 = "d9ab6bdcd8137223386e8a0ac1d0a723f7ed029310111b9c60ef9ed6cedb4828"


def load_api(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("diarization_candidate", path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def system_limits():
    limits = {}
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            limits[name] = path.read_text().strip()
    if hasattr(os, "sched_getaffinity"):
        limits["affinity"] = sorted(os.sched_getaffinity(0))
    return limits


def memory_bytes():
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5))
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


async def request(api, audio, stream=False, response_format="verbose_json"):
    class Upload:
        filename = audio.name
        def __init__(self):
            self.file = audio.open("rb")
        async def read(self, size):
            return self.file.read(size)
    upload = Upload()
    try:
        response = await api._handle_audio(
            task="transcribe", file=upload, model="whisper-1", language="en",
            prompt=None, response_format=response_format, temperature=0,
            stream="true" if stream else None, beam=None,
        )
        if stream:
            events = [frame async for frame in response.body_iterator]
            assert "data: [DONE]\n\n" in events, "SSE did not complete"
            done = [json.loads(frame[6:]) for frame in events
                    if frame.startswith("data: {") and '"transcript.text.done"' in frame]
            assert len(done) == 1 and done[0]["text"].strip(), "SSE has no speech"
            assert not any('"speaker"' in frame for frame in events), "SSE gained diarization"
            return done[0]
        if response_format == "verbose_json":
            result = json.loads(response.body)
            assert result["text"].strip() and result["segments"], "ASR has no speech segments"
            assert all(segment.get("speaker", "").startswith("SPEAKER_") for segment in result["segments"])
            return result
        text = response.body.decode()
        assert "SPEAKER_" in text, response_format + " has no speaker labels"
        return text
    finally:
        upload.file.close()


def check(args):
    assert hashlib.sha256(args.audio.read_bytes()).hexdigest() == FIXTURE_SHA256, "Audio checksum mismatch"
    args.cache.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "WHISPER_MODEL": "tiny", "WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE_TYPE": "int8",
        "WHISPER_THREADS": "2", "WHISPER_BEAM": "5", "WHISPER_DIARIZATION": "true",
        "WHISPER_DIARIZE_NUM_SPEAKERS": "2", "WHISPER_DIARIZE_THRESHOLD": "0.5",
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "true", "WHISPER_NO_REPEAT_NGRAM_SIZE": "0",
        "WHISPER_LOCAL_ONLY": "", "HF_HOME": str(args.cache),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    })
    api = load_api(args.api)
    api._load_decoding_options()
    api._load_model()
    import diarizer
    results = {
        "platform": platform.platform(), "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(), "limits": system_limits(),
        "ram_bytes": memory_bytes(), "processor": platform.processor(),
        "versions": {name: version(name) for name in ("sherpa-onnx", "onnxruntime", "faster-whisper", "ctranslate2", "av", "numpy")},
        "audio_sha256": FIXTURE_SHA256, "model": "tiny", "num_speakers": 2,
        "threshold": 0.5, "runs_per_setting": args.runs, "settings": [],
    }
    for label, override in (("legacy", "1"), ("inherited", ""), ("explicit", "6")):
        os.environ["WHISPER_DIARIZE_THREADS"] = override
        api._load_diarization_options()
        start = time.monotonic()
        # The native pipeline does not expose its config after construction.
        # Capture the argument while still invoking the real native constructor.
        constructor = diarizer.sherpa_onnx.OfflineSpeakerDiarization
        with mock.patch.object(diarizer.sherpa_onnx, "OfflineSpeakerDiarization", wraps=constructor) as native:
            api._load_diarizer()
        assert api._diarization_enabled and diarizer.is_loaded(), "Native pipeline did not load"
        initialization_seconds = time.monotonic() - start
        config = native.call_args.args[0]
        assert config.segmentation.num_threads == api._diarize_threads
        assert config.embedding.num_threads == api._diarize_threads
        assert Path(config.embedding.model).name == api._diarize_embedding_model
        assert diarizer.diarize(str(args.audio)), "Warm-up returned no native diarization turns"
        timings, cpu_seconds, all_turns = [], [], []
        for _ in range(args.runs):
            start = time.monotonic()
            usage = resource.getrusage(resource.RUSAGE_SELF)
            turns = diarizer.diarize(str(args.audio))
            after = resource.getrusage(resource.RUSAGE_SELF)
            timings.append(time.monotonic() - start)
            cpu_seconds.append(after.ru_utime + after.ru_stime - usage.ru_utime - usage.ru_stime)
            assert turns and all(end > begin >= 0 and speaker.startswith("SPEAKER_") for begin, end, speaker in turns)
            all_turns.append(turns)
        output = asyncio.run(request(api, args.audio))
        results["settings"].append({
            "label": label, "override": override, "threads": api._diarize_threads,
            "initialization_seconds": initialization_seconds, "wall_seconds": timings,
            "median_wall_seconds": statistics.median(timings), "cpu_seconds": cpu_seconds,
            "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
            "turns": all_turns, "transcript": output,
            "speaker_counts": [len({turn[2] for turn in turns}) for turns in all_turns],
        })
    # Inherited from WHISPER_DIARIZE_EMBEDDING_MODEL; the default when unset.
    results["embedding_model"] = api._diarize_embedding_model
    for format_name in ("text", "srt", "vtt"):
        asyncio.run(request(api, args.audio, response_format=format_name))

    # A real batch diarization pass overlaps a separate SSE Whisper worker.
    # Preserve production locking/scheduling; do not add a new global lock.
    barrier = threading.Barrier(2, timeout=30)
    intervals = {}
    def batch():
        barrier.wait()
        intervals["batch_start"] = time.monotonic()
        try:
            return diarizer.diarize(str(args.audio))
        finally:
            intervals["batch_end"] = time.monotonic()
    def sse():
        barrier.wait()
        intervals["sse_start"] = time.monotonic()
        try:
            return asyncio.run(request(api, args.audio, stream=True))
        finally:
            intervals["sse_end"] = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch_result = pool.submit(batch)
        stream_result = pool.submit(sse)
        assert batch_result.result(timeout=300), "Overlapping batch produced no turns"
        assert stream_result.result(timeout=300)["text"].strip()
    results["overlapping_batch_sse"] = "passed"
    assert max(intervals["batch_start"], intervals["sse_start"]) < min(intervals["batch_end"], intervals["sse_end"])
    results["overlap_intervals"] = intervals
    results["model_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in (diarizer._SEG_MODEL_REL, api._diarize_embedding_model)
        for path in (args.cache / name,)
    }
    results["whisper_snapshot_ids"] = sorted(path.name for path in args.cache.glob(
        "models--Systran--faster-whisper-tiny/snapshots/*") if path.is_dir())
    for entry in results["settings"]:
        entry["turns_equal_to_legacy"] = entry["turns"][0] == results["settings"][0]["turns"][0]
    # Review comparisons, not cross-hardware byte-identity or a promised speedup.
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    # Keep dependency/resource/quality evidence in CI logs without printing the
    # full public transcript. Peak RSS is the process's cumulative high-water mark.
    summary = {key: value for key, value in results.items() if key != "settings"}
    summary["settings"] = [{key: value for key, value in entry.items()
                            if key not in ("transcript", "turns")}
                           for entry in results["settings"]]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be positive")
    check(arguments)

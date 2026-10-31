"""CPU audio smoke/controlled comparisons, not a reproduction of issue #11."""
import argparse
import asyncio
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
import sys
from pathlib import Path
import tempfile
import time
import wave

import numpy as np
from faster_whisper.audio import decode_audio

FIXTURE_SHA256 = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"


def load_api(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def request(api, audio, task, stream):
    class Upload:
        filename = audio.name
        def __init__(self):
            self.file = audio.open("rb")
        async def read(self, size):
            return self.file.read(size)
    upload = Upload()
    try:
        response = await api._handle_audio(
            task=task, file=upload, model="whisper-1", language="en", prompt=None,
            response_format="json" if stream else "verbose_json",
            temperature=0, stream="true" if stream else None, beam=None,
        )
        if stream:
            events = [frame async for frame in response.body_iterator]
            done = [
                json.loads(frame[6:]) for frame in events
                if frame.startswith("data: {") and "transcript.text.done" in frame
            ]
            assert len(done) == 1 and "data: [DONE]\n\n" in events
            return {"text": done[0]["text"]}
        return json.loads(response.body)
    finally:
        upload.file.close()


def check(args):
    assert hashlib.sha256(args.audio.read_bytes()).hexdigest() == FIXTURE_SHA256
    os.environ.update({
        "WHISPER_MODEL": "tiny", "WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE_TYPE": "int8",
        "WHISPER_THREADS": "2", "WHISPER_BEAM": "5", "WHISPER_WORD_TIMESTAMPS": "",
        "WHISPER_DIARIZATION": "", "HF_HOME": str(args.cache),
        "WHISPER_MAX_UPLOAD_MB": "1024",
    })
    os.environ.pop("WHISPER_LOCAL_ONLY", None)
    for name in ("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "WHISPER_NO_REPEAT_NGRAM_SIZE"):
        os.environ.pop(name, None)
    api = load_api(args.api, "audio_candidate")
    api._load_decoding_options()
    api._load_model()
    baseline = None
    if args.baseline:
        baseline = load_api(args.baseline, "audio_baseline")
        baseline._model = api._model
        baseline._model_name = api._model_name

    with tempfile.TemporaryDirectory() as directory:
        # Repeat a real phrase, then add deterministic low-level nonspeech and
        # later speech. The VAD may remove the noise. This is not reporter audio.
        speech = decode_audio(str(args.audio), sampling_rate=16000)
        count = int(args.noise_seconds * 16000)
        rng = np.random.default_rng(20261017)
        noise = (rng.normal(0, 0.001, count)
                 + 0.001 * np.sin(np.arange(count) * (2 * np.pi * 60 / 16000)))
        long_audio = np.concatenate([speech, speech, noise, speech]).astype(np.float32)
        long_path = Path(directory) / "long-noise-speech.wav"
        with wave.open(str(long_path), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes((np.clip(long_audio, -1, 1) * 32767).astype("<i2").tobytes())
        results = {
            "versions": {name: version(name) for name in ("faster-whisper", "ctranslate2", "av")},
            "fixture_sha256": FIXTURE_SHA256,
            "long_sha256": hashlib.sha256(long_path.read_bytes()).hexdigest(),
            "noise_seconds": args.noise_seconds, "model": "tiny", "threads": 2,
            "device": "cpu", "compute_type": "int8", "runs": [],
        }
        default_text = {}
        for label, condition, ngram in (("unset", None, None), ("defaults", "true", "0"), ("opt-in", "false", "3")):
            for name, value in (
                ("WHISPER_CONDITION_ON_PREVIOUS_TEXT", condition),
                ("WHISPER_NO_REPEAT_NGRAM_SIZE", ngram),
            ):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            api._load_decoding_options()
            for fixture, path in (("short", args.audio), ("long", long_path)):
                # All four transport/task combinations on short speech; long
                # comparisons focus on transcription's timestamped batch path.
                combinations = [("transcribe", False)]
                if fixture == "short":
                    combinations = [(task, stream) for task in ("transcribe", "translate") for stream in (False, True)]
                for task, stream in combinations:
                    started = time.monotonic()
                    response = asyncio.run(request(api, path, task, stream))
                    assert response["text"].strip(), (label, fixture, task, stream)
                    key = (fixture, task, stream)
                    if label == "unset":
                        default_text[key] = response["text"]
                        if baseline is not None:
                            old = asyncio.run(request(baseline, path, task, stream))
                            assert old["text"] == response["text"], "Legacy/default decoding drift"
                    elif label == "defaults":
                        assert response["text"] == default_text[key], "Unset/explicit-default drift"
                    if fixture == "long":
                        late_start = len(speech) * 2 / 16000 + args.noise_seconds
                        assert any(
                            segment["end"] > late_start and segment["text"].strip()
                            for segment in response["segments"]
                        ), "Later speech missing"
                    results["runs"].append({
                        "setting": label, "fixture": fixture, "task": task, "stream": stream,
                        "seconds": round(time.monotonic() - started, 3),
                        "text": response["text"],
                        "duration_after_vad": response.get("duration_after_vad"),
                    })
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print("Real CPU inference: unchanged defaults, all task/transport paths and later speech: OK")
        print("Opt-in transcripts are recorded without a universal quality claim:", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--noise-seconds", type=int, default=600)
    check(parser.parse_args())

"""Exercise the real HTTP stack with a deterministic, lightweight inference spy."""
import argparse
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient



def check_translation_models(api, client, headers, calls, directory):
    """Check real routing/status/content type with deterministic inference."""
    description = api.app.openapi()["paths"]["/v1/audio/translations"]["post"]["description"]
    assert "HTTP 400" in description and "large-v3-turbo" in description
    for loaded in ("base.en", "small.en", "large-v3-turbo", "turbo",
                   "tiny", "base", "small", "medium", "large-v3"):
        with mock.patch.object(api, "_model_name", loaded):
            unsupported = loaded.endswith(".en") or loaded in {"large-v3-turbo", "turbo"}
            if unsupported:
                for auth in ({}, {"Authorization": "Bearer wrong"}):
                    response = client.post(
                        "/v1/audio/translations", headers=auth,
                        files={"file": ("audio.wav", b"fixture audio")},
                        data={"model": "whisper-1", "stream": "true"},
                    )
                    assert response.status_code == 401
            for endpoint, task in (("translations", "translate"), ("transcriptions", "transcribe")):
                for stream in (False, True):
                    # Request labels cannot bypass a blocked loaded model or block a valid one.
                    for request_model in ("whisper-1", "small" if unsupported else "turbo"):
                        before = len(calls)
                        response = client.post(
                            "/v1/audio/" + endpoint, headers=headers,
                            files={"file": ("audio.wav", b"fixture audio", "audio/wav")},
                            data={"model": request_model, "stream": str(stream).lower()},
                        )
                        if task == "translate" and unsupported:
                            assert response.status_code == 400, response.text
                            assert response.headers["content-type"].startswith("application/json")
                            assert loaded in response.json()["detail"]
                            assert "multilingual non-turbo" in response.json()["detail"]
                            assert len(calls) == before, "Rejected request reached inference"
                            assert "data:" not in response.text, "Rejected request began SSE"
                        else:
                            assert response.status_code == 200, response.text
                            assert len(calls) == before + 1
                            assert calls[-1]["task"] == task
                            if stream:
                                assert response.headers["content-type"].startswith("text/event-stream")
                                assert "transcript.text.done" in response.text and "data: [DONE]" in response.text
                            else:
                                assert response.json() == {"text": "hello"}
                        assert not list(Path(directory).iterdir()), "Upload files leaked"


def check(source, baseline=None):
    spec = importlib.util.spec_from_file_location("checked_api", source)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    if baseline is not None:
        old_spec = importlib.util.spec_from_file_location("baseline_api", baseline)
        old_api = importlib.util.module_from_spec(old_spec)
        old_spec.loader.exec_module(old_api)
        assert api.app.openapi() == old_api.app.openapi(), "Public OpenAPI schema changed"
    calls = []
    segment = SimpleNamespace(
        id=0, seek=0, start=0.0, end=1.0, text=" hello ", tokens=[1],
        avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.1,
        temperature=0.0,
        words=[SimpleNamespace(start=0.0, end=1.0, word="hello", probability=1.0)],
    )
    info = SimpleNamespace(
        language="en", language_probability=1.0, duration=1.0, duration_after_vad=1.0,
    )

    class Model:
        def transcribe(self, path, **kwargs):
            assert api._inference_lock.locked(), "Inference escaped its lock"
            assert Path(path).is_file()
            calls.append(kwargs)
            def segments():
                assert api._inference_lock.locked(), "Lazy decoding escaped its lock"
                yield segment
            return segments(), info

    def load():
        api._model = Model()
        api._model_name = "tiny"

    with tempfile.TemporaryDirectory() as directory:
        with mock.patch.object(api.tempfile, "tempdir", directory):
            for condition, ngram in (("", ""), ("true", "0"), ("false", "3")):
                env = {
                    "WHISPER_API_KEY": "synthetic-http-test-key",
                    "WHISPER_CONDITION_ON_PREVIOUS_TEXT": condition,
                    "WHISPER_NO_REPEAT_NGRAM_SIZE": ngram,
                    "WHISPER_DIARIZATION": "",
                    "WHISPER_DIARIZE_THREADS": "",
                }
                with mock.patch.dict(os.environ, env):
                    with mock.patch.object(api, "_load_model", side_effect=load):
                        with TestClient(api.app) as client:
                            headers = {"Authorization": "Bearer synthetic-http-test-key"}
                            for auth in ({}, {"Authorization": "Bearer wrong"}):
                                assert client.get("/v1/models", headers=auth).status_code == 401
                            assert client.get("/health").status_code == 200
                            for endpoint, task in (
                                ("transcriptions", "transcribe"), ("translations", "translate")
                            ):
                                for stream in (False, True):
                                    response = client.post(
                                        "/v1/audio/" + endpoint, headers=headers,
                                        files={"file": ("audio.wav", b"fixture audio", "audio/wav")},
                                        data={"model": "whisper-1", "stream": str(stream).lower()},
                                    )
                                    assert response.status_code == 200, response.text
                                    call = calls[-1]
                                    assert call["condition_on_previous_text"] is (condition != "false")
                                    assert call["no_repeat_ngram_size"] == int(ngram or "0")
                                    assert call["task"] == task
                                    if stream:
                                        assert "transcript.text.done" in response.text
                                        assert "data: [DONE]" in response.text
                                    else:
                                        assert response.json() == {"text": "hello"}
                            if condition == "":
                                check_translation_models(api, client, headers, calls, directory)
                            for response_format in ("text", "verbose_json", "srt", "vtt"):
                                response = client.post(
                                    "/v1/audio/transcriptions", headers=headers,
                                    files={"file": ("audio.wav", b"fixture audio")},
                                    data={"model": "whisper-1", "response_format": response_format},
                                )
                                assert response.status_code == 200, response.text
                            previous_calls = len(calls)
                            with mock.patch.object(api, "_max_upload_bytes", 1):
                                response = client.post(
                                    "/v1/audio/transcriptions", headers=headers,
                                    files={"file": ("audio.wav", b"oversized")},
                                    data={"model": "whisper-1"},
                                )
                                assert response.status_code == 413
                                assert len(calls) == previous_calls
                            assert not list(Path(directory).iterdir()), "Upload files leaked"
            with mock.patch.dict(os.environ, {
                "WHISPER_API_KEY": "", "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "",
                "WHISPER_NO_REPEAT_NGRAM_SIZE": "", "WHISPER_DIARIZATION": "",
                "WHISPER_DIARIZE_THREADS": "",
            }):
                with mock.patch.object(api, "_load_model", side_effect=load):
                    with TestClient(api.app) as client:
                        assert client.get("/v1/models").status_code == 200
            with mock.patch.dict(os.environ, {"WHISPER_NO_REPEAT_NGRAM_SIZE": "01"}):
                with mock.patch.object(api, "_load_model") as load_model:
                    try:
                        with TestClient(api.app):
                            raise AssertionError("Invalid startup accepted")
                    except ValueError as error:
                        assert "WHISPER_NO_REPEAT_NGRAM_SIZE" in str(error)
                    load_model.assert_not_called()
            with mock.patch.dict(os.environ, {
                "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "", "WHISPER_NO_REPEAT_NGRAM_SIZE": "",
                "WHISPER_DIARIZATION": "", "WHISPER_DIARIZE_THREADS": "6",
            }):
                with mock.patch.object(api, "_load_model", side_effect=load):
                    with TestClient(api.app):
                        assert api._diarize_threads == 6 and not api._diarization_enabled
            for enabled in ("true", "false"):
                for value in ("0", "2147483648", "yes"):
                    with mock.patch.dict(os.environ, {
                        "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "", "WHISPER_NO_REPEAT_NGRAM_SIZE": "",
                        "WHISPER_DIARIZATION": enabled, "WHISPER_DIARIZE_THREADS": value,
                    }):
                        with mock.patch.object(api, "_load_model") as load_model:
                            try:
                                with TestClient(api.app):
                                    raise AssertionError("Invalid diarization thread configuration accepted")
                            except ValueError as error:
                                assert "WHISPER_DIARIZE_THREADS" in str(error)
                            load_model.assert_not_called()
    print("HTTP defaults, overrides, task/transport parity, formats, auth and locks: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    check(args.api.resolve(), args.baseline)

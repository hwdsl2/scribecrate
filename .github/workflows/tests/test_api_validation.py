import importlib
import asyncio
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


class _HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _FastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda func: func

    def post(self, *args, **kwargs):
        return lambda func: func


class _Response:
    def __init__(self, content=None, media_type=None, headers=None):
        self.content = content
        self.media_type = media_type
        self.headers = headers or {}


class _StreamingResponse(_Response):
    def __init__(self, content=None, media_type=None, headers=None):
        super().__init__(content=content, media_type=media_type, headers=headers)
        self.body_iterator = content


def _install_stubs():
    fastapi = types.ModuleType("fastapi")
    fastapi.Depends = lambda *args, **kwargs: None
    fastapi.FastAPI = _FastAPI
    fastapi.File = lambda default=..., **kwargs: default
    fastapi.Form = lambda default=..., **kwargs: default
    fastapi.Header = lambda *args, **kwargs: None
    fastapi.HTTPException = _HTTPException
    fastapi.UploadFile = object
    sys.modules["fastapi"] = fastapi

    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _Response
    responses.PlainTextResponse = _Response
    responses.StreamingResponse = _StreamingResponse
    sys.modules["fastapi.responses"] = responses

    sys.modules["uvicorn"] = types.ModuleType("uvicorn")


_install_stubs()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
api_server = importlib.import_module("api_server")


class OpenAITranscriptionCompatTests(unittest.TestCase):
    def _assert_rejected(self, **kwargs):
        params = {
            "model": "whisper-1",
            "response_format": "json",
            "include": None,
            "chunking_strategy": None,
            "known_speaker_names": None,
            "known_speaker_references": None,
        }
        params.update(kwargs)
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server._validate_openai_transcription_compat(**params)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_standard_request_passes(self):
        api_server._validate_openai_transcription_compat(
            model="whisper-1",
            response_format="json",
            include=None,
            chunking_strategy=None,
            known_speaker_names=None,
            known_speaker_references=None,
        )

    def test_rejects_diarize_model(self):
        self._assert_rejected(model="gpt-4o-transcribe-diarize")

    def test_rejects_diarize_model_case_insensitive(self):
        self._assert_rejected(model=" GPT-4O-TRANSCRIBE-DIARIZE ")

    def test_rejects_diarized_json(self):
        self._assert_rejected(response_format="diarized_json")

    def test_rejects_diarized_json_case_insensitive(self):
        self._assert_rejected(response_format=" DIARIZED_JSON ")

    def test_rejects_include_logprobs(self):
        self._assert_rejected(include=["logprobs"])

    def test_rejects_chunking_strategy(self):
        self._assert_rejected(chunking_strategy="auto")

    def test_rejects_known_speaker_names(self):
        self._assert_rejected(known_speaker_names=["agent"])

    def test_rejects_known_speaker_references(self):
        self._assert_rejected(known_speaker_references=["data:audio/wav;base64,AAAA"])

    def test_empty_optional_unsupported_fields_pass(self):
        api_server._validate_openai_transcription_compat(
            model="whisper-1",
            response_format="json",
            include=[],
            chunking_strategy="   ",
            known_speaker_names=[],
            known_speaker_references=[],
        )

    def test_merge_form_lists(self):
        self.assertEqual(api_server._merge_form_lists(["logprobs"], None), ["logprobs"])
        self.assertEqual(api_server._merge_form_lists(None, ["logprobs"]), ["logprobs"])
        self.assertIsNone(api_server._merge_form_lists(None, []))


class TemperatureValidationTests(unittest.TestCase):
    def test_accepts_temperature_bounds(self):
        api_server._validate_temperature(0)
        api_server._validate_temperature(0.5)
        api_server._validate_temperature(1)

    def test_rejects_temperature_below_zero(self):
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server._validate_temperature(-0.01)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("between 0 and 1", ctx.exception.detail)

    def test_rejects_temperature_above_one(self):
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server._validate_temperature(1.01)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("between 0 and 1", ctx.exception.detail)


class BeamValidationTests(unittest.TestCase):
    def setUp(self):
        self.old_beam_size = api_server._beam_size
        self.old_max_request_beam = api_server._max_request_beam
        api_server._beam_size = 5
        api_server._max_request_beam = 10

    def tearDown(self):
        api_server._beam_size = self.old_beam_size
        api_server._max_request_beam = self.old_max_request_beam

    def test_omitted_beam_uses_global_default(self):
        self.assertEqual(api_server._resolve_request_beam(None), 5)

    def test_accepts_valid_request_beams(self):
        self.assertEqual(api_server._resolve_request_beam("1"), 1)
        self.assertEqual(api_server._resolve_request_beam("5"), 5)
        self.assertEqual(api_server._resolve_request_beam("10"), 10)
        self.assertEqual(api_server._resolve_request_beam(" 7 "), 7)

    def test_rejects_invalid_request_beams(self):
        for value in ("", "   ", "0", "01", "+1", "-1", "abc", "1.5", "1_000"):
            with self.subTest(value=value):
                with self.assertRaises(api_server.HTTPException) as ctx:
                    api_server._resolve_request_beam(value)
                self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_request_beam_above_cap(self):
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server._resolve_request_beam("11")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("less than or equal to 10", ctx.exception.detail)

    def test_allows_uncapped_request_beam(self):
        api_server._max_request_beam = 0
        self.assertEqual(api_server._resolve_request_beam("25"), 25)


class _FakeUpload:
    filename = "audio.wav"

    def __init__(self, data=b"audio"):
        self._data = data

    async def read(self, _size):
        data = self._data
        self._data = b""
        return data


class _FakeSegment:
    text = " hello "


class _FakeInfo:
    language = "en"
    language_probability = 1.0
    duration = 1.0
    duration_after_vad = 1.0


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, _path, **kwargs):
        self.calls.append(kwargs)
        return iter([_FakeSegment()]), _FakeInfo()


class BeamPropagationTests(unittest.TestCase):
    def setUp(self):
        self.old_model = api_server._model
        self.old_model_name = api_server._model_name
        self.old_beam_size = api_server._beam_size
        self.old_max_request_beam = api_server._max_request_beam
        self.old_max_upload_bytes = api_server._max_upload_bytes
        self.old_diarization_enabled = api_server._diarization_enabled
        api_server._model_name = "base"
        api_server._beam_size = 5
        api_server._max_request_beam = 10
        api_server._max_upload_bytes = 0
        api_server._diarization_enabled = False

    def tearDown(self):
        api_server._model = self.old_model
        api_server._model_name = self.old_model_name
        api_server._beam_size = self.old_beam_size
        api_server._max_request_beam = self.old_max_request_beam
        api_server._max_upload_bytes = self.old_max_upload_bytes
        api_server._diarization_enabled = self.old_diarization_enabled

    def test_batch_transcription_uses_request_beam(self):
        model = _FakeModel()
        api_server._model = model

        response = asyncio.run(api_server._handle_audio(
            task="transcribe",
            file=_FakeUpload(),
            model="whisper-1",
            language=None,
            prompt=None,
            response_format="json",
            temperature=0,
            stream=None,
            beam="7",
        ))

        self.assertEqual(response.content, {"text": "hello"})
        self.assertEqual(model.calls[0]["beam_size"], 7)

    def test_batch_transcription_falls_back_to_global_beam(self):
        model = _FakeModel()
        api_server._model = model

        asyncio.run(api_server._handle_audio(
            task="translate",
            file=_FakeUpload(),
            model="whisper-1",
            language=None,
            prompt=None,
            response_format="json",
            temperature=0,
            stream=None,
            beam=None,
        ))

        self.assertEqual(model.calls[0]["beam_size"], 5)

    def test_streaming_transcription_uses_request_beam(self):
        model = _FakeModel()
        api_server._model = model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        async def collect_stream():
            frames = []
            async for frame in api_server._stream_sse(
                tmp_path, lang=None, prompt=None, temperature=0, beam_size=9, task="transcribe"
            ):
                frames.append(frame)
            return frames

        try:
            frames = asyncio.run(collect_stream())
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        self.assertEqual(model.calls[0]["beam_size"], 9)
        self.assertTrue(any("transcript.text.done" in frame for frame in frames))


class _FakeInnerModel:
    """Stand-in for the CTranslate2 engine: records unload_model() calls."""

    def __init__(self):
        self.unload_calls = 0

    def unload_model(self):
        self.unload_calls += 1


class _FakeWhisperModel:
    """Stand-in for faster_whisper.WhisperModel with a loaded CTranslate2 model."""

    def __init__(self):
        self.model = _FakeInnerModel()


class GracefulShutdownReleaseTests(unittest.TestCase):
    """The lifespan teardown must release the CTranslate2 CUDA context so a
    stop never leaves an orphaned VRAM allocation on a shared GPU."""

    def _run_shutdown(self, model):
        def fake_load(*_args, **_kwargs):
            return None

        with mock.patch.object(api_server, "_load_decoding_options", fake_load), \
             mock.patch.object(api_server, "_load_diarization_options", fake_load), \
             mock.patch.object(api_server, "_load_model", fake_load), \
             mock.patch.object(api_server, "_load_diarizer", fake_load):

            async def run_lifespan():
                async with api_server._lifespan(object()):
                    pass

            asyncio.run(run_lifespan())

        return model

    def test_shutdown_releases_loaded_model(self):
        model = _FakeWhisperModel()
        api_server._model = model
        try:
            self._run_shutdown(model)
            self.assertEqual(model.model.unload_calls, 1)
        finally:
            api_server._model = None

    def test_shutdown_is_a_noop_when_no_model(self):
        api_server._model = None
        self._run_shutdown(None)

    def test_shutdown_swallows_unload_errors(self):
        model = _FakeWhisperModel()

        def boom():
            raise RuntimeError("driver gone")

        model.model.unload_model = boom
        api_server._model = model
        try:
            # Must not raise: a failed release must not break the shutdown.
            self._run_shutdown(model)
        finally:
            api_server._model = None


if __name__ == "__main__":
    unittest.main()

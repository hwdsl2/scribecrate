"""Diarization configuration regressions without NumPy, ONNX or downloads."""
import asyncio
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from test_api_validation import api_server


OPTION = "WHISPER_DIARIZE_THREADS"
BAD = [
    "0", "00", "01", "-1", "+1", "1.5", "1_000", "2147483648",
    "9" * 10000, "\u0666", "\uff16", "6\u00a0", "\u00a06", "6\n7",
    "6\ninvalid", "6 7", "'\"6\"'", "'6", '6"', "NaN", "$(id)",
]
VALID = [("", 2), ("1", 1), ("6", 6), ("2147483647", 2147483647),
         (" \t\r\n\v\f' 6 '\t", 6), ('""', 2), ("' \r\n '", 2)]
ROOT = Path(__file__).resolve().parents[3]


class DiarizationOptionTests(unittest.TestCase):
    def setUp(self):
        for patch in (
            mock.patch.dict(os.environ, {
                OPTION: "", "WHISPER_THREADS": "", "WHISPER_DIARIZATION": "true",
                "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "", "WHISPER_NO_REPEAT_NGRAM_SIZE": "",
            }),
            mock.patch.object(api_server, "_diarize_threads", 2),
            mock.patch.object(api_server, "_diarization_enabled", False),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        for name in ("_model", "_model_name", "_beam_size", "_max_request_beam",
                     "_word_timestamps", "_max_upload_bytes"):
            patch = mock.patch.object(api_server, name, getattr(api_server, name))
            patch.start()
            self.addCleanup(patch.stop)

    def test_defaults_inheritance_and_normalization(self):
        for value, expected in VALID:
            with self.subTest(value=value):
                os.environ[OPTION] = value
                api_server._load_diarization_options()
                self.assertEqual(api_server._diarize_threads, expected)
        os.environ.pop(OPTION)
        os.environ.pop("WHISPER_THREADS", None)
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 2)
        os.environ["WHISPER_THREADS"] = "4"
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 4)
        os.environ[OPTION] = "1"
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 1)

    def test_invalid_override_fails_in_both_modes_without_changing_state(self):
        for enabled in ("true", "false"):
            os.environ["WHISPER_DIARIZATION"] = enabled
            for value in BAD:
                with self.subTest(enabled=enabled, value=value[:60]):
                    os.environ[OPTION] = value
                    api_server._diarize_threads = 6
                    with self.assertRaisesRegex(ValueError, OPTION):
                        api_server._load_diarization_options()
                    self.assertEqual(api_server._diarize_threads, 6)

    def test_invalid_startup_never_loads_either_model(self):
        async def start():
            async with api_server._lifespan(api_server.app):
                self.fail("Invalid config began serving")
        for enabled in ("true", "false"):
            os.environ.update({OPTION: "0", "WHISPER_DIARIZATION": enabled})
            with mock.patch.object(api_server, "_load_model") as whisper:
                with mock.patch.object(api_server, "_load_diarizer") as diarizer:
                    with self.assertRaisesRegex(ValueError, OPTION):
                        asyncio.run(start())
                    whisper.assert_not_called()
                    diarizer.assert_not_called()

    def test_direct_python_legacy_nonpositive_whisper_is_preserved(self):
        for value in ("0", "-2"):
            os.environ["WHISPER_THREADS"] = value
            with self.assertLogs(api_server.logger, level="WARNING") as logs:
                api_server._load_diarization_options()
            self.assertEqual(api_server._diarize_threads, 1)
            self.assertIn("legacy diarization", " ".join(logs.output))
            whisper = mock.Mock()
            with mock.patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=whisper)}):
                with mock.patch.object(api_server, "_model"), mock.patch.object(api_server, "_model_name"):
                    api_server._load_model()
            self.assertEqual(whisper.call_args.kwargs["cpu_threads"], int(value))

    def test_inherited_overflow_only_fails_when_used(self):
        for value in ("2147483648", "9" * 100):
            os.environ.update({"WHISPER_THREADS": value, "WHISPER_DIARIZATION": "true"})
            with self.assertRaisesRegex(ValueError, "Inherited WHISPER_THREADS"):
                api_server._load_diarization_options()
            os.environ["WHISPER_DIARIZATION"] = "false"
            api_server._load_diarization_options()
        os.environ.update({OPTION: "6", "WHISPER_DIARIZATION": "true"})
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 6)

    def test_enabled_load_propagates_exact_config_and_retains_failure_policy(self):
        fake = types.ModuleType("diarizer")
        fake.load = mock.Mock()
        os.environ.update({OPTION: "6", "WHISPER_DIARIZE_NUM_SPEAKERS": "2",
                           "WHISPER_DIARIZE_THRESHOLD": "0.7", "HF_HOME": "/fixture/cache"})
        with mock.patch.dict(sys.modules, {"diarizer": fake}):
            api_server._load_diarization_options()
            with self.assertLogs(api_server.logger, level="INFO") as logs:
                api_server._load_diarizer()
            fake.load.assert_called_once_with(cache_dir="/fixture/cache", num_speakers=2,
                                              cluster_threshold=0.7, num_threads=6,
                                              embedding_model=api_server._DEFAULT_DIARIZE_EMBEDDING_MODEL)
            self.assertIn("num_threads=6", " ".join(logs.output))
            self.assertTrue(api_server._diarization_enabled)
            fake.load.side_effect = RuntimeError("fixture native load failure")
            with self.assertLogs(api_server.logger, level="WARNING"):
                api_server._load_diarizer()
            self.assertFalse(api_server._diarization_enabled)

    def test_disabled_diarization_has_no_optional_imports(self):
        original = __import__
        def guarded(name, *args, **kwargs):
            if name in ("diarizer", "sherpa_onnx", "numpy"):
                self.fail("Disabled diarization imported " + name)
            return original(name, *args, **kwargs)
        for inherited in ("0", "-2", "2147483648", "6"):
            os.environ.update({"WHISPER_THREADS": inherited, "WHISPER_DIARIZATION": "false"})
            with mock.patch("builtins.__import__", side_effect=guarded):
                api_server._load_diarization_options()
                api_server._load_diarizer()

    def test_startup_resolves_threads_before_whisper_loading(self):
        os.environ[OPTION] = "6"
        async def start():
            async with api_server._lifespan(api_server.app):
                self.assertEqual(api_server._diarize_threads, 6)
        def check():
            self.assertEqual(api_server._diarize_threads, 6)
        with mock.patch.object(api_server, "_load_model", side_effect=check):
            with mock.patch.object(api_server, "_load_diarizer", side_effect=check):
                asyncio.run(start())


class DiarizerConstructionTests(unittest.TestCase):
    def setUp(self):
        self.native = types.ModuleType("sherpa_onnx")
        for name in ("OfflineSpeakerSegmentationModelConfig",
                     "OfflineSpeakerSegmentationPyannoteModelConfig",
                     "SpeakerEmbeddingExtractorConfig", "FastClusteringConfig"):
            setattr(self.native, name, mock.Mock(side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs)))
        self.config = types.SimpleNamespace(validate=mock.Mock(return_value=True))
        self.native.OfflineSpeakerDiarizationConfig = mock.Mock(return_value=self.config)
        self.native.OfflineSpeakerDiarization = mock.Mock(return_value=types.SimpleNamespace(sample_rate=16000))
        spec = importlib.util.spec_from_file_location("diarizer_thread_fixture", ROOT / "diarizer.py")
        self.diarizer = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy"), "sherpa_onnx": self.native}):
            spec.loader.exec_module(self.diarizer)
        self.model_patch = mock.patch.object(self.diarizer, "_ensure_models", return_value=("seg.onnx", "emb.onnx"))
        self.models = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def test_both_models_receive_threads_and_legacy_positionals_work(self):
        for threads in (1, 2, 6, 2147483647):
            self.diarizer.load("/cache", 2, 0.7, num_threads=threads)
            self.assertEqual(self.native.OfflineSpeakerSegmentationModelConfig.call_args.kwargs["num_threads"], threads)
            self.native.SpeakerEmbeddingExtractorConfig.assert_called_with(model="emb.onnx", num_threads=threads)
            self.native.FastClusteringConfig.assert_called_with(num_clusters=2, threshold=0.7)
            args = self.native.OfflineSpeakerDiarizationConfig.call_args.kwargs
            self.assertEqual((args["min_duration_on"], args["min_duration_off"]), (0.3, 0.5))
            self.assertTrue(self.diarizer.is_loaded())
        self.diarizer.load("/cache", -1, 0.5)
        self.assertEqual(self.native.OfflineSpeakerSegmentationModelConfig.call_args.kwargs["num_threads"], 1)
        self.native.FastClusteringConfig.assert_called_with(num_clusters=-1, threshold=0.5)

    def test_direct_invalid_values_fail_before_download_or_native_setup(self):
        for value in (0, -1, 2147483648, 10**100, True, False, 1.0, "6", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "num_threads"):
                    self.diarizer.load(num_threads=value)
        self.models.assert_not_called()
        self.native.OfflineSpeakerDiarizationConfig.assert_not_called()

    def test_invalid_native_config_does_not_initialize_pipeline(self):
        self.config.validate.return_value = False
        with self.assertRaisesRegex(RuntimeError, "config validation failed"):
            self.diarizer.load(num_threads=2)
        self.native.OfflineSpeakerDiarization.assert_not_called()

    def test_cache_reuse_does_not_download(self):
        # Exercise the real resolver against already cached files.
        self.model_patch.stop()
        with tempfile.TemporaryDirectory() as directory:
            for name in (self.diarizer._SEG_MODEL_REL, self.diarizer._EMB_MODEL_REL):
                path = Path(directory) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with mock.patch.object(self.diarizer, "_download_file") as download:
                self.diarizer.load(directory, num_threads=6)
            download.assert_not_called()

    def test_turns_and_speaker_assignment_are_unchanged(self):
        turns = [types.SimpleNamespace(start=0.0, end=1.0, speaker=0),
                 types.SimpleNamespace(start=1.0, end=2.0, speaker=1)]
        result = types.SimpleNamespace(sort_by_start_time=lambda: turns)
        self.diarizer._sd = types.SimpleNamespace(sample_rate=16000, process=lambda _: result)
        with mock.patch.object(self.diarizer, "_load_audio", return_value=object()):
            actual = self.diarizer.diarize("fixture.wav")
        self.assertEqual(actual, [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")])
        segments = [{"start": 0.2, "end": 0.8}, {"start": 1.2, "end": 1.8}]
        self.diarizer.assign_speakers(segments, actual)
        self.assertEqual([s["speaker"] for s in segments], ["SPEAKER_00", "SPEAKER_01"])


class ShellDiarizationOptionTests(unittest.TestCase):
    def test_shell_python_agreement_and_plain_env_export(self):
        source = (ROOT / "run.sh").read_text()
        helpers = source[source.index("normalize_decoding_option() {"):
                         source.index("# End strict diarization option helpers.")]
        script = ('exiterr() { echo "$1" >&2; exit 1; }\n' + helpers +
                  '\nunset WHISPER_DIARIZE_THREADS\n. "$1"\nconfigure_diarization_threads\n'
                  'exec "$2" -c \'import os; print(repr(os.environ["WHISPER_DIARIZE_THREADS"]))\'\n')
        cases = [(v, "2", e, True) for v, _ in VALID for e in ("true", "false")]
        cases += [(v, "2", e, False) for v in BAD for e in ("true", "false")]
        cases += [("", "6", "true", True), ("", "2147483647", "true", True),
                  ("", "2147483648", "true", False), ("", "2147483648", "false", True),
                  ("1", "2147483648", "true", True)]
        def quote(value):
            return "'" + value.replace("'", "'\"'\"'") + "'"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "whisper.env"
            for value, inherited, enabled, valid in cases:
                config.write_text(OPTION + "=" + quote(value) + "\nWHISPER_THREADS=" + quote(inherited) +
                                  "\nWHISPER_DIARIZATION=" + quote(enabled) + "\n")
                result = subprocess.run(["bash", "-c", script, "diarization-test", str(config), sys.executable],
                                        text=True, capture_output=True, timeout=10)
                with self.subTest(value=value[:60], inherited=inherited, enabled=enabled):
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    with mock.patch.dict(os.environ, {OPTION: value, "WHISPER_THREADS": inherited,
                                                       "WHISPER_DIARIZATION": enabled}):
                        with mock.patch.object(api_server, "_diarize_threads", 2):
                            if valid:
                                api_server._load_diarization_options()
                                self.assertEqual(result.stdout.strip(), repr(api_server._normalized_env_option(OPTION)))
                            else:
                                self.assertIn("WHISPER_", result.stderr)
                                with self.assertRaises(ValueError):
                                    api_server._load_diarization_options()


if __name__ == "__main__":
    unittest.main()

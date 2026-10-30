"""Speaker embedding model selection without NumPy, ONNX or downloads."""
import asyncio
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from test_api_validation import api_server


OPTION = "WHISPER_DIARIZE_EMBEDDING_MODEL"
DEFAULT = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
ENGLISH = ("wespeaker_en_voxceleb_resnet34_LM.onnx", "nemo_en_titanet_large.onnx")
BASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
VALID = [("", DEFAULT), ('""', DEFAULT), ("' \t '", DEFAULT), (DEFAULT, DEFAULT),
         (ENGLISH[0], ENGLISH[0]), (ENGLISH[1], ENGLISH[1]),
         (" \t\r\n\v\f' " + ENGLISH[0] + " '\t", ENGLISH[0]),
         ("wespeaker_en_voxceleb_CAM++.onnx", "wespeaker_en_voxceleb_CAM++.onnx")]
BAD = [
    "wespeaker_en_voxceleb_resnet34_LM", "WESPEAKER_EN_VOXCELEB_RESNET34_LM.ONNX",
    "unknown.onnx", "../" + ENGLISH[0], "/var/lib/whisper/" + ENGLISH[0],
    BASE_URL + ENGLISH[0], ENGLISH[0] + "\n" + ENGLISH[1], ENGLISH[0] + " " + ENGLISH[1],
    "'" + ENGLISH[0], ENGLISH[0] + '"', "'\"" + ENGLISH[0] + "\"'", "*", "*.onnx",
    "checksum.txt", "fangjun-sr-1.wav", "$(id)", ENGLISH[0] + " ", "x" * 10000,
]
ROOT = Path(__file__).resolve().parents[3]


def shell_list(name):
    source = (ROOT / name).read_text()
    match = re.search(r'^DIARIZE_EMBEDDING_MODELS="\\\n(.*?)"$', source, re.M | re.S)
    return tuple(match.group(1).replace("\\\n", " ").split())


def load_diarizer(native):
    spec = importlib.util.spec_from_file_location("diarizer_embedding_fixture", ROOT / "diarizer.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy"), "sherpa_onnx": native}):
        spec.loader.exec_module(module)
    return module


def native_stub():
    native = types.ModuleType("sherpa_onnx")
    for name in ("OfflineSpeakerSegmentationModelConfig",
                 "OfflineSpeakerSegmentationPyannoteModelConfig",
                 "SpeakerEmbeddingExtractorConfig", "FastClusteringConfig"):
        setattr(native, name, mock.Mock(side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs)))
    native.OfflineSpeakerDiarizationConfig = mock.Mock(
        return_value=types.SimpleNamespace(validate=mock.Mock(return_value=True)))
    native.OfflineSpeakerDiarization = mock.Mock(return_value=types.SimpleNamespace(sample_rate=16000))
    return native


class EmbeddingModelListTests(unittest.TestCase):
    def test_all_copies_agree_and_default_is_unchanged(self):
        diarizer = load_diarizer(native_stub())
        models = diarizer.EMBEDDING_MODELS
        self.assertEqual(len(models), len(set(models)))
        self.assertEqual(api_server._DIARIZE_EMBEDDING_MODELS, models)
        self.assertEqual(shell_list("run.sh"), models)
        self.assertEqual(shell_list("manage.sh"), models)
        for model in ENGLISH + (DEFAULT,):
            self.assertIn(model, models)
        self.assertEqual(diarizer._EMB_MODEL_REL, DEFAULT)
        self.assertEqual(diarizer._EMB_MODEL_URL, BASE_URL + DEFAULT)
        self.assertEqual(api_server._DEFAULT_DIARIZE_EMBEDDING_MODEL, DEFAULT)
        self.assertIn("model=" + DEFAULT + "\n", (ROOT / "run.sh").read_text())
        self.assertIn('DEFAULT_EMBEDDING_MODEL="' + DEFAULT + '"', (ROOT / "manage.sh").read_text())
        for model in models:
            self.assertRegex(model, r"\A[A-Za-z0-9][A-Za-z0-9_.+-]*\.onnx\Z")


class EmbeddingModelOptionTests(unittest.TestCase):
    def setUp(self):
        for patch in (
            mock.patch.dict(os.environ, {
                OPTION: "", "WHISPER_DIARIZE_THREADS": "", "WHISPER_THREADS": "",
                "WHISPER_DIARIZATION": "true", "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "",
                "WHISPER_NO_REPEAT_NGRAM_SIZE": "",
            }),
            mock.patch.object(api_server, "_diarize_threads", 2),
            mock.patch.object(api_server, "_diarize_embedding_model", DEFAULT),
            mock.patch.object(api_server, "_diarization_enabled", False),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_default_and_normalization(self):
        for value, expected in VALID:
            with self.subTest(value=value):
                os.environ[OPTION] = value
                api_server._load_diarization_options()
                self.assertEqual(api_server._diarize_embedding_model, expected)
        os.environ.pop(OPTION)
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_embedding_model, DEFAULT)

    def test_invalid_values_fail_in_both_modes_without_changing_state(self):
        for enabled in ("true", "false"):
            os.environ["WHISPER_DIARIZATION"] = enabled
            for value in BAD:
                with self.subTest(enabled=enabled, value=value[:60]):
                    os.environ.update({OPTION: value, "WHISPER_DIARIZE_THREADS": "6"})
                    api_server._diarize_embedding_model = ENGLISH[1]
                    api_server._diarize_threads = 3
                    with self.assertRaisesRegex(ValueError, OPTION + " must be one of: "):
                        api_server._load_diarization_options()
                    self.assertEqual(api_server._diarize_embedding_model, ENGLISH[1])
                    self.assertEqual(api_server._diarize_threads, 3)

    def test_invalid_startup_never_loads_either_model(self):
        async def start():
            async with api_server._lifespan(api_server.app):
                self.fail("Invalid config began serving")
        for enabled in ("true", "false"):
            os.environ.update({OPTION: "unknown.onnx", "WHISPER_DIARIZATION": enabled})
            with mock.patch.object(api_server, "_load_model") as whisper:
                with mock.patch.object(api_server, "_load_diarizer") as diarizer:
                    with self.assertRaisesRegex(ValueError, OPTION):
                        asyncio.run(start())
                    whisper.assert_not_called()
                    diarizer.assert_not_called()

    def test_enabled_load_propagates_model(self):
        fake = types.ModuleType("diarizer")
        fake.load = mock.Mock()
        os.environ.update({OPTION: ENGLISH[0], "WHISPER_DIARIZE_THREADS": "6",
                           "WHISPER_DIARIZE_NUM_SPEAKERS": "", "WHISPER_DIARIZE_THRESHOLD": "0.8",
                           "HF_HOME": "/fixture/cache"})
        with mock.patch.dict(sys.modules, {"diarizer": fake}):
            api_server._load_diarization_options()
            with self.assertLogs(api_server.logger, level="INFO") as logs:
                api_server._load_diarizer()
        fake.load.assert_called_once_with(cache_dir="/fixture/cache", num_speakers=-1,
                                          cluster_threshold=0.8, num_threads=6,
                                          embedding_model=ENGLISH[0])
        output = " ".join(logs.output)
        self.assertIn("Diarization enabled (num_threads=6 per model).", output)
        self.assertIn("Diarization embedding model: " + ENGLISH[0], output)

    def test_disabled_diarization_has_no_optional_imports(self):
        original = __import__
        def guarded(name, *args, **kwargs):
            if name in ("diarizer", "sherpa_onnx", "numpy"):
                self.fail("Disabled diarization imported " + name)
            return original(name, *args, **kwargs)
        os.environ.update({OPTION: ENGLISH[0], "WHISPER_DIARIZATION": "false"})
        with mock.patch("builtins.__import__", side_effect=guarded):
            api_server._load_diarization_options()
            api_server._load_diarizer()
        self.assertEqual(api_server._diarize_embedding_model, ENGLISH[0])


class DiarizerEmbeddingModelTests(unittest.TestCase):
    def setUp(self):
        self.native = native_stub()
        self.diarizer = load_diarizer(self.native)

    def test_load_uses_selected_model_and_legacy_calls_keep_default(self):
        with mock.patch.object(self.diarizer, "_ensure_models", return_value=("seg.onnx", "emb.onnx")) as models:
            self.diarizer.load("/cache", -1, 0.8, num_threads=2, embedding_model=ENGLISH[1])
            models.assert_called_with("/cache", ENGLISH[1])
            self.native.SpeakerEmbeddingExtractorConfig.assert_called_with(model="emb.onnx", num_threads=2)
            self.diarizer.load("/cache", 2, 0.5)
            models.assert_called_with("/cache", DEFAULT)

    def test_invalid_direct_values_fail_before_download_or_native_setup(self):
        for value in BAD + [None, b"nemo_en_titanet_large.onnx", 1, ("x",)]:
            with self.subTest(value=repr(value)[:60]):
                with mock.patch.object(self.diarizer, "_download_file") as download:
                    with self.assertRaisesRegex(ValueError, "embedding_model"):
                        self.diarizer.load(embedding_model=value)
                    with self.assertRaisesRegex(ValueError, "embedding_model"):
                        self.diarizer._ensure_models("/cache", value)
                download.assert_not_called()
        self.native.OfflineSpeakerDiarizationConfig.assert_not_called()

    def test_downloads_selected_model_url_into_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            segmentation = Path(directory) / self.diarizer._SEG_MODEL_REL
            segmentation.parent.mkdir(parents=True)
            segmentation.touch()
            cases = [(ENGLISH[0], BASE_URL + ENGLISH[0]),
                     ("wespeaker_en_voxceleb_CAM++.onnx", BASE_URL + "wespeaker_en_voxceleb_CAM%2B%2B.onnx"),
                     (DEFAULT, BASE_URL + DEFAULT)]
            for model, url in cases:
                with mock.patch.object(self.diarizer, "_download_file") as download:
                    seg_path, emb_path = self.diarizer._ensure_models(directory, model)
                download.assert_called_once_with(url, os.path.join(directory, model))
                self.assertEqual((seg_path, emb_path), (str(segmentation), os.path.join(directory, model)))
            # Cached files, including the unchanged default, are reused.
            (Path(directory) / ENGLISH[0]).touch()
            (Path(directory) / DEFAULT).touch()
            with mock.patch.object(self.diarizer, "_download_file") as download:
                self.diarizer._ensure_models(directory, ENGLISH[0])
                self.diarizer._ensure_models(directory)
            download.assert_not_called()


class ShellEmbeddingModelTests(unittest.TestCase):
    def test_run_sh_agrees_with_python(self):
        source = (ROOT / "run.sh").read_text()
        helpers = source[source.index("normalize_decoding_option() {"):
                         source.index("# End strict diarization option helpers.")]
        script = ('exiterr() { echo "$1" >&2; exit 1; }\n' + helpers +
                  '\nunset ' + OPTION + '\n. "$1"\nconfigure_diarization_embedding_model\n'
                  'exec "$2" -c \'import os; print(repr(os.environ["' + OPTION + '"]))\'\n')
        def quote(value):
            return "'" + value.replace("'", "'\"'\"'") + "'"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "whisper.env"
            for value, valid in [(v, True) for v, _ in VALID] + [(v, False) for v in BAD]:
                config.write_text(OPTION + "=" + quote(value) + "\n")
                result = subprocess.run(["bash", "-c", script, "embedding-test", str(config), sys.executable],
                                        text=True, capture_output=True, timeout=10)
                with self.subTest(value=value[:60]):
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    with mock.patch.dict(os.environ, {OPTION: value, "WHISPER_DIARIZE_THREADS": "1"}):
                        with mock.patch.object(api_server, "_diarize_embedding_model", DEFAULT):
                            if valid:
                                api_server._load_diarization_options()
                                self.assertEqual(result.stdout.strip(), repr(api_server._diarize_embedding_model))
                            else:
                                self.assertIn(OPTION + " must be one of: " + DEFAULT.split("_")[0], result.stderr)
                                with self.assertRaises(ValueError):
                                    api_server._load_diarization_options()

    def test_manage_sh_downloads_the_saved_model(self):
        source = (ROOT / "manage.sh").read_text()
        function = source[source.index("do_download_diarize() {"):source.index("\ncheck_container\n")]
        header = source[source.index("EMBEDDING_MODEL_FILE="):source.index("\nexiterr() {")]
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / ".diarize_embedding_model"
            log = Path(directory) / "curl.log"
            # Stub curl, tar and rm so nothing is downloaded or removed.
            script = ('exiterr() { echo "Error: $1" >&2; exit 1; }\n'
                      'curl() { printf "%s\\n" "${@: -1}" >> "$CURL_LOG"; }\n'
                      'tar() { :; }\nrm() { :; }\n' + header.replace(
                          '"${WHISPER_DATA}/.diarize_embedding_model"', '"$STATE_FILE"') +
                      '\n' + function + '\ndo_download_diarize\n')
            cases = [(None, BASE_URL + DEFAULT), ("", BASE_URL + DEFAULT),
                     (ENGLISH[0], BASE_URL + ENGLISH[0]),
                     ("wespeaker_en_voxceleb_CAM++.onnx", BASE_URL + "wespeaker_en_voxceleb_CAM%2B%2B.onnx")]
            for saved, url in cases:
                if state.exists():
                    state.unlink()
                if saved is not None:
                    state.write_text(saved)
                if log.exists():
                    log.unlink()
                env = dict(os.environ, STATE_FILE=str(state), CURL_LOG=str(log))
                result = subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                                        timeout=10, env=env)
                with self.subTest(saved=saved):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(url, log.read_text().splitlines())
            state.write_text("../unknown.onnx")
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=10,
                                    env=dict(os.environ, STATE_FILE=str(state), CURL_LOG=str(log)))
            self.assertEqual(result.returncode, 1)
            self.assertIn("Unknown speaker embedding model", result.stderr)


if __name__ == "__main__":
    unittest.main()

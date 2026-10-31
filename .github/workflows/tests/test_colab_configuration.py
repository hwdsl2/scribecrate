"""Check the notebook's runtime-file downloads and configuration bridge offline."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks/colab_demo.ipynb"


class ColabConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text())
        cls.cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}

    def configuration(self):
        namespace = {}
        with mock.patch("subprocess.run", return_value=SimpleNamespace(returncode=1)), \
             redirect_stdout(io.StringIO()):
            exec(compile(self.cells["config"], "colab-config", "exec"), namespace)
        return namespace

    def server_environment(self, namespace):
        # Stop before CUDA setup, filesystem writes and server/network activity.
        source = self.cells["start-server"]
        self.assertIn("# CTranslate2 needs", source)
        prefix = source.split("# CTranslate2 needs", 1)[0]
        with mock.patch.dict(os.environ, {
            "WHISPER_DIARIZE_EMBEDDING_MODEL": "stale-value.onnx",
            "WHISPER_DIARIZE_THRESHOLD": "1.2",
        }, clear=True):
            exec(compile(prefix, "colab-server-environment", "exec"), namespace)
        return namespace["server_env"]

    def test_existing_defaults_and_cpu_fallback(self):
        namespace = self.configuration()
        self.assertFalse(namespace["has_gpu"])
        self.assertEqual(namespace["WHISPER_COMPUTE_TYPE"], "int8")
        env = self.server_environment(namespace)
        self.assertEqual(env["WHISPER_MODEL"], "small")
        self.assertEqual(env["WHISPER_DIARIZATION"], "true")
        self.assertEqual(env["WHISPER_DIARIZE_NUM_SPEAKERS"], "2")
        self.assertEqual(env["WHISPER_DIARIZE_EMBEDDING_MODEL"], "")
        self.assertEqual(env["WHISPER_DIARIZE_THRESHOLD"], "0.5")

    def test_embedding_choices_cross_the_server_environment(self):
        namespace = self.configuration()
        for name in ("", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                     "wespeaker_en_voxceleb_resnet34_LM.onnx", "nemo_en_titanet_large.onnx",
                     "wespeaker_en_voxceleb_CAM++.onnx"):
            with self.subTest(name=name):
                namespace["WHISPER_DIARIZE_EMBEDDING_MODEL"] = name
                self.assertEqual(self.server_environment(namespace)["WHISPER_DIARIZE_EMBEDDING_MODEL"], name)

    def test_automatic_count_and_threshold_cross_the_server_environment(self):
        namespace = self.configuration()
        namespace["WHISPER_DIARIZE_NUM_SPEAKERS"] = ""
        namespace["WHISPER_DIARIZE_THRESHOLD"] = "0.8"
        env = self.server_environment(namespace)
        self.assertEqual(env["WHISPER_DIARIZE_NUM_SPEAKERS"], "")
        self.assertEqual(env["WHISPER_DIARIZE_THRESHOLD"], "0.8")

    def test_download_cell_covers_local_runtime_dependencies(self):
        with mock.patch("urllib.request.urlretrieve") as retrieve, redirect_stdout(io.StringIO()):
            exec(compile(self.cells["download-server"], "colab-download", "exec"), {})
        downloaded = set()
        for call in retrieve.call_args_list:
            url, destination = call.args
            filename = Path(destination).name
            self.assertEqual(Path(destination).parent, Path("/content"))
            self.assertEqual(url, "https://raw.githubusercontent.com/hwdsl2/scribecrate/main/" + filename)
            downloaded.add(filename)

        # Follow repository-local imports without importing optional native libs.
        required, pending = set(), {"api_server.py"}
        while pending:
            filename = pending.pop()
            required.add(filename)
            for node in ast.walk(ast.parse((ROOT / filename).read_text())):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for module in modules:
                    dependency = module.split(".")[0] + ".py"
                    if (ROOT / dependency).is_file() and dependency not in required:
                        pending.add(dependency)
        self.assertTrue(required <= downloaded, "Notebook omitted runtime files: " + str(required - downloaded))

    def test_startup_wait_handles_slow_downloads_exit_and_deadline(self):
        source = self.cells["start-server"].split(
            "# Wait for startup using the same download budget as the container.", 1)[1]
        for case, enabled, offline, expected_ready, expected_time in (
                ("delayed", "true", "", True, 700),
                ("dead", "true", "", False, 0),
                ("unavailable", "", "", False, 300),
                ("unavailable", "true", "true", False, 300)):
            with self.subTest(case=case, enabled=enabled, offline=offline):
                clock = [0]
                def sleep(seconds):
                    clock[0] += seconds
                response = mock.MagicMock()
                response.__enter__.return_value.read.return_value = b'{"model":"fixture"}'
                def request(*args, **kwargs):
                    self.assertLessEqual(kwargs["timeout"], 3)
                    if case == "delayed" and clock[0] >= 700:
                        return response
                    raise urllib.error.URLError("fixture not ready")
                poll = mock.Mock(return_value=1 if case == "dead" else None)
                namespace = {
                    "server_env": {"WHISPER_DIARIZATION": enabled, "WHISPER_LOCAL_ONLY": offline},
                    "WHISPER_PORT": 9000,
                    "server_proc": SimpleNamespace(pid=123, poll=poll),
                    "time": SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep),
                    "urllib": SimpleNamespace(request=SimpleNamespace(urlopen=mock.Mock(side_effect=request))),
                    "json": json,
                }
                with mock.patch("builtins.open", mock.mock_open(read_data="fixture startup log")), \
                        redirect_stdout(io.StringIO()) as output:
                    exec(compile(source, "colab-startup-wait", "exec"), namespace)
                self.assertIs(namespace["ready"], expected_ready)
                self.assertEqual(clock[0], expected_time)
                if case == "dead":
                    namespace["urllib"].request.urlopen.assert_not_called()
                    self.assertIn("Server exited with code 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()

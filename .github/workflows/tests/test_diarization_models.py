"""Verified ONNX downloads, safe caches, offline behavior and explicit prefetch."""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import unquote

from test_api_validation import api_server
import diarization_models as models

ROOT = Path(__file__).resolve().parents[3]
NAMES = tuple(models.EMBEDDING_MODELS)
DEFAULT = models.DEFAULT_EMBEDDING_MODEL


class ModelMetadataTests(unittest.TestCase):
    def test_full_allowlist_has_pinned_assets_and_unchanged_default(self):
        self.assertEqual(len(NAMES), 21)
        self.assertEqual(DEFAULT, "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
        self.assertEqual(models.resolve_embedding_model(), DEFAULT)
        for name, asset in models.EMBEDDING_MODELS.items():
            with self.subTest(name=name):
                self.assertEqual(models.resolve_embedding_model(name), name)
                self.assertGreater(asset.size, 0)
                self.assertRegex(asset.sha256, r"\A[0-9a-f]{64}\Z")

    def test_invalid_selection_fails_before_creating_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "absent"
            with mock.patch.object(models.subprocess, "run") as download:
                for name in ("../model.onnx", "https://example.com/x.onnx", "unknown.onnx", None):
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        models.ensure_models(str(cache), name)
            download.assert_not_called()
            self.assertFalse(cache.exists())


class AutomaticOfflineTests(unittest.TestCase):
    def test_api_preserves_existing_trimmed_nonempty_local_only_semantics(self):
        for value, expected in (("", False), (" \t", False), ("false", True), ("0", True), ("true", True)):
            fake = SimpleNamespace(load=mock.Mock())
            with self.subTest(value=value), mock.patch.dict(os.environ, {
                "WHISPER_DIARIZATION": "true", "WHISPER_LOCAL_ONLY": value,
            }), mock.patch.dict(sys.modules, {"diarizer": fake}), \
                    mock.patch.object(api_server, "_diarization_enabled"):
                api_server._load_diarizer()
            self.assertIs(fake.load.call_args.kwargs["local_files_only"], expected)

    def test_missing_offline_models_warn_without_losing_loaded_whisper(self):
        with tempfile.TemporaryDirectory() as directory:
            def load(**kwargs):
                models.ensure_models(kwargs["cache_dir"], kwargs["embedding_model"],
                                     local_files_only=kwargs["local_files_only"])
            fake = SimpleNamespace(load=load)
            whisper = object()
            with mock.patch.dict(os.environ, {
                "WHISPER_DIARIZATION": "true", "WHISPER_LOCAL_ONLY": "1", "HF_HOME": directory,
            }), mock.patch.dict(sys.modules, {"diarizer": fake}), \
                    mock.patch.object(api_server, "_model", whisper), \
                    mock.patch.object(api_server, "_diarization_enabled", False), \
                    mock.patch.object(models.subprocess, "run") as download:
                with self.assertLogs(api_server.logger, level="WARNING") as logs:
                    api_server._load_diarizer()
                self.assertFalse(api_server._diarization_enabled)
                self.assertIs(api_server._model, whisper)
                self.assertIn("WHISPER_LOCAL_ONLY", " ".join(logs.output))
                download.assert_not_called()


class StartupBudgetTests(unittest.TestCase):
    def test_online_budget_covers_both_downloads_and_preserves_other_startups(self):
        payload = b"download budget fixture"
        asset = models.ModelAsset(len(payload), hashlib.sha256(payload).hexdigest())
        def download(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(payload)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(models.subprocess, "run", side_effect=download) as curl:
            models._download_file("https://example.invalid/model", Path(directory) / "model", asset)
        process_timeout = curl.call_args.kwargs["timeout"]
        command = curl.call_args.args[0]
        attempt_timeout = int(command[command.index("--max-time") + 1])
        attempts = 1 + int(command[command.index("--retry") + 1])
        self.assertGreater(process_timeout, attempt_timeout * attempts)
        for enabled, offline in ((False, False), (False, True), (True, True)):
            self.assertEqual(models.startup_timeout_seconds(
                diarization_enabled=enabled, local_files_only=offline), 300)
        self.assertGreater(models.startup_timeout_seconds(
            diarization_enabled=True, local_files_only=False), 300 + 2 * process_timeout)

    def test_container_wait_handles_slow_startup_exit_and_timeout(self):
        source = (ROOT / "run.sh").read_text()
        bridge = source[source.index("startup_timeout=$(python3 - <<'PYEOF'"):
                        source.index("\ncd /opt/src && python3 /opt/src/api_server.py &")]
        bridge = bridge.replace('"/opt/src"', repr(str(ROOT)))
        function = source[source.index("wait_for_server() {"):
                          source.index("\nif ! wait_for_server; then")]
        script = ('exiterr() { echo "$1" >&2; exit 1; }\n' + bridge + '\n' + function + '''
WHISPER_PID=fixture
WHISPER_PORT=9000
calls=0
unset SECONDS
SECONDS=0
kill() { [ "$TEST_CASE" != dead ]; }
curl() {
  calls=$((calls + 1))
  [[ "$*" == *"--max-time 2"* ]] || exit 99
  [ "$TEST_CASE" = ready ] && [ "$calls" -ge 350 ]
}
sleep() { SECONDS=$((SECONDS + 1)); }
[ -z "${TEST_LIMIT:-}" ] || startup_timeout="$TEST_LIMIT"
if wait_for_server; then result=0; else result=1; fi
printf '%s\\n' "$calls"
exit "$result"
''')
        cases = [("true", "", "ready", "", 0, 350),
                 ("", "", "ready", "", 1, 300),
                 ("true", "true", "ready", "", 1, 300),
                 ("true", "", "dead", "", 1, 0),
                 ("true", "", "unavailable", "3", 1, 3)]
        for enabled, offline, case, limit, expected_status, calls in cases:
            with self.subTest(enabled=enabled, offline=offline, case=case):
                result = subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                                        timeout=10, env=dict(os.environ, WHISPER_DIARIZATION=enabled,
                                            WHISPER_LOCAL_ONLY=offline, TEST_CASE=case, TEST_LIMIT=limit,
                                            PYTHONDONTWRITEBYTECODE="1"))
                self.assertEqual(result.returncode, expected_status, result.stderr)
                self.assertEqual(result.stdout.strip(), str(calls))


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.seg = b"segmentation fixture"
        self.payloads = {name: name.encode() for name in NAMES}
        self.assets = {name: self.asset(data) for name, data in self.payloads.items()}
        for patch in (mock.patch.object(models, "EMBEDDING_MODELS", self.assets),
                      mock.patch.object(models, "_SEGMENTATION_ASSET", self.asset(self.seg))):
            patch.start()
            self.addCleanup(patch.stop)
        seg_path = self.root / models.SEGMENTATION_MODEL
        seg_path.parent.mkdir()
        seg_path.write_bytes(self.seg)

    @staticmethod
    def asset(data):
        return models.ModelAsset(len(data), hashlib.sha256(data).hexdigest())

    def fake_curl(self, command, **kwargs):
        dest = Path(command[command.index("--output") + 1])
        dest.write_bytes(self.payloads[unquote(command[-1].split("/")[-1])])

    def test_all_models_download_once_and_switch_back_offline(self):
        # Explicit prefetch remains permitted even when automatic startup is offline.
        with mock.patch.dict(os.environ, {"WHISPER_LOCAL_ONLY": "1"}), \
                mock.patch.object(models.subprocess, "run", side_effect=self.fake_curl) as curl:
            for name in NAMES:
                _, emb = models.ensure_models(str(self.root), name)
                self.assertEqual(Path(emb).read_bytes(), self.payloads[name])
            self.assertEqual(curl.call_count, len(NAMES))
            for name in (*NAMES, ""):
                models.ensure_models(str(self.root), name, local_files_only=True)
            self.assertEqual(curl.call_count, len(NAMES))
            command = curl.call_args.args[0]
            self.assertEqual(command[command.index("--proto-redir") + 1], "=https")
            self.assertEqual(command[command.index("--max-filesize") + 1], str(self.assets[NAMES[-1]].size))
        self.assertFalse(list(self.root.rglob(".diarization-*")))

    def test_missing_offline_never_downloads(self):
        with mock.patch.object(models.subprocess, "run") as curl:
            with self.assertRaisesRegex(RuntimeError, "WHISPER_LOCAL_ONLY"):
                models.ensure_models(str(self.root), NAMES[1], local_files_only=True)
            curl.assert_not_called()

    def test_cam_plus_url_is_encoded_and_only_selected_embedding_is_fetched(self):
        name = "wespeaker_en_voxceleb_CAM++.onnx"
        with mock.patch.object(models.subprocess, "run", side_effect=self.fake_curl) as curl:
            models.ensure_models(str(self.root), name)
        curl.assert_called_once()
        self.assertTrue(curl.call_args.args[0][-1].endswith("/wespeaker_en_voxceleb_CAM%2B%2B.onnx"))
        self.assertEqual({p.name for p in self.root.glob("*.onnx")}, {name})

    def test_recovery_preserves_bad_file_and_reuses_replacement_offline(self):
        path = self.root / DEFAULT
        path.write_bytes(b"incorrect cached weights")
        with mock.patch.object(models.subprocess, "run") as curl:
            with self.assertRaisesRegex(RuntimeError, "Move this file aside"):
                models.ensure_models(str(self.root))
            curl.assert_not_called()
        preserved = path.with_suffix(".onnx.invalid")
        path.rename(preserved)
        with mock.patch.object(models.subprocess, "run", side_effect=self.fake_curl) as curl:
            models.ensure_models(str(self.root))
            self.assertEqual(curl.call_count, 1)
            models.ensure_models(str(self.root), local_files_only=True)
            self.assertEqual(curl.call_count, 1)
        self.assertEqual(preserved.read_bytes(), b"incorrect cached weights")

    def test_segmentation_directory_symlink_is_reused_offline(self):
        path = self.root / models.SEGMENTATION_MODEL
        path.unlink()
        path.parent.rmdir()
        (self.root / DEFAULT).write_bytes(self.payloads[DEFAULT])
        with tempfile.TemporaryDirectory() as shared:
            target = Path(shared) / "model.onnx"
            target.write_bytes(self.seg)
            path.parent.symlink_to(shared, target_is_directory=True)
            with mock.patch.object(models.subprocess, "run") as curl:
                result = models.ensure_models(str(self.root), local_files_only=True)
                curl.assert_not_called()
            self.assertEqual(result, (str(path), str(self.root / DEFAULT)))
            self.assertTrue(path.parent.is_symlink())
            self.assertEqual(target.read_bytes(), self.seg)

    def test_regular_file_symlink_targets_are_verified_and_reused_offline(self):
        seg_path = self.root / models.SEGMENTATION_MODEL
        emb_path = self.root / DEFAULT
        seg_path.unlink()
        with tempfile.TemporaryDirectory() as shared:
            targets = ((seg_path, Path(shared) / "segmentation.onnx", self.seg),
                       (emb_path, Path(shared) / "embedding.onnx", self.payloads[DEFAULT]))
            for _, target, data in targets:
                target.write_bytes(data)
                target.chmod(0o444)
            for relative, chained in ((False, False), (True, False), (False, True), (True, True)):
                for link, target, _ in targets:
                    destination = target
                    if chained:
                        destination = target.with_suffix(".link")
                        destination.symlink_to(target.name)
                    link.symlink_to(os.path.relpath(destination, link.parent) if relative else destination)
                with self.subTest(relative=relative, chained=chained), \
                        mock.patch.object(models.subprocess, "run") as curl:
                    result = models.ensure_models(str(self.root), local_files_only=True)
                    curl.assert_not_called()
                self.assertEqual(result, (str(seg_path), str(emb_path)))
                for link, target, data in targets:
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(target.read_bytes(), data)
                    self.assertEqual(target.stat().st_mode & 0o777, 0o444)
                    link.unlink()
                    if chained:
                        target.with_suffix(".link").unlink()

    def test_missing_segmentation_download_is_published_through_directory_link(self):
        path = self.root / models.SEGMENTATION_MODEL
        path.unlink()
        path.parent.rmdir()
        (self.root / DEFAULT).write_bytes(self.payloads[DEFAULT])
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
            for name, data in ((models.SEGMENTATION_MODEL, self.seg),
                               (str(Path(models.SEGMENTATION_MODEL).parent / "LICENSE"), b"license")):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        archive = buffer.getvalue()
        def download(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(archive)
        with tempfile.TemporaryDirectory() as shared:
            path.parent.symlink_to(shared, target_is_directory=True)
            with mock.patch.object(models, "_SEGMENTATION_DOWNLOAD", self.asset(archive)), \
                    mock.patch.object(models.subprocess, "run", side_effect=download) as curl:
                models.ensure_models(str(self.root))
                self.assertEqual(curl.call_count, 1)
                models.ensure_models(str(self.root), local_files_only=True)
                self.assertEqual(curl.call_count, 1)
            self.assertTrue(path.parent.is_symlink())
            self.assertEqual((Path(shared) / "model.onnx").read_bytes(), self.seg)
            self.assertEqual((Path(shared) / "LICENSE").read_bytes(), b"license")
            self.assertEqual({p.name for p in Path(shared).iterdir()}, {"model.onnx", "LICENSE"})

    def test_invalid_file_symlinks_are_preserved_without_download(self):
        path = self.root / DEFAULT
        with tempfile.TemporaryDirectory() as shared:
            for kind in ("missing", "wrong-size", "wrong-hash", "directory", "loop"):
                target = Path(shared) / kind
                data = b"short" if kind == "wrong-size" else b"x" * self.assets[DEFAULT].size
                if kind in ("wrong-size", "wrong-hash"):
                    target.write_bytes(data)
                elif kind == "directory":
                    target.mkdir()
                elif kind == "loop":
                    target = path
                path.symlink_to(target)
                original = os.readlink(path)
                for offline in (False, True):
                    with self.subTest(kind=kind, offline=offline), \
                            mock.patch.object(models.subprocess, "run") as curl:
                        with self.assertRaises(RuntimeError):
                            models.ensure_models(str(self.root), local_files_only=offline)
                        curl.assert_not_called()
                    self.assertTrue(path.is_symlink())
                    self.assertEqual(os.readlink(path), original)
                    if kind in ("wrong-size", "wrong-hash"):
                        self.assertEqual(target.read_bytes(), data)
                    elif kind == "missing":
                        self.assertFalse(target.exists())
                path.unlink()

    def test_invalid_segmentation_directory_links_are_preserved(self):
        path = self.root / models.SEGMENTATION_MODEL
        path.unlink()
        path.parent.rmdir()
        for kind in ("missing", "file", "loop"):
            target = self.root / kind
            if kind == "file":
                target.write_bytes(b"preserve this file")
            elif kind == "loop":
                target = path.parent
            path.parent.symlink_to(target, target_is_directory=True)
            original = os.readlink(path.parent)
            with self.subTest(kind=kind), mock.patch.object(models.subprocess, "run") as curl:
                with self.assertRaises((RuntimeError, OSError)):
                    models.ensure_models(str(self.root))
                curl.assert_not_called()
            self.assertTrue(path.parent.is_symlink())
            self.assertEqual(os.readlink(path.parent), original)
            if kind == "file":
                self.assertEqual(target.read_bytes(), b"preserve this file")
            path.parent.unlink()

    def test_repaired_model_link_is_reused_without_replacing_link(self):
        link = self.root / DEFAULT
        with tempfile.TemporaryDirectory() as shared:
            target = Path(shared) / "embedding.onnx"
            target.write_bytes(b"x" * self.assets[DEFAULT].size)
            link.symlink_to(target)
            with mock.patch.object(models.subprocess, "run") as curl:
                with self.assertRaises(RuntimeError):
                    models.ensure_models(str(self.root))
                target.write_bytes(self.payloads[DEFAULT])
                models.ensure_models(str(self.root), local_files_only=True)
                curl.assert_not_called()
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.read_bytes(), self.payloads[DEFAULT])

    def test_fifo_and_fifo_symlink_are_rejected_without_blocking(self):
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        link = self.root / DEFAULT
        link.symlink_to(fifo)
        script = '''import hashlib, sys
from pathlib import Path
from diarization_models import _verified, ModelAsset
try:
    _verified(Path(sys.argv[1]), ModelAsset(0, hashlib.sha256(b"").hexdigest()))
except RuntimeError:
    print("Rejected non-regular target")
else:
    raise AssertionError("Accepted a FIFO")
'''
        for path in (fifo, link):
            with self.subTest(path=path):
                result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=ROOT,
                                        text=True, capture_output=True, timeout=5,
                                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Rejected non-regular target", result.stdout)
        self.assertTrue(link.is_symlink())
        self.assertTrue(fifo.exists())

    def test_corrupt_regular_cache_is_preserved_without_download(self):
        path = self.root / DEFAULT
        for data in (b"partial", b"x" * self.assets[DEFAULT].size):
            path.write_bytes(data)
            with mock.patch.object(models.subprocess, "run") as curl:
                with self.assertRaisesRegex(RuntimeError, "verification"):
                    models.ensure_models(str(self.root))
                curl.assert_not_called()
            self.assertEqual(path.read_bytes(), data)

    def test_failed_or_bad_download_never_publishes_or_leaves_temp_files(self):
        for failure in (subprocess.CalledProcessError(22, "curl"), subprocess.TimeoutExpired("curl", 1), "short", "wrong-hash"):
            def fail(command, **kwargs):
                self.fake_curl(command, **kwargs)
                if isinstance(failure, Exception):
                    raise failure
                dest = Path(command[command.index("--output") + 1])
                dest.write_bytes(b"short" if failure == "short" else b"x" * self.assets[DEFAULT].size)
            with mock.patch.object(models.subprocess, "run", side_effect=fail):
                with self.assertRaises((RuntimeError, subprocess.SubprocessError)):
                    models.ensure_models(str(self.root))
            self.assertFalse((self.root / DEFAULT).exists())
            self.assertFalse(list(self.root.rglob(".diarization-*")))

    def test_segmentation_extracts_only_exact_regular_model(self):
        seg_path = self.root / models.SEGMENTATION_MODEL
        seg_path.unlink()
        (self.root / DEFAULT).write_bytes(self.payloads[DEFAULT])
        for linked_member in (None, "model", "license"):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
                extra = tarfile.TarInfo("../../unexpected")
                extra.size = 4
                tar.addfile(extra, io.BytesIO(b"oops"))
                model = tarfile.TarInfo(models.SEGMENTATION_MODEL)
                if linked_member == "model":
                    model.type = tarfile.SYMTYPE
                    model.linkname = "../../unexpected"
                    tar.addfile(model)
                else:
                    model.size = len(self.seg)
                    tar.addfile(model, io.BytesIO(self.seg))
                license_member = tarfile.TarInfo(str(Path(models.SEGMENTATION_MODEL).parent / "LICENSE"))
                if linked_member == "license":
                    license_member.type = tarfile.SYMTYPE
                    license_member.linkname = "../../unexpected"
                    tar.addfile(license_member)
                else:
                    license_member.size = 7
                    tar.addfile(license_member, io.BytesIO(b"license"))
            archive = buf.getvalue()
            def download(command, **kwargs):
                Path(command[command.index("--output") + 1]).write_bytes(archive)
            with mock.patch.object(models, "_SEGMENTATION_DOWNLOAD", self.asset(archive)), \
                 mock.patch.object(models.subprocess, "run", side_effect=download):
                if linked_member:
                    with self.assertRaisesRegex(RuntimeError, "archive " + linked_member):
                        models.ensure_models(str(self.root))
                else:
                    models.ensure_models(str(self.root))
                    self.assertEqual(seg_path.read_bytes(), self.seg)
                    self.assertEqual((seg_path.parent / "LICENSE").read_bytes(), b"license")
                    seg_path.unlink()
            self.assertFalse(list(self.root.rglob(".diarization-*")))
            self.assertEqual(sorted(p.name for p in self.root.iterdir()), [DEFAULT, seg_path.parent.name])

    def test_publication_failure_cleans_verified_temporary_download(self):
        with mock.patch.object(models.subprocess, "run", side_effect=self.fake_curl), \
                mock.patch.object(models.os, "replace", side_effect=PermissionError("fixture failure")):
            with self.assertRaises(PermissionError):
                models.ensure_models(str(self.root))
        self.assertFalse((self.root / DEFAULT).exists())
        self.assertFalse(list(self.root.rglob(".diarization-*")))


class ManagementDownloadTests(unittest.TestCase):
    def test_whisper_prefetch_retains_offline_check_and_propagates_failure(self):
        source = (ROOT / "manage.sh").read_text()
        function = source[source.index("do_download_model() {"):
                          source.index("do_download_diarize() {")]
        script = ('exiterr() { echo "$1" >&2; exit 1; }\n' + function + '''
model_to_download=tiny
python3() { echo "fixture download attempted"; return 1; }
do_download_model
''')
        for offline in ("", "true"):
            with self.subTest(offline=offline):
                result = subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                                        timeout=10, env=dict(os.environ, WHISPER_LOCAL_ONLY=offline))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn("To activate this model", result.stdout)
                if offline:
                    self.assertNotIn("fixture download attempted", result.stdout)
                    self.assertIn("model downloads are disabled", result.stderr)
                else:
                    self.assertIn("Whisper model download failed", result.stderr)

    def test_invalid_explicit_prefetch_fails_without_changing_saved_selection(self):
        source = (ROOT / "manage.sh").read_text()
        function = source[source.index("do_download_diarize() {"):source.index("\ncheck_container\n")]
        function = function.replace("/opt/src/diarization_models.py", str(ROOT / "diarization_models.py"))
        script = ('exiterr() { echo "$1" >&2; exit 1; }\n' + function +
                  '\nWHISPER_DATA="$1"\nEMBEDDING_MODEL_FILE="$1/.diarize_embedding_model"\n'
                  'embedding_to_download="../bad.onnx"\ndo_download_diarize\n')
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / ".diarize_embedding_model"
            saved.write_text(DEFAULT)
            result = subprocess.run(["bash", "-c", script, "prefetch-test", directory],
                                    text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("WHISPER_DIARIZE_EMBEDDING_MODEL must be one of:", result.stderr)
            self.assertEqual(saved.read_text(), DEFAULT)
            self.assertEqual(list(Path(directory).iterdir()), [saved])

    def test_default_saved_selection_and_explicit_prefetch(self):
        source = (ROOT / "manage.sh").read_text()
        parser = source[source.index("parse_args() {"):source.index("check_args() {")]
        download = source[source.index("do_download_diarize() {"):source.index("\ncheck_container\n")]
        script = ('exiterr() { echo "$1" >&2; exit 1; }\n'
                  'show_usage() { exit "${2:-1}"; }\n' + parser + download +
                  '\nWHISPER_DATA="$1"\nEMBEDDING_MODEL_FILE="$1/.diarize_embedding_model"\nshift\n'
                  'python3() { printf "ARG=<%s>\\n" "$@"; }\n'
                  'parse_args "$@"\ndo_download_diarize\n')
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / ".diarize_embedding_model"
            cases = [(None, [], ""), (NAMES[1], [], NAMES[1]),
                     (NAMES[1], [NAMES[2]], NAMES[2])]
            for active, arguments, expected in cases:
                if active:
                    saved.write_text(active)
                result = subprocess.run(["bash", "-c", script, "management-test", directory,
                                         "--downloaddiarize", *arguments],
                                        capture_output=True, text=True, timeout=10,
                                        env=dict(os.environ, WHISPER_LOCAL_ONLY="true"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("ARG=<--embedding-model=" + expected + ">", result.stdout)
                if active:
                    self.assertEqual(saved.read_text(), active, "Prefetch changed active selection")


if __name__ == "__main__":
    unittest.main()

"""Secret input contract and isolated launcher/helper regressions, without models.

Only launcher setup fragments execute, with every writable absolute path mapped
to a disposable fixture. ROLLOUT_SOURCE_REF can compare a pre-change revision.
"""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
CONFIG = {
    "docker-whisper": ("WHISPER_API_KEY", "whisper", "whisper.env"),
    "docker-docling": ("DOCLING_API_KEY", "docling", "docling.env"),
    "docker-kokoro": ("KOKORO_API_KEY", "kokoro", "kokoro.env"),
    "docker-whisper-live": ("WHISPERLIVE_API_KEY", "whisper-live", "whisper-live.env"),
    "docker-embeddings": ("EMBED_API_KEY", "embeddings", "embed.env"),
    "docker-ollama": ("OLLAMA_API_KEY", "ollama", "ollama.env"),
    "docker-mcp-gateway": ("MCP_API_KEY", "mcp", "mcp.env"),
    "docker-litellm": ("LITELLM_MASTER_KEY", "litellm", "litellm.env"),
}
# Detect variable names so arbitrary checkout directory names work too.
KEY, DATA, ENV = next(values for values in CONFIG.values()
                      if re.search(r"\b" + values[0] + r"\b", (ROOT / "run.sh").read_text()))
INTEGRATED = DATA in ("ollama", "mcp", "litellm")


def source(name):
    ref = os.environ.get("ROLLOUT_SOURCE_REF")
    if ref:
        return subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT, text=True)
    return (ROOT / name).read_text()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / "data"
        self.data.mkdir()
        (self.base / "run" / (DATA + "-secrets")).mkdir(parents=True, mode=0o700)
        self.envfile = self.base / ENV
        self.envfile.touch()
        self.mounts = self.base / "mounts"
        self.mounts.write_text(f"fixture {self.data} fixture rw 0 0\n")
        self.saved = self.data / (".master_key" if DATA == "litellm" else ".api_key")
        self.external = self.base / "secret with spaces"
        self.external.write_bytes(b"synthetic-file-key\r\n\n")

    def remap(self, script):
        script = script.replace(f"/var/lib/{DATA}", str(self.data))
        if DATA == "litellm":
            script = script.replace("/etc/litellm", str(self.data))
        script = script.replace(f"/{ENV}", str(self.envfile))
        script = script.replace("/proc/mounts", str(self.mounts))
        script = script.replace("/run/", str(self.base / "run") + "/")
        return script

    def launch(self, values=None, extra_keys=()):
        script = source("run.sh")
        if INTEGRATED:
            prefix = script.split("# Apply defaults", 1)[0]
            if DATA == "litellm":
                block = script.split("# Generate or load master key", 1)[1].split(
                    "# Export master key", 1)[0]
                active = "master_key"
            else:
                block = script.split("# Generate or load API key", 1)[1].split(
                    "# Save port" if DATA == "ollama" else "# Determine server address", 1)[0]
                active = "api_key"
            script = prefix + '\nAPI_KEY_FILE="$TEST_SAVED"\nMASTER_KEY_FILE="$TEST_SAVED"\n'
            if DATA == "mcp":
                if not shutil.which("node"):
                    self.skipTest("Node required for real UplinkCrate saved-key validator")
                script += 'POLICY_CLI="/app/dist/uplinkPolicy.js"\n'
                self.skipTest("Requires built UplinkCrate policy; covered by policy/container tests")
            script += block + f'\nprintf "%s" "${{{active}}}"\n'
        else:
            script = script.split("# Determine server address", 1)[0]
            script += f'\nprintf "%s" "${{{KEY}}}"\n'
        if extra_keys:
            script += "\nprintf '\\0'\n"
            for variable in extra_keys:
                script += "printf '%s\\0' \"${" + variable + "}\"\n"
        env = {"PATH": os.environ["PATH"], "KUBERNETES_SERVICE_HOST": "fixture",
               "TEST_SAVED": str(self.saved)}
        env.update(values or {})
        return subprocess.run(["bash", "-c", self.remap(script)], env=env,
                              capture_output=True, timeout=15)

    def test_legacy_fresh_saved_empty_direct_and_normalization(self):
        first = self.launch()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(first.stdout)
        self.assertEqual(self.saved.read_bytes(), first.stdout)
        self.assertEqual(self.launch().stdout, first.stdout)
        self.assertEqual(self.launch({KEY: ""}).stdout, first.stdout if INTEGRATED else b"")
        direct = self.launch({KEY: '  "synthetic-direct-key"  '})
        self.assertEqual(direct.stdout, b"synthetic-direct-key")
        self.assertEqual(self.saved.read_bytes(), direct.stdout)

    def test_legacy_existing_noauth_unmounted_and_env_precedence(self):
        (self.data / "existing-model").touch()
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(bool(result.stdout), INTEGRATED)
        if self.saved.exists():
            self.saved.unlink()
        self.mounts.write_text("")
        result = self.launch()
        self.assertEqual(bool(result.stdout), INTEGRATED)
        self.envfile.write_text(f"{KEY}=synthetic-mounted-key\n")
        self.assertEqual(self.launch({KEY: "synthetic-container-key"}).stdout,
                         b"synthetic-mounted-key")

    def test_file_overrides_saved_without_changing_external_permissions(self):
        if os.getenv("ROLLOUT_SOURCE_REF"):
            self.skipTest("File-input candidate only")
        self.saved.write_bytes(b"synthetic-saved-key")
        self.external.chmod(0o400)
        result = self.launch({KEY + "_FILE": str(self.external)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"synthetic-file-key")
        self.assertEqual(self.saved.read_bytes(), result.stdout if INTEGRATED else b"synthetic-saved-key")
        self.assertEqual(self.external.stat().st_mode & 0o777, 0o400)

    def test_active_helper_rotation_and_legacy_rollback(self):
        if INTEGRATED or os.getenv("ROLLOUT_SOURCE_REF"):
            self.skipTest("Private-snapshot candidate only")
        self.saved.write_bytes(b"synthetic-saved-key")
        values = {KEY + "_FILE": str(self.external)}
        self.assertEqual(self.launch(values).returncode, 0)
        snapshot = self.base / "run" / (DATA + "-secrets") / "api_key"
        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o600)
        self.assertEqual(snapshot.parent.stat().st_mode & 0o777, 0o700)
        def helper(action="--getkey"):
            return subprocess.run(["bash", "-c", self.remap(source("manage.sh")), "helper", action],
                                  env={"PATH": os.environ["PATH"], "KUBERNETES_SERVICE_HOST": "fixture"},
                                  capture_output=True, timeout=5)
        # Path exists only in the mounted env file; exec inherits neither input.
        self.envfile.write_text(KEY + "_FILE=" + shlex.quote(str(self.external)) + "\n")
        self.external.write_bytes(b"synthetic-next-key")
        self.assertEqual(helper().stdout, b"synthetic-file-key")
        self.external.unlink()
        self.assertEqual(helper().stdout, b"synthetic-file-key")
        self.external.write_bytes(b"synthetic-next-key")
        self.assertEqual(self.launch().stdout, b"synthetic-next-key")
        self.assertEqual(helper().stdout, b"synthetic-next-key")
        self.envfile.write_text("")
        self.assertEqual(self.launch().stdout, b"synthetic-saved-key")
        self.assertFalse(snapshot.exists())
        self.assertEqual(helper().stdout, b"synthetic-saved-key")
        if KEY == "DOCLING_API_KEY":
            header_key = b'synthetic $(literal) "quoted" `raw`'
            self.external.write_bytes(header_key)
            self.assertEqual(self.launch(values).stdout, header_key)
            shown = helper("--showkey")
            self.assertEqual(shown.returncode, 0, shown.stderr)
            line = next(line for line in shown.stdout.decode().splitlines()
                        if line.startswith("Use with: "))
            self.assertEqual(shlex.split(line.removeprefix("Use with: ")),
                             ["-H", "X-Api-Key: " + header_key.decode()])

    def test_invalid_and_conflicting_inputs_leave_existing_state_unchanged(self):
        if os.getenv("ROLLOUT_SOURCE_REF"):
            self.skipTest("File-input candidate only")
        self.saved.write_bytes(b"synthetic-saved-key")
        before = {p.name: p.read_bytes() for p in self.data.iterdir()}
        for values in ({KEY + "_FILE": ""}, {KEY + "_FILE": str(self.base / "missing")},
                       {KEY + "_FILE": str(self.external), KEY: ""}):
            with self.subTest(values=list(values)):
                result = self.launch(values)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn((KEY + "_FILE").encode(), result.stderr)
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.data.iterdir()})
        self.envfile.write_text(f"{KEY}=''\n")
        self.assertNotEqual(self.launch({KEY + "_FILE": str(self.external)}).returncode, 0)

    def test_persisted_file_mode_helper_ignores_stale_container_env(self):
        if DATA not in ("ollama", "litellm") or os.getenv("ROLLOUT_SOURCE_REF"):
            self.skipTest("Persisted-source-marker candidate only")
        self.envfile.write_text("unset " + KEY + "\n" + KEY + "_FILE=" +
                                shlex.quote(str(self.external)) + "\n")
        result = self.launch({KEY: "synthetic-stale-container-key"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"synthetic-file-key")
        marker = Path(str(self.saved) + ".file-source")
        self.assertEqual(marker.read_bytes(), b"1")
        self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
        self.external.unlink()
        result = subprocess.run(["bash", "-c", self.remap(source("manage.sh")),
                                 "helper", "--getkey"],
                                env={"PATH": os.environ["PATH"],
                                     "KUBERNETES_SERVICE_HOST": "fixture",
                                     KEY: "synthetic-stale-container-key"},
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"synthetic-file-key")
        self.envfile.write_text("")
        self.assertEqual(self.launch({KEY: "synthetic-next-direct-key"}).returncode, 0)
        self.assertFalse(marker.exists())

    def test_remove_file_mode_without_saved_key_preserves_legacy_noauth(self):
        if INTEGRATED or os.getenv("ROLLOUT_SOURCE_REF"):
            self.skipTest("No-persistent-copy candidate only")
        self.assertEqual(self.launch({KEY + "_FILE": str(self.external)}).returncode, 0)
        self.assertFalse(self.saved.exists())
        # All five real launchers save this metadata after successful setup;
        # EmbedCrate's fragment stops before those writes, so model that state.
        (self.data / ".server_addr").write_text("synthetic-running-server")
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertFalse(self.saved.exists())
        result = subprocess.run(["bash", "-c", self.remap(source("manage.sh")),
                                 "helper", "--getkey"], capture_output=True, timeout=5,
                                env={"PATH": os.environ["PATH"],
                                     "KUBERNETES_SERVICE_HOST": "fixture"})
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")


@unittest.skipIf(bool(os.getenv("ROLLOUT_SOURCE_REF")), "File-input candidate only")
class ParserTests(unittest.TestCase):
    def test_every_declared_input_and_mounted_source_conflicts(self):
        # Exercise the actual env-file and normalization bridge for every role.
        text = source("run.sh")
        pairs = re.findall(r"^secret_resolve (\w+) (\w+)$", text, re.M)
        prefix = text.split("# Apply defaults", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / ENV
            config.touch()
            prefix = prefix.replace(f"/{ENV}", str(config))
            for name, kind in pairs:
                value = (b'postgresql://user:p%40ss@db/name?x="quoted"&space= ' if "URL" in name
                         else b' password "quotes" \\ $literal ' if kind == "text"
                         else b'synthetic header "key":/==' if kind == "header"
                         else b"synthetic-token._~+/==")
                secret = path / "secret with spaces"
                secret.write_bytes(value + b"\r\n\n")
                env = {"PATH": os.environ["PATH"], "KUBERNETES_SERVICE_HOST": "fixture",
                       name + "_FILE": str(secret)}
                command = prefix + '\nprintf "%s" "${' + name + '}"\n'
                result = subprocess.run(["bash", "-c", command], env=env,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, value)
                config.write_text(name + "=''\n")
                result = subprocess.run(["bash", "-c", command], env=env,
                                        capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn((name + "_FILE").encode(), result.stderr)
                config.write_text(name + "_FILE=" + shlex.quote(str(secret)) + "\n")
                # Matching mounted values override container assignments, while
                # conflicts across the two different names still fail.
                env[name + "_FILE"] = str(path / "missing")
                result = subprocess.run(["bash", "-c", command], env=env,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, value)
                env[name] = ""
                result = subprocess.run(["bash", "-c", command], env=env,
                                        capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn((name + "_FILE").encode(), result.stderr)
                config.write_text("")

    def test_exact_bytes_and_source_presence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secret with spaces"
            helpers = source("run.sh").split("# Begin secret-file helpers\n", 1)[1].split(
                "# End secret-file helpers", 1)[0]
            shell = helpers + '\nsecret_resolve TEST_SECRET text; printf "%s" "$TEST_SECRET"'
            def resolve(values):
                return subprocess.run(["bash", "-c", shell],
                                      env={"PATH": os.environ["PATH"], **values},
                                      capture_output=True, timeout=5)
            self.assertEqual(resolve({}).returncode, 0)
            self.assertEqual(resolve({"TEST_SECRET": ""}).returncode, 0)
            for ending in (b"", b"\n", b"\r\n", b"\n\r\n\n\r\n"):
                value = b' password "quotes" \\ $() `literal` :@/?#% '
                path.write_bytes(value + ending)
                result = resolve({"TEST_SECRET_FILE": str(path)})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, value)
            for value in (b"", b"\n\r\n", b"key\r", b"key\nnext", b"key\rnext", b"key\r\r\n", b"key\n\r",
                          b"key\0ignored", b"key\t", b"key\x01", b"key\x7f"):
                path.write_bytes(value)
                result = resolve({"TEST_SECRET_FILE": str(path)})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn(b"ignored", result.stderr)
            for control in [*range(32), 127]:
                path.write_bytes(b"key" + bytes([control]) + b"tail")
                self.assertNotEqual(resolve({"TEST_SECRET_FILE": str(path)}).returncode, 0)
            path.write_bytes(b"synthetic-key")
            path.chmod(0)
            if os.geteuid() != 0:
                self.assertNotEqual(resolve({"TEST_SECRET_FILE": str(path)}).returncode, 0)
            path.chmod(0o600)
            for direct in ("", "synthetic-direct"):
                self.assertNotEqual(resolve({"TEST_SECRET": direct,
                                             "TEST_SECRET_FILE": str(path)}).returncode, 0)
            self.assertNotEqual(resolve({"TEST_SECRET_FILE": ""}).returncode, 0)
            for target in (Path(directory), Path(directory) / "missing"):
                self.assertNotEqual(resolve({"TEST_SECRET_FILE": str(target)}).returncode, 0)
            link = Path(directory) / "link"
            link.symlink_to(path)
            self.assertEqual(resolve({"TEST_SECRET_FILE": str(link)}).stdout, b"synthetic-key")
            path.unlink()
            self.assertNotEqual(resolve({"TEST_SECRET_FILE": str(link)}).returncode, 0)
            os.mkfifo(path)
            self.assertNotEqual(resolve({"TEST_SECRET_FILE": str(path)}).returncode, 0)


if __name__ == "__main__":
    unittest.main()

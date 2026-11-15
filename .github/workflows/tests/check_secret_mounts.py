"""Verify real standalone Compose/Podman secret mounts using a candidate image.

Usage: python3 check_secret_mounts.py docker|podman candidate-image
This checks mounting and resolution, not model/service authorization. No user
volumes are used. Invoke only with an image built from the matching candidate.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    engine, image = sys.argv[1:]
    name = "secret-input-test-" + uuid.uuid4().hex
    # Load only packaged helper definitions, without executing service startup.
    command = ['bash', '-c', 'set -e; helpers=$(sed -n "'
               '/^# Begin secret-file helpers$/,/^# End secret-file helpers$/p" '
               '/opt/src/run.sh); [ -n "$helpers" ]; . /dev/stdin <<< "$helpers"; '
               'secret_resolve TEST_SECRET bearer; '
               '[ "$TEST_SECRET" = synthetic-mount-test ]']
    if engine == "docker":
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "key").write_bytes(b"synthetic-mount-test\r\n")
            (path / "key").chmod(0o444)
            config = {"services": {"check": {"image": image,
                      "entrypoint": command, "environment": {
                          "TEST_SECRET_FILE": "/run/secrets/test_secret"},
                      "secrets": ["test_secret"]}},
                      "secrets": {"test_secret": {"file": "./key"}}}
            (path / "compose.json").write_text(json.dumps(config))
            base = [engine, "compose", "-p", name, "-f", str(path / "compose.json")]
            try:
                run(base + ["up", "--abort-on-container-exit", "--exit-code-from", "check"])
            finally:
                run(base + ["down", "--volumes", "--remove-orphans"])
    elif engine == "podman":
        run([engine, "secret", "create", name, "-"], input=b"synthetic-mount-test\n")
        try:
            run([engine, "run", "--rm", "--name", name,
                 "--secret", name + ",type=mount,target=test_secret,mode=0444",
                 "-e", "TEST_SECRET_FILE=/run/secrets/test_secret",
                 "--entrypoint", command[0], image, *command[1:]])
        finally:
            subprocess.run([engine, "rm", "-f", name], capture_output=True)
            run([engine, "secret", "rm", name])
    else:
        raise ValueError("Use docker or podman")


if __name__ == "__main__":
    main()

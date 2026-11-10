# Regression checks

## Running the checks

Run the dependency-free unit suite from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s .github/workflows/tests
```

For image checks, maintainers can select **test** in GitHub Actions, choose
**Run workflow**, and select the branch. The [test workflow](../test.yml) runs
the [CPU](../whisper_test.yml) and [CUDA-image](../cuda_test.yml) checks without
publishing images. Use disposable Linux runners or containers for model-backed
checks, with separate model caches and output directories.

## Coverage

- Unit tests cover API compatibility, decoding controls, configuration parsing,
  authentication, SSE cleanup and translation-model validation. Diarization
  tests cover thread inheritance, embedding selection, verified model caches,
  offline behavior and startup readiness, with optional dependencies stubbed.
- [Colab tests](test_colab_configuration.py) check configuration, runtime-file
  downloads and startup waiting with external operations mocked.
- [HTTP checks](check_api_http.py) exercise real FastAPI routing with stubbed
  inference, including translation errors before SSE starts. Run
  `python3 .github/workflows/tests/check_api_http.py api_server.py` with isolated
  test-only `fastapi`, `python-multipart`, `uvicorn` and `httpx` dependencies.
- [Audio decoding](check_audio_decoding.py) compares default and opt-in settings
  using real CPU inference with short speech and a generated long recording.
- [Diarization checks](check_diarization.py) exercise real model configuration,
  speaker labels, output formats, offline cache reuse and concurrent batch/SSE
  requests. Image jobs also check bind-mounted settings and management commands.

The inference helpers accept `--api`, `--audio`, `--cache` and `--output` paths
and may download models. Dependency setup and CI invocations are defined in the
image workflows; keep dependencies isolated from the system Python environment.

## Public audio fixtures

- Decoding: OpenAI Whisper's [jfk.flac](https://github.com/openai/whisper/blob/6e3be77e1a105e59086e3e21ff5f609fd6fa89a5/tests/jfk.flac), under its [MIT license, Copyright (c) 2022 OpenAI](https://github.com/openai/whisper/blob/6e3be77e1a105e59086e3e21ff5f609fd6fa89a5/LICENSE).
- Diarization: Azure Samples' [katiesteve.wav](https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/5886c5b7c528bf036508abfd8b4d449ddab0164a/sampledata/audiofiles/katiesteve.wav), under its [MIT license, Copyright (c) Microsoft Corporation](https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/5886c5b7c528bf036508abfd8b4d449ddab0164a/LICENSE.md).

The image workflows pin fixture revisions and verify checksums before use.
Audio binaries are not committed here. Preserve upstream attribution and
licenses when redistributing them. The long decoding fixture combines repeated
speech and synthetic noise; it is not a reproduction of a private recording.

## Coverage limits

Unit and HTTP tests do not measure model accuracy. The CPU image checks include
amd64 and arm64; the CUDA image is tested in CPU mode on a non-GPU runner.
These jobs do not establish GPU performance or a successful hosted Colab run.
Diarization timings are diagnostic and do not guarantee fixed speedups,
identical clustering across machines or new concurrency guarantees.

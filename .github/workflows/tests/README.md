# Decoding regression checks

The dependency-free suite retains the API compatibility/beam tests and adds strict
startup parsing, default preservation, propagation, authentication and SSE cleanup:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s .github/workflows/tests
```

Run each repository's suite in a separate process. Both APIs are independently
maintained; the native API remains a self-contained file supporting Python 3.9+.
The native suite intentionally skips the Docker-only shell bridge test.

`check_api_http.py api_server.py` exercises the real FastAPI stack with a
deterministic inference spy. It requires isolated test-only `fastapi`,
`python-multipart`, `uvicorn` and `httpx` dependencies. Do not install them globally.

`check_audio_decoding.py` runs real CPU inference with `tiny`, int8, two threads,
beam 5 and fixed dependencies/model cache. It compares unset/default/opt-in options,
all short-audio task/transport combinations and a generated long recording. Its
optional `--baseline` accepts a reviewed old API for same-runtime comparisons.
Supply `--api`, `--audio`, `--cache` and `--output` paths. Keep the model/cache
and output in disposable scratch space; the helper may download the model.

## Public audio fixture

- Source: OpenAI Whisper [tests/jfk.flac](https://github.com/openai/whisper/blob/6e3be77e1a105e59086e3e21ff5f609fd6fa89a5/tests/jfk.flac).
- Immutable revision: `6e3be77e1a105e59086e3e21ff5f609fd6fa89a5`.
- SHA-256: `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`.
- Repository license: [MIT, Copyright (c) 2022 OpenAI](https://github.com/openai/whisper/blob/6e3be77e1a105e59086e3e21ff5f609fd6fa89a5/LICENSE).
- The recording is public speech, not a user's private audio. No audio binary is
  committed here. Preserve the upstream attribution/license when redistributing.
- The long fixture repeats the speech twice, adds 600 seconds of seeded Gaussian
  noise plus 60 Hz hum at low amplitude, then appends the speech again. It contains
  legitimate repetition and later speech. VAD can remove the noise; it is not a
  reproduction of issue #11's private recording.

The helper records resolved inference dependency versions, fixture hashes,
transcripts and timing. CPU checks do not establish CUDA inference quality, and
these tests do not prove that any setting eliminates every hallucination.

## Diarization thread checks

`test_diarization_embedding_model.py` covers the allowlist/default, normalization,
early startup rejection, shell/Python parity, URL encoding, cache reuse and
selection propagation. Its optional runtime dependencies are stubbed.

`test_diarization_threads.py` covers strict startup validation, inheritance,
direct-Python legacy fallback, disabled-feature isolation, shell export/parsing
parity, both native model arguments and the helper's one-thread compatibility
default. Native constructors, NumPy and model downloads are stubbed only in the
dependency-free unit suite. No extreme thread-count boundary is allocated.

`check_diarization.py` uses the real installed bindings and the public two-speaker
fixture below. Supply `--api`, `--audio`, `--cache` and `--output` in disposable
scratch space. It may download the unchanged Whisper and ONNX models. It checks
both actual native config arguments, nonempty native diarization turns, speech
segments/speaker labels, text/SRT/VTT output, and overlapping batch diarization
with a separate SSE request. SSE must complete without speaker fields. It does
not change production scheduling or establish new concurrency guarantees.

The helper compares explicit `1`, inherited `2`, and explicit `6`, with a warm-up
and three timed runs each. It records model hashes, resolved dependency versions,
CPU architecture/count, RAM, cgroup quota/cpuset/memory constraints when available,
wall/aggregate CPU time, process-wide peak RSS, turns and clusters. Peak RSS is a
cumulative process high-water mark, not an isolated allocation measurement for
each setting. Compare medians/turns; do not require cross-machine byte identity or
a fixed speedup. Resource settings can oversubscribe a constrained container.

The CPU workflow covers amd64 and arm64 diarization; the CUDA workflow exercises
the CUDA image's CPU-based diarization on a non-GPU runner. Both run the helper
with a two-CPU container quota. These are not GPU inference/performance tests.
Real container jobs must pass before release; local unit/static checks do not
establish that those jobs ran. Existing main/cron publication paths depend on
the reusable test workflows, and test-only changes do not trigger publishing.
The CPU diarization startup also checks plain bind-mounted assignments and
env-file precedence: an empty file override replaces container value `1` and
inherits the file's `WHISPER_THREADS=2`. The CUDA startup checks explicit `1`.
The mounted file also selects WeSpeaker over a conflicting container TitaNet
setting. CPU jobs check the live server log, saved selection and selected cache,
then pass WeSpeaker explicitly into the native helper and verify its result.
Docker exec processes do not inherit exports made inside the entrypoint.
The management command is checked against the saved selection while the original
container environment still names TitaNet. The CUDA job retains default-model
coverage; none of these checks rank model accuracy.

`test_colab_configuration.py` runs configuration/environment-building cells with
GPU detection and downloads mocked. It checks unchanged defaults, embedding
choices, automatic speaker counts, threshold forwarding, and that notebook
downloads cover repository-local runtime imports. It does not start a server or
establish a successful hosted Colab/GPU run. For a pre-merge hosted smoke check,
use the candidate commit's runtime scripts rather than the current main scripts.

### Public diarization fixture

- Source: Azure Samples [sampledata/audiofiles/katiesteve.wav](https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/5886c5b7c528bf036508abfd8b4d449ddab0164a/sampledata/audiofiles/katiesteve.wav).
- Immutable revision: `5886c5b7c528bf036508abfd8b4d449ddab0164a`.
- SHA-256: `d9ab6bdcd8137223386e8a0ac1d0a723f7ed029310111b9c60ef9ed6cedb4828`.
- License: [MIT, Copyright (c) Microsoft Corporation](https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/5886c5b7c528bf036508abfd8b4d449ddab0164a/LICENSE.md).
- Public two-speaker speech, 29.4875 seconds, mono 16 kHz. No private reporter audio
  or audio binary is committed. Preserve the upstream license/attribution on
  redistribution. Clustering need not be perfect for the test to pass, but native
  turns and actual ASR speech segments must be nonempty.

## Translation model compatibility

`TranslationModelTests` checks that both turbo aliases and English-only models
return HTTP 400 for batch/SSE translation before upload persistence or stream
creation. It also checks multilingual translation, turbo transcription, source
language forwarding, the loaded model versus request label, and unloaded-model
503 behavior. These tests run in the existing dependency-free suite.

`check_api_http.py` additionally checks real FastAPI routing, authentication,
JSON HTTP 400 responses before SSE, continued successful requests, cleanup and
the translation endpoint documentation. Inference remains a deterministic spy;
these checks do not download models or measure translation quality.

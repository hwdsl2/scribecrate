# Changelog

Notable changes to ScribeCrate are documented here.

## 2026-10-30

### Added

- Select a speaker embedding model with `WHISPER_DIARIZE_EMBEDDING_MODEL`, using
  a maintained allowlist of sherpa-onnx release filenames. The existing
  3D-Speaker default is unchanged; alternatives include WeSpeaker ResNet34-LM
  and NeMo TitaNet large. Models use separate cache files for reuse.
  Thanks to [@surfnterd](https://github.com/surfnterd) for configurable diarization
  embedding model support in [#16](https://github.com/hwdsl2/scribecrate/pull/16).
- `whisper_manage --downloaddiarize` now pre-downloads the embedding model
  configured at container startup, including bind-mounted env-file settings.
- Configure the embedding model and clustering threshold in the Colab notebook.

### Compatibility

- Unset/empty embedding selection preserves the existing default model,
  threshold, cache filename, thread behavior, and API formats. Invalid nonempty
  values stop startup before model loading, even with diarization disabled.
- Clustering thresholds are model-specific. Compare representative recordings
  and retune after switching, or set an exact speaker count if known.

### Fixed

- Clarified that the approximate total ONNX model size applies to the default
  embedding; alternative embedding sizes exclude the shared segmentation model.
- Added real alternative-model CI coverage for mounted-file precedence, native
  configuration and saved selection in management commands, plus dependency-free
  notebook configuration checks.

## 2026-10-27

Release: `2026.10.2`

### Compatibility

- `/v1/audio/translations` now returns HTTP 400 when the loaded model is
  `large-v3-turbo` or `turbo`, for both batch and SSE requests. Previously,
  requests were accepted even though turbo is not trained for translation.
  Use a multilingual non-turbo model such as `base`, `small`, `medium`, or
  `large-v3`: set server-side `WHISPER_MODEL`, then restart for a bind-mounted
  env file or recreate the container for Docker/Compose environment changes.
  The request's `model` field does not switch the loaded model. Turbo
  transcription and the default `base` model are unchanged.

### Changed

- The Colab demo now defaults to the multilingual `small` model and clarifies
  model-selection guidance. The Docker default remains `base`.

### Fixed

- Corrected translation model guidance in API errors, endpoint documentation,
  READMEs, model-list help and configuration comments.

## 2026-10-24

Release: `2026.10.1`

### Changed

- Established the first tagged source release. Docker images continue to update
  independently through existing moving tags.

## 2026-10-21

### Added

- Added `WHISPER_DIARIZE_THREADS` to control CPU threads for each diarization
  model.

### Compatibility

- Diarization now inherits `WHISPER_THREADS` by default, normally using two
  threads per model instead of one. Set `WHISPER_DIARIZE_THREADS=1` to retain
  the previous setting. Diarization remains on CPU, including CUDA images.

## 2026-10-17

### Added

- Added `WHISPER_CONDITION_ON_PREVIOUS_TEXT` and `WHISPER_NO_REPEAT_NGRAM_SIZE`
  for tuning repetition-prone recordings. Defaults preserve previous decoding
  behavior; malformed settings prevent startup.

## 2026-10-05

### Fixed

- Restored audio-decoding compatibility by excluding PyAV 19, which removed
  an option still used by faster-whisper.

## 2026-10-04

### Changed

- Rebranded as ScribeCrate (formerly `docker-whisper`); the Docker image name
  remains `hwdsl2/whisper-server`.

## 2026-07-09

### Added

- Added a per-request `beam` override for transcription and translation,
  bounded by `WHISPER_MAX_REQUEST_BEAM` (default: 10).
- Added anonymous aggregate deployment/upgrade counts. Opt out with
  `WHISPER_DISABLE_USAGE_COUNTS=1`.

## 2026-06-25

### Changed

- Transcription and translation now reject `temperature` values outside 0-1
  with HTTP 400.

## 2026-06-21

### Changed

- Unsupported OpenAI-only diarization and logprob options now return HTTP 400
  instead of being silently ignored.

## 2026-06-20

### Added

- Added automatic API-key generation by default for fresh installs with
  persistent storage. Retrieve the key with `whisper_manage --getkey`.
  Existing volumes without a key retain their previous authentication behavior.

## 2026-06-16

### Changed

- Limited uploaded audio to 1024 MB by default; configure
  `WHISPER_MAX_UPLOAD_MB` to change the limit or set it to `0` to disable it.

### Fixed

- Corrected subtitle timestamp rounding for SRT and VTT output.

## 2026-06-04

### Fixed

- Fixed diarization being silently disabled when only
  `WHISPER_DIARIZATION=true` was set and tuning options were left at defaults.

## Earlier notable changes

- Added OpenAI-compatible transcription and English-translation endpoints
  and word-level timestamps.
- Added OpenAI-compatible streaming with `transcript.text.delta` and
  `transcript.text.done` events.
- Added optional speaker diarization with speaker labels in non-streaming
  responses.
- Added NVIDIA CUDA images, persistent model caching, and offline operation
  with pre-downloaded models.

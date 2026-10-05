# Contributing

Thanks for helping improve this project. This repository maintains ScribeCrate, the self-hosted transcription API and its Docker image; changes that only affect multi-service orchestration belong in [self-hosted-ai-stack](https://github.com/hwdsl2/self-hosted-ai-stack).

## Before You Start

- Search existing issues and pull requests.
- Keep changes focused and easy to review.
- For upstream `faster-whisper`, CTranslate2, or Whisper behavior, check the upstream project first.
- Do not include API keys, audio with private content, model files, logs with secrets, or provider credentials.

## Pull Requests

- Update `README.md`, env examples, or compose examples when behavior changes.
- Include the Docker image/tag, architecture, and whether CPU or CUDA was tested.
- For upstream version changes, link the upstream release, tag, or commit.

## Testing

Test the smallest relevant path before opening a PR, for example:

- Build or run the CPU image when Dockerfile/runtime behavior changes.
- Build or run the CUDA image when GPU behavior changes.
- Exercise the transcription API or helper script touched by the change.
- Run ShellCheck when editing shell scripts.

---
name: Bug report
about: Tell us about a problem you are experiencing
title: ''
labels: ''
assignees: ''

---
**Checklist**

- [ ] I read the [README](https://github.com/hwdsl2/scribecrate/blob/main/README.md) or the relevant section
- [ ] I searched existing [Issues](https://github.com/hwdsl2/scribecrate/issues?q=is%3Aissue)
- [ ] This issue is about the ScribeCrate Docker image/config/API, not only faster-whisper itself

<!---
If this is a reproducible bug in the transcription engine itself, it may belong in faster-whisper: https://github.com/SYSTRAN/faster-whisper. This project uses faster-whisper as its runtime engine; OpenAI compatibility refers to the API shape and Whisper model family.
--->

**Describe the issue**
A clear and concise description of the problem.

**Deployment context**
- [ ] Standalone container
- [ ] Part of [self-hosted-ai-stack](https://github.com/hwdsl2/self-hosted-ai-stack)

**To Reproduce**
Steps to reproduce the behavior:

1. ...
2. ...

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment**
- Docker host OS: [e.g. Ubuntu 24.04]
- Hosting provider (if applicable): [e.g. AWS, GCP, home server]
- CPU architecture: [e.g. amd64, arm64]
- Image/tag: [e.g. `hwdsl2/whisper-server:latest`]
- Start method: [docker run / docker compose / other]
- Published port(s): [9000]

**Configuration**
Remove secrets, API keys, tokens and private URLs before posting.

- Env file or variables changed: [whisper.env / `-e` / compose `environment`]
- Docker run or compose changes:

**Service details**
- Endpoint used (`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`, `/docs`):
- Audio file format/size and request parameters:
- Active model and `WHISPER_*` settings:
- Streaming, response format, word timestamps, or diarization settings, if relevant:
- Management command output, if relevant (for example `docker exec whisper whisper_manage --showinfo`):
- GPU/CUDA image tag and NVIDIA driver/toolkit versions, if relevant:

**Logs**
Add relevant logs with secrets removed.

```bash
docker logs whisper
```

If using Docker Compose, you can also include:

```bash
docker compose logs whisper
```

**Additional context**
Add any other context about the problem here.

# python-agent — Voice Agent driver for the self-contained container

Drives Deepgram's Voice Agent (`/v1/agent/converse`) against the self-contained
SageMaker image, where Flux ASR, a local LLM, and Flux TTS all run inside one
container with no egress.

Built for the image in `deepgram-aws-sagemaker` at
`docker/inference-billing-va/Dockerfile`.

```bash
uv sync
```

## Check the container first

```bash
uv run agent_converse.py health --host <ec2-host>
```

Probes each process on its own port and reports which are up:

| component | port | probe |
|---|---|---|
| impeller-asr | 8083 | `/v2/models` |
| impeller-tts | 8183 | `/v2/models` |
| vLLM | 3081 | `/v1/models` — **loopback-only, reported SKIP** |
| stem | 8092 | `/health`, `/v1/agent/settings/think/providers` |
| inference-shim | 8080 | `/ping` |

Exits non-zero if anything is down. Worth running before every conversation —
it turns "the agent didn't reply" into a named component in a couple of seconds.

**vLLM always reports SKIP, and that is correct.** The entrypoint starts it with
`--host 127.0.0.1`, so it listens on the *container's* loopback and `-p 3081:3081`
cannot reach it. That is deliberate — the LLM is an internal leg that stem dials
on loopback from inside the container, and it should not be exposed. Check it
directly instead:

```bash
docker exec <container> curl -sf http://127.0.0.1:3081/v1/models
```

An HTTP error on that port (as opposed to a connection failure) is still reported
as a real FAIL, since something answered.

## Run a conversation

```bash
uv run agent_converse.py talk --host <ec2-host> \
    --file ../test-audio-files/127389__acclivity__thetimehascome_mono16k.wav \
    --out reply.wav
```

Streams the WAV at real-time pace, holds the socket open (`--linger`, default
20 s) while end-of-turn fires and the LLM and TTS run, then scores the three legs
independently:

```
  PASS  ASR  (Flux -> user transcript)     1 turn(s)
  PASS  LLM  (agent text reply)            1 turn(s)
  PASS  TTS  (non-silent audio out)        412800 bytes, peak amplitude 8134
```

Input must be **mono 16-bit PCM at 16 kHz** — stem transcodes into Flux at that
rate regardless, so matching it avoids a resample and makes a mismatch fail here
rather than degrade quality silently.

## Notes

- **Port 8092 is stem, not the shim.** Talking to stem directly keeps the
  SageMaker bidi transport and SigV4 out of the picture while the speech pipeline
  is what's being debugged. The shim front door (8080) is what a SageMaker deploy
  uses.
- **`--llm-url` is resolved inside the container**, so it stays loopback
  (`http://127.0.0.1:3081/...`) even when `--host` is a remote EC2 box.
- **`--llm-model` must match the container's vLLM `--served-model-name`**
  (`dsva` by default).
- The `think` leg uses provider type `open_ai` with an explicit endpoint rather
  than stem's built-in `deepgram` provider. The latter also defaults to a
  localhost OpenAI-shaped URL, but it rejects function calling and requires a
  server-side model allowlist; an explicit endpoint skips both.
- `speak` deliberately sends **no** endpoint. Speak endpoints are restricted to
  `https`/`wss` unconditionally, so a loopback one is rejected — leaving it unset
  routes TTS through stem's driver pool to the TTS impeller.

## Audio came back but nothing was transcribed

That is usually the greeting alone: it is synthesized before any audio is sent,
so TTS can succeed while ASR produced nothing. Check the ASR engine and the
`--asr-model` value. The driver calls this case out explicitly when it sees it.

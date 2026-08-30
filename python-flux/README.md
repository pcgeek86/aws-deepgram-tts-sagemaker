# Deepgram Flux STT SageMaker Stress Test

Streams audio to multiple simultaneous bidirectional connections to a
**Deepgram Flux** model deployed on Amazon SageMaker. Two input modes are
supported:

- **`file`** — streams a WAV file at real-time pace (repeatable load testing).
- **`microphone`** — captures live audio from a microphone via PyAudio.

## What is Flux?

Flux (`flux-general-en`) is Deepgram's conversational speech recognition model
built for voice agents. Unlike Nova-3, Flux uses the **`/v2/listen` endpoint**
and a **turn-based protocol** with integrated end-of-turn detection — no
external VAD or server-side VAD configuration required.

Key differences from Nova-3:

| Aspect | Nova-3 | Flux |
|---|---|---|
| Endpoint path | `/v1/listen` | `/v2/listen` |
| Response type | `channel.alternatives` | `TurnInfo` events |
| Turn detection | External | Model-integrated |
| Barge-in detection | Manual | Native (`StartOfTurn`) |
| Dynamic configuration | No | `Configure` message |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS credentials configured (CLI, environment variables, or IAM role)
- A SageMaker endpoint running the Deepgram Flux model
  - The endpoint's engine config must include `listen_v2 = true`
- **For `file` mode:** a 16-bit PCM WAV file
- **For `microphone` mode:** PortAudio library (see [Setup](#setup))

### Convert audio to the required format

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 input.wav
```

## Setup

```bash
cd python-flux
uv sync
```

**macOS — microphone support requires PortAudio:**

```bash
brew install portaudio
uv sync
```

## Usage

```
uv run flux_stress.py <SUBCOMMAND> <endpoint_name> [options]
```

Subcommands:

| Subcommand | Description |
|---|---|
| `list-endpoints` | List available SageMaker endpoints in the target region |
| `file` | Stream a WAV audio file at real-time pace |
| `microphone` | Capture live microphone input and stream it in real-time |

Run `uv run flux_stress.py <subcommand> --help` for full option details.

---

## `list-endpoints` subcommand

Lists SageMaker endpoints in the target region along with their status and
timestamps. Useful for discovering available endpoint names before running a
stress test.

### Options

| Flag | Default | Description |
|---|---|---|
| `--region REGION` | `us-east-1` | AWS region to query |
| `--status STATUS` | *(all)* | Filter by status: `inservice`, `creating`, `updating`, `rollingback`, `systemupdating`, `failed`, `deleting`, `outofservice` |
| `--log-level LEVEL` | `WARNING` | Logging verbosity |

### Examples

List all endpoints in the default region:

```bash
uv run flux_stress.py list-endpoints
```

List only `InService` endpoints in a specific region:

```bash
uv run flux_stress.py list-endpoints --region us-west-2 --status inservice
```

### Example output

```
SageMaker Endpoints  [us-east-1]
------------------------------------------------------------
ENDPOINT NAME            STATUS     CREATED               LAST MODIFIED
------------------------------------------------------------
my-flux-endpoint         InService  2026-03-01 14:22:05   2026-03-05 09:11:42
my-flux-endpoint-dev     Creating   2026-03-10 08:00:13   2026-03-10 08:00:13
------------------------------------------------------------
2 endpoint(s)
```

---

## `file` subcommand

### Options

| Flag | Default | Description |
|---|---|---|
| `--file WAV_FILE` | *(required)* | 16-bit PCM WAV file to stream |
| `--connections N` | `1` | Number of simultaneous Flux connections |
| `--model MODEL` | `flux-general-en` | Flux model variant |
| `--eot-threshold 0.5-0.9` | `0.7` (server default) | EndOfTurn confidence threshold |
| `--eager-eot-threshold 0.3-0.9` | *(disabled)* | Enables EagerEndOfTurn events; must be ≤ `--eot-threshold` |
| `--eot-timeout-ms 500-10000` | `5000` (server default) | Max silence before forced EndOfTurn |
| `--keyterms TERM1,TERM2` | *(none)* | Comma-separated keyterms for recognition boosting |
| `--encoding ENC` | `linear16` | Audio encoding (`linear16`/`linear32`/`mulaw`/`alaw`/`opus`/`ogg-opus`) |
| `--language-hints en,es` | *(none)* | Bias recognition to language codes — **multilingual model only** (`flux-general-multi`); sending to `flux-general-en` returns HTTP 400 |
| `--profanity-filter true\|false` | *(server default `false`)* | Profanity filtering |
| `--extra "k=v&k2=v2"` | *(none)* | Extra query params appended verbatim (e.g. `mip_opt_out=true&tag=foo`) |
| `--summary-jsonl PATH` | *(none)* | Write a per-connection JSON summary (turns, Configure acks, errors) — consumed by the e2e driver |
| `--batch-size N` | *(all at once)* | Open connections in batches of N (ramp-up) |
| `--batch-delay SECONDS` | `0` | Delay between connection batches |
| `--keepalive-interval SECONDS` | *(off)* | Send a tiny binary keepalive frame every N seconds (during the idle hold and while streaming). **Not** a JSON `KeepAlive` — that's Nova-only; Flux keeps alive via WebSocket ping / continued frames |
| `--idle-before-audio SECONDS` | *(off)* | Hold the connection open for N seconds of silence **before** streaming audio, sending only keepalive frames (every `--keepalive-interval`, default 10s). Verifies the connection survives a long idle gap — the audio that follows must still transcribe |
| `--finalize-at-end` | *(off)* | Send `Finalize` after the last audio chunk to flush the final turn |
| `--reconfigure-after SECONDS` | *(off)* | Send one mid-stream `Configure` after N seconds (with the `--reconfigure-*` values below) — exercises `ConfigureSuccess`/`ConfigureFailure` |
| `--reconfigure-eot-threshold` / `--reconfigure-eager-eot-threshold` / `--reconfigure-eot-timeout-ms` / `--reconfigure-keyterms` / `--reconfigure-language-hints` | *(none)* | Values applied in the mid-stream `Configure` |
| `--region REGION` | `us-east-1` | AWS region |
| `--skip-verify` | *(off)* | Skip the `DescribeEndpoint` != `InService` pre-flight check (e.g. blue/green `Updating`) |
| `--loop` | *(off)* | Loop the WAV file continuously |
| `--duration SECONDS` | *(until file ends)* | Stop automatically after N seconds |
| `--log-level LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR / CRITICAL |

> **Per-connection streaming:** in `file` mode each connection streams its own
> independent copy of the WAV from the start (paced to real time), so
> per-connection transcripts stay complete and comparable even under ramped
> concurrency (`--batch-size`/`--batch-delay`). The `microphone` mode broadcasts
> one shared capture to all connections.

### Examples

Single connection, one pass through the file:

```bash
uv run flux_stress.py file my-flux-endpoint --file audio.wav
```

10 connections looping for 60 seconds:

```bash
uv run flux_stress.py file my-flux-endpoint \
  --file audio.wav \
  --connections 10 \
  --loop \
  --duration 60
```

Custom end-of-turn thresholds with eager detection:

```bash
uv run flux_stress.py file my-flux-endpoint \
  --file audio.wav \
  --eot-threshold 0.8 \
  --eager-eot-threshold 0.5 \
  --eot-timeout-ms 3000
```

Keyterm boosting with debug logging:

```bash
uv run flux_stress.py file my-flux-endpoint \
  --file audio.wav \
  --keyterms "SageMaker,Deepgram,Flux" \
  --log-level DEBUG
```

---

## `microphone` subcommand

### Options

| Flag | Default | Description |
|---|---|---|
| `--connections N` | `1` | Number of simultaneous Flux connections |
| `--model MODEL` | `flux-general-en` | Flux model variant |
| `--sample-rate HZ` | `16000` | Microphone sample rate |
| `--device INDEX` | system default | PyAudio input device index |
| `--list-devices` | — | List available input devices and exit |
| `--eot-threshold 0.5-0.9` | `0.7` (server default) | EndOfTurn confidence threshold |
| `--eager-eot-threshold 0.3-0.9` | *(disabled)* | EagerEndOfTurn threshold; must be ≤ `--eot-threshold` |
| `--eot-timeout-ms 500-10000` | `5000` (server default) | Max silence before forced EndOfTurn |
| `--keyterms TERM1,TERM2` | *(none)* | Comma-separated keyterms for recognition boosting |
| `--region REGION` | `us-east-1` | AWS region |
| `--skip-verify` | *(off)* | Skip the `DescribeEndpoint` != `InService` pre-flight check (e.g. blue/green `Updating`) |
| `--duration SECONDS` | *(until Ctrl+C)* | Stop automatically after N seconds |
| `--log-level LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR / CRITICAL |

### Examples

List available microphone input devices:

```bash
uv run flux_stress.py microphone my-flux-endpoint --list-devices
```

Single connection using the system default microphone:

```bash
uv run flux_stress.py microphone my-flux-endpoint
```

5 connections for 30 seconds:

```bash
uv run flux_stress.py microphone my-flux-endpoint \
  --connections 5 \
  --duration 30
```

Specific device with eager end-of-turn detection:

```bash
uv run flux_stress.py microphone my-flux-endpoint \
  --device 2 \
  --eot-threshold 0.8 \
  --eager-eot-threshold 0.5
```

---

## Output format

```
[Conn 1]   hello how are you [update]                        ← Update (interim, ~250ms)
[Conn 1] ~ hello, how are you? (87.3%) [eager, turn 0]      ← EagerEndOfTurn
[Conn 1] ✓ hello, how are you? (91.2%) [turn 0]             ← EndOfTurn (final)
[Conn 1]   ... resumed [turn 1]                              ← TurnResumed (barge-in after eager)
```

Legend:
- `✓` — `EndOfTurn` (final transcript)
- `~` — `EagerEndOfTurn` (high likelihood turn is complete; useful for speculative LLM pre-processing)
- `[update]` — `Update` (periodic interim transcript, not a turn boundary)
- `... resumed` — `TurnResumed` (user continued speaking after `EagerEndOfTurn`)

## Flux Protocol Summary

### Client → Server messages

| Message | Format | Purpose |
|---|---|---|
| Audio | Binary bytes | Raw PCM audio (80ms chunks recommended) |
| `Configure` | JSON text | Update thresholds or keyterms mid-stream |
| `Finalize` | JSON text | Flush buffered audio; force end current turn |
| `CloseStream` | JSON text | Gracefully terminate the stream |

> **No `KeepAlive` message.** Unlike Nova (`/v1/listen`), Flux has no JSON
> `{"type":"KeepAlive"}` control message — it relies on the WebSocket ping for
> idle liveness. Over the SageMaker bidirectional-stream transport there's no
> WS-ping primitive (only UTF8/BINARY payload parts), so this client holds a
> connection open through silence by sending a minimal binary frame; any frame
> resets the server's idle timer (see `--idle-before-audio`).

### Server → Client messages

| Message | Description |
|---|---|
| `Connected` | Emitted once on stream open |
| `TurnInfo` | Transcript update; `event` field indicates state (see below) |
| `ConfigureSuccess` | Confirms a `Configure` was applied |
| `ConfigureFailure` | Rejects a `Configure` (e.g. constraint violation) |
| `Error` | Fatal server error; connection terminated |

### TurnInfo event states

```
[Ready]
    │ StartOfTurn
    ▼
[Speaking]
    │ EagerEndOfTurn (if eager_eot_threshold set)
    ▼
[AwaitingEnd]
   / \
TurnResumed  EndOfTurn
   │              │
[Speaking]   [Ready, turn_index++]
```

## End-to-end correctness driver (`e2e/`)

A run-everything correctness gate sits on top of `flux_stress.py` in `e2e/`,
mirroring the STT suite. Flux is streaming-only, so there is a single driver:

- **`e2e/e2e_test_streaming.py`** — downloads `https://dpgr.am/spacewalk.wav`
  (~25 s English mono), multiplies it to a ~15 min long-form variant, drives
  `flux_stress.py file` through ~15 scenarios, reads each connection's
  `--summary-jsonl`, and validates the combined `EndOfTurn` transcript against
  the known reference via **Word Error Rate** (≤ 5 % by default) plus
  Flux-specific assertions (eager events emitted, `Configure` accepted/rejected,
  language detection).

The suite runs against **either Flux model** — pass `--model` to match the
endpoint. The language-hint coverage adapts to the model:

```bash
cd python-flux
# English-only endpoint (default):
uv run e2e/e2e_test_streaming.py your-flux-en-endpoint --region us-east-2

# multilingual endpoint:
uv run e2e/e2e_test_streaming.py your-flux-multi-endpoint \
  --model flux-general-multi --region us-east-2

# list scenarios for a model (the language-hint scenario differs by model):
uv run e2e/e2e_test_streaming.py --list                          # en
uv run e2e/e2e_test_streaming.py --list --model flux-general-multi
```

| Scenario | What it checks |
|---|---|
| `basic_25s` | 1 conn, 25 s file, defaults — baseline WER |
| `concurrent_5x_25s` | 5 simultaneous connections |
| `concurrent_10x_15min` | 10 connections on the ~15 min file — sustained load |
| `ramp_10x_step5` | 10 conns in batches of 5, 2 s apart |
| `feature_eot_threshold_high` | `--eot-threshold 0.9` (later, higher-confidence EoT) |
| `feature_eot_timeout_ms` | `--eot-timeout-ms 600` (force EoT on short silence) |
| `feature_eager_eot` | `--eager-eot-threshold 0.5` — asserts `EagerEndOfTurn` is emitted |
| `feature_keyterm` | `--keyterms spacewalk,female` — presence check (soft) |
| `feature_profanity_filter` | `--profanity-filter true` — PASS-WITH-NOTE on bundles that reject `profanity_filter` as an unknown query param |
| `feature_encoding_linear16` | explicit `--encoding linear16` |
| `feature_mip_opt_out` | `mip_opt_out=true` (smoke) |
| `feature_configure_thresholds` | mid-stream `Configure` → asserts `ConfigureSuccess`; PASS-WITH-NOTE on `CloseStream`-only bundles that reject `Configure` |
| `feature_configure_failure` | mid-stream `Configure` with `eager > eot` → asserts `ConfigureFailure`; PASS-WITH-NOTE on bundles that reject `Configure` |
| `feature_keepalive_idle` | `--idle-before-audio 65 --keepalive-interval 10` — holds the stream open through a 65 s silence gap with tiny binary keepalive frames, then transcribes; asserts WER + `session_duration ≥ 60 s`. Post-idle transcription is the proof the connection survived |
| `feature_finalize` | `--finalize-at-end` — PASS-WITH-NOTE on bundles that reject `Finalize` as an unknown variant |
| `feature_lang_hint_multi` *(multilingual model only)* | `language_hint en` — asserts `TurnInfo.languages` populated |
| `negative_lang_hint_on_en` *(English-only model only)* | `language_hint es` — expects rejection (HTTP 400; not supported on `flux-general-en`) |

Exit code 0 = all pass, non-zero = any scenario failed. Per-scenario
stdout / stderr / summary-jsonl land in `<workdir>/logs/`; aggregated
`results.json` at `<workdir>/results.json`. Default workdir:
`/tmp/dg-sagemaker-e2e/flux/<timestamp>/`.

Not covered (need re-encoded fixtures the suite doesn't generate): non-`linear16`
encodings and alternate sample rates — the canonical sample is 16 kHz `linear16`.
Parameter coverage is scoped to the Flux docs
(https://developers.deepgram.com/docs/flux/) as of the June 2026 audit.

**In-band control-message support is bundle-versioned.** Observed against the
deployed packages: `flux-english-20260311` accepts only `CloseStream`;
`flux-multi-20260417` adds `Configure`; neither accepts `Finalize` (rejected as
`UNPARSABLE_CLIENT_MESSAGE: unknown variant`). The `feature_configure_*` /
`feature_finalize` scenarios PASS-WITH-NOTE when the message is rejected and
pass outright once a bundle implements it — so the suite stays green across
versions without hiding the gap. `feature_keepalive_idle` uses no control
message (just binary frames), so it is not bundle-gated: it fails for real if a
connection can't survive a 60 s idle gap.

## Self-Hosted / SageMaker Notes

- The SageMaker endpoint engine configuration must include `listen_v2 = true`
- Flux must run on dedicated GPU resources isolated from other models
- The bidirectional stream uses HTTP/2 on port 8443:
  `https://runtime.sagemaker.<region>.amazonaws.com:8443`

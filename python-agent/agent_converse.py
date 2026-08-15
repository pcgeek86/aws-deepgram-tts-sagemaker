#!/usr/bin/env python3
"""Drive Deepgram's Voice Agent (`/v1/agent/converse`) against a self-contained
container: Flux ASR + a local LLM + Flux TTS, all inside one endpoint.

Two modes:

  health   Probe every process in the container and report what is up. Run this
           first — it turns "the conversation didn't work" into a specific
           failing component, and it is much faster than a full session.

  talk     Stream a WAV file at real-time pace, then collect the agent's reply:
           the user transcript, the LLM's text, and the synthesized audio.

Against the container's stem directly (the EC2 loop):

    uv run agent_converse.py health --host <ec2-host>
    uv run agent_converse.py talk   --host <ec2-host> \
        --file ../test-audio-files/127389__acclivity__thetimehascome_mono16k.wav \
        --out reply.wav

The default port is stem's 8092, NOT the shim's 8080. Talking to stem directly
keeps the SageMaker bidi transport and its SigV4 signing out of the picture while
the speech pipeline itself is what's being debugged. The shim front door is what
a SageMaker deploy uses later.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
import wave

try:
    import websockets
except ImportError:
    sys.exit("missing dependency: run `uv sync` in python-agent/ first")


# Input must be linear16 mono @ 16 kHz: stem transcodes everything into Flux at
# that rate anyway (listen_v2/transcoder.rs), so feeding it directly avoids a
# resample and makes a mismatch a loud failure here rather than a quiet quality
# loss later.
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_MS = 20


def build_settings(args) -> dict:
    """The Settings frame that wires all three legs to loopback services.

    The `think` leg uses provider type `open_ai` with an explicit endpoint rather
    than stem's built-in `deepgram` provider, which also defaults to a localhost
    OpenAI-shaped URL. Two reasons: the `deepgram` provider rejects function
    calling outright, and it requires the model to appear in a server-side
    allowlist that would then have to track whatever vLLM is serving. With an
    explicit endpoint stem uses the URL verbatim, injects no Authorization
    header, and skips the allowlist entirely.

    `speak` deliberately carries NO endpoint. A speak endpoint is restricted to
    https/wss unconditionally, so a loopback one is rejected; leaving it unset
    routes TTS through stem's driver pool, which the container pins to the TTS
    impeller by driver purpose.
    """
    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": INPUT_RATE},
            "output": {
                "encoding": "linear16",
                "sample_rate": OUTPUT_RATE,
                "container": "none",
            },
        },
        "agent": {
            "greeting": args.greeting,
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": args.asr_model,
                    "version": "v2",
                }
            },
            "think": {
                "provider": {"type": "open_ai", "model": args.llm_model},
                "endpoint": {"url": args.llm_url, "headers": {}},
                "prompt": args.prompt,
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "version": "v2",
                    "model": args.tts_model,
                }
            },
        },
    }


def read_wav(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(
                f"{path}: need mono 16-bit PCM, got "
                f"{w.getnchannels()}ch/{w.getsampwidth() * 8}-bit"
            )
        if w.getframerate() != INPUT_RATE:
            sys.exit(f"{path}: need {INPUT_RATE} Hz, got {w.getframerate()} Hz")
        return w.readframes(w.getnframes())


def peak_amplitude(pcm: bytes) -> int:
    """Max |sample| over 16-bit LE PCM. 0 means digital silence."""
    if not pcm:
        return 0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return max(max(samples), -min(samples))


# ─────────────────────────────── health ────────────────────────────────


def _http_ok(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(400).decode("utf-8", "replace").strip()
            return 200 <= r.status < 300, f"HTTP {r.status} {body[:180]}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:  # connection refused, timeout, DNS, ...
        return False, f"{type(e).__name__}: {e}"


def cmd_health(args) -> int:
    h = args.host
    # Ordered so the first failure is usually the root cause: the engines and the
    # LLM come up independently, stem depends on nothing at boot, and the shim's
    # /ping AND-combines the stem and impeller probes.
    #
    # vLLM is marked loopback-only: the entrypoint starts it with
    # `--host 127.0.0.1`, so it listens on the CONTAINER's loopback and a
    # published -p 3081:3081 cannot reach it. That is deliberate — the LLM is an
    # internal leg, and stem dials it on loopback from inside the container. So a
    # FAIL here from a remote host says nothing about vLLM's health, and treating
    # it as a real failure would be a standing false alarm.
    checks = [
        ("impeller-asr  models", f"http://{h}:8083/v2/models", False),
        ("impeller-tts  models", f"http://{h}:8183/v2/models", False),
        ("vllm          models", f"http://{h}:3081/v1/models", True),
        ("stem          health", f"http://{h}:8092/health", False),
        ("stem          think providers",
         f"http://{h}:8092/v1/agent/settings/think/providers", False),
        ("shim          ping", f"http://{h}:8080/ping", False),
    ]
    worst = 0
    loopback_skipped = False
    for label, url, loopback_only in checks:
        ok, detail = _http_ok(url)
        if ok:
            print(f"  OK    {label:28} {url}")
            continue
        # Unreachable-from-outside is the EXPECTED result for a loopback-only
        # service, so only a connection-level failure is excused. An HTTP error
        # means something answered, and that is a real failure worth failing on.
        if loopback_only and not detail.startswith("HTTP "):
            print(f"  SKIP  {label:28} {url}")
            print("          not published outside the container (expected);"
                  " verify from inside:")
            print("          docker exec <container> curl -sf"
                  " http://127.0.0.1:3081/v1/models")
            loopback_skipped = True
            continue
        print(f"  FAIL  {label:28} {url}")
        print(f"          {detail}")
        worst = 1
    print()
    if worst:
        print("At least one component is down. `docker logs <container>` names the")
        print("process on exit; the entrypoint prints a per-process diagnostic.")
    elif loopback_skipped:
        print("All externally-published components responding "
              "(vLLM unchecked — see above).")
    else:
        print("All components responding.")
    return worst


# ──────────────────────────────── talk ─────────────────────────────────


async def cmd_talk(args) -> int:
    pcm = read_wav(args.file)
    url = f"ws://{args.host}:{args.port}/v1/agent/converse"
    settings = build_settings(args)

    print(f"connecting: {url}")
    print(f"  asr={args.asr_model}  llm={args.llm_model} @ {args.llm_url}")
    print(f"  tts={args.tts_model}")
    print(f"  audio: {len(pcm) / 2 / INPUT_RATE:.1f}s from {args.file}")
    print()

    out_audio = bytearray()
    transcripts: list[tuple[str, str]] = []
    events: list[str] = []
    fatal: str | None = None

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(settings))

        async def sender() -> None:
            # Real-time pace. Flux is a turn-taking model: shovelling the whole
            # file instantly would collapse the turn structure the end-of-turn
            # detector exists to find, so the run would not resemble a call.
            chunk = INPUT_RATE * 2 * CHUNK_MS // 1000
            start = time.monotonic()
            sent = 0
            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i : i + chunk])
                sent += 1
                target = start + sent * (CHUNK_MS / 1000)
                await asyncio.sleep(max(0, target - time.monotonic()))

            # Keep streaming SILENCE rather than going quiet. Two reasons, and
            # the first is not optional:
            #
            #  1. End-of-turn is detected from the audio itself. If the stream
            #     simply stops, Flux never receives the pause that ends the turn,
            #     so no transcript is finalized and the agent is never asked
            #     anything — the run fails with 0 user turns while ASR is fine.
            #  2. stem times the connection out when no websocket message arrives
            #     ("We waited too long for a websocket message").
            #
            # A real client streams mic audio continuously, so silence here is
            # the faithful behavior, not a workaround.
            silence = b"\x00" * chunk
            for _ in range(int(args.linger * 1000 / CHUNK_MS)):
                await ws.send(silence)
                sent += 1
                target = start + sent * (CHUNK_MS / 1000)
                await asyncio.sleep(max(0, target - time.monotonic()))

        async def receiver() -> None:
            nonlocal fatal
            async for msg in ws:
                if isinstance(msg, bytes):
                    out_audio.extend(msg)
                    continue
                try:
                    m = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                t = m.get("type", "?")
                events.append(t)
                if t == "ConversationText":
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    transcripts.append((role, content))
                    print(f"  [{role}] {content}")
                elif t in ("Error", "Warning"):
                    detail = m.get("description") or m.get("message") or json.dumps(m)
                    print(f"  [{t}] {detail}")
                    if t == "Error":
                        fatal = detail
                elif args.verbose:
                    print(f"  [{t}]")

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        await send_task
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    print()
    if args.out and out_audio:
        with wave.open(args.out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(OUTPUT_RATE)
            w.writeframes(bytes(out_audio))
        print(f"wrote {args.out} ({len(out_audio) / 2 / OUTPUT_RATE:.1f}s)")

    # Score each leg separately so a failure names the component, not the run.
    user_turns = [c for r, c in transcripts if r == "user" and c.strip()]
    agent_turns = [c for r, c in transcripts if r == "assistant" and c.strip()]
    peak = peak_amplitude(bytes(out_audio))

    results = [
        ("ASR  (Flux -> user transcript)", bool(user_turns),
         f"{len(user_turns)} turn(s)"),
        ("LLM  (agent text reply)", bool(agent_turns),
         f"{len(agent_turns)} turn(s)"),
        ("TTS  (non-silent audio out)", peak > 0,
         f"{len(out_audio)} bytes, peak amplitude {peak}"),
    ]
    print()
    for label, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:34} {detail}")
    if fatal:
        print(f"\n  server Error: {fatal}")

    failed = [label for label, ok, _ in results if not ok]
    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        # The greeting is synthesized before any audio is sent, so TTS-only
        # output with no transcript is a specific, common shape worth naming.
        if not user_turns and peak > 0:
            print("Audio came back but nothing was transcribed — that is likely the")
            print("greeting alone. Check the ASR engine and the listen provider model.")
        return 1
    print("PASS: all three legs of the loop worked end to end.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="127.0.0.1")

    ph = sub.add_parser("health", parents=[common],
                        help="probe every process in the container")
    ph.set_defaults(func=cmd_health)

    pt = sub.add_parser("talk", parents=[common], help="run one conversation")
    pt.add_argument("--port", type=int, default=8092, help="stem (default 8092)")
    pt.add_argument("--file", required=True, help="mono 16-bit 16 kHz WAV")
    pt.add_argument("--out", default="agent_reply.wav")
    pt.add_argument("--asr-model", default="flux-general-multi")
    pt.add_argument("--tts-model", default="flux-alexis-en")
    pt.add_argument("--llm-model", default="dsva",
                    help="must match the container's vLLM --served-model-name")
    pt.add_argument("--llm-url", default="http://127.0.0.1:3081/v1/chat/completions",
                    help="resolved INSIDE the container, so keep it loopback")
    pt.add_argument("--prompt",
                    default="You are a helpful voice assistant. Reply in one or "
                            "two short spoken sentences.")
    pt.add_argument("--greeting", default="Hi, how can I help?")
    pt.add_argument("--linger", type=float, default=20.0,
                    help="seconds to keep the socket open after the audio ends, "
                         "waiting for end-of-turn, the LLM, and TTS")
    pt.add_argument("--verbose", action="store_true", help="print every event")
    pt.set_defaults(func=cmd_talk)

    args = p.parse_args()
    if asyncio.iscoroutinefunction(args.func):
        return asyncio.run(args.func(args))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

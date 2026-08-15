#!/usr/bin/env python3
"""Talk to the self-contained Voice Agent with your microphone, from a local terminal.

Speech in and out stay on your machine's audio devices; the container does ASR,
the LLM, and TTS with no egress of its own.

The container is reached over an SSM port-forwarding tunnel rather than a public
port — the EC2 host does not accept inbound connections. In one terminal:

    aws ssm start-session --target <instance-id> --region us-east-2 \
        --document-name AWS-StartPortForwardingSession \
        --parameters '{"portNumber":["8092"],"localPortNumber":["8092"]}'

then in another:

    uv run agent_live.py                    # talks to ws://127.0.0.1:8092

Press Ctrl-C to hang up.

WHY THIS EXISTS ALONGSIDE `agent_converse.py talk`: a canned WAV keeps streaming
while the agent is still speaking its greeting, which triggers barge-in on every
run. A human waits for the greeting, so a live session exercises the ordinary
conversational path instead of the interrupt path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys

try:
    import sounddevice as sd
except ImportError:
    sys.exit(
        "missing dependency: run `uv sync` in python-agent/ first.\n"
        "sounddevice needs PortAudio; on macOS `brew install portaudio` if the "
        "import still fails."
    )

try:
    import websockets
except ImportError:
    sys.exit("missing dependency: run `uv sync` in python-agent/ first")

from agent_converse import CHUNK_MS, INPUT_RATE, OUTPUT_RATE, build_settings

# Mic frames per send. Matches the WAV driver's pacing so both clients present
# the same cadence to Flux's end-of-turn detector.
FRAMES_PER_CHUNK = INPUT_RATE * CHUNK_MS // 1000


async def run(args) -> int:
    url = f"ws://{args.host}:{args.port}/v1/agent/converse"
    settings = build_settings(args)

    # Mic callback runs on a PortAudio thread, so hand frames to asyncio through a
    # plain queue rather than touching the loop from that thread.
    mic_q: queue.Queue[bytes] = queue.Queue()

    def on_mic(indata, frames, time_info, status):
        if status:
            print(f"  [audio] input status: {status}", file=sys.stderr)
        mic_q.put(bytes(indata))

    print(f"connecting: {url}")
    print(f"  asr={args.asr_model}  llm={args.llm_model}  tts={args.tts_model}")

    try:
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps(settings))

            speaker = sd.RawOutputStream(
                samplerate=OUTPUT_RATE, channels=1, dtype="int16"
            )
            speaker.start()
            mic = sd.RawInputStream(
                samplerate=INPUT_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAMES_PER_CHUNK,
                callback=on_mic,
            )
            mic.start()
            print("\nMicrophone is live — start talking. Ctrl-C to hang up.\n")

            async def sender() -> None:
                loop = asyncio.get_running_loop()
                while True:
                    # Block on the queue in a worker thread so the event loop
                    # stays free to receive and play the agent's audio.
                    chunk = await loop.run_in_executor(None, mic_q.get)
                    await ws.send(chunk)

            async def receiver() -> None:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        speaker.write(msg)
                        continue
                    try:
                        m = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    t = m.get("type", "?")
                    if t == "ConversationText":
                        role = m.get("role", "?")
                        print(f"  [{role}] {m.get('content', '')}")
                    elif t == "UserStartedSpeaking":
                        # Drop already-queued agent audio so a barge-in sounds
                        # like an interruption instead of both talking at once.
                        if args.barge_in_stops_playback:
                            speaker.stop()
                            speaker.start()
                    elif t in ("Error", "Warning"):
                        detail = (
                            m.get("description") or m.get("message") or json.dumps(m)
                        )
                        print(f"  [{t}] {detail}")
                    elif args.verbose:
                        print(f"  [{t}]")

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())
            try:
                done, pending = await asyncio.wait(
                    {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
            finally:
                mic.stop()
                mic.close()
                speaker.stop()
                speaker.close()
    # Both failures mean "the tunnel isn't working", but they look nothing alike:
    # nothing listening gives ConnectionRefusedError, while a STALE tunnel (the SSM
    # session died but the local listener survives) accepts the TCP connection and
    # then never completes the upgrade, surfacing as a handshake TimeoutError with a
    # traceback that says nothing about SSM.
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        stale = isinstance(e, TimeoutError)
        print(
            f"\nCould not reach {url} ({type(e).__name__}).\n"
            + (
                "The port is open but the connection hung, which usually means a "
                "STALE tunnel: the SSM session dropped (look for 'broken pipe' in "
                "its output) while the local listener stayed up. Kill it and start "
                "a fresh one.\n"
                if stale
                else "Nothing is listening locally, so the tunnel is not running.\n"
            )
            + "\nThe EC2 host accepts no inbound connections, so this needs an SSM "
            "tunnel in another terminal:\n\n"
            "  aws ssm start-session --target <instance-id> --region us-east-2 \\\n"
            "      --document-name AWS-StartPortForwardingSession \\\n"
            "      --parameters '{\"portNumber\":[\"8092\"],"
            "\"localPortNumber\":[\"8092\"]}'\n\n"
            "Verify before retrying:  curl -sf -o /dev/null -w '%{http_code}\\n' "
            "http://127.0.0.1:8092/health   # expect 204\n",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="local end of the SSM tunnel (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8092, help="stem (default 8092)")
    p.add_argument("--asr-model", default="flux-general-multi")
    p.add_argument("--tts-model", default="flux-alexis-en")
    p.add_argument("--llm-model", default="dsva",
                   help="must match the container's vLLM --served-model-name")
    p.add_argument("--llm-url", default="http://127.0.0.1:3081/v1/chat/completions",
                   help="resolved INSIDE the container, so keep it loopback")
    p.add_argument("--prompt",
                   default="You are a helpful voice assistant. Reply in one or "
                           "two short spoken sentences.")
    p.add_argument("--greeting", default="Hi, how can I help?")
    p.add_argument("--barge-in-stops-playback", action="store_true", default=True,
                   help="flush queued agent audio when you start speaking")
    p.add_argument("--verbose", action="store_true", help="print every event")
    args = p.parse_args()

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nhung up.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

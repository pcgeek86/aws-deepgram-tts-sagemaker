#!/usr/bin/env python3
"""Flux TTS streaming e2e battery — WebSocket `/v2/speak` via SageMaker.

    AWS_PROFILE=shared-dev uv run e2e/e2e_test_streaming.py <endpoint> --region us-east-2
    uv run e2e/e2e_test_streaming.py <endpoint> --list
    ... <endpoint> --scenarios basic,concurrent_5,speed

Judge by the PASS/FAIL lines. The SageMaker bidirectional-streaming transport
raises on nearly every teardown, so a logged transport exception is not by itself
a failure — every scenario's verdict comes from the audio and the protocol
messages.

What each scenario is actually guarding against, since "it returned bytes" is a
weak assertion for TTS:

  basic              the happy path, plus the RMS floor — a voice tensor that
                     fails to load can still return the right NUMBER of bytes,
                     of silence, which a byte-count check passes.
  turn_accounting    per-turn SpeechMetadata sums to SessionMetadata. This is the
                     client-side half of the billing check: metering counts the
                     same characters.
  multi_turn         two manual Flushes ⇒ exactly two SpeechMetadata, two
                     SpeechStarted, distinct speech_ids. Catches a server that
                     merges turns (which would under-report per-turn billing).
  incremental_speak  text fed as several Speak messages in one turn — the actual
                     voice-agent shape, and the path that exercises the internal
                     auto-flush boundaries.
  long_turn          a turn spanning many segments still ends with ONE
                     SpeechMetadata.
  speed              `speed` query param, a supported GA control. Renders the
                     same text at the default rate and at the fastest supported
                     rate (1.15) and requires the faster one to be measurably
                     shorter — accepting the param but ignoring it would
                     otherwise pass. (1.15 is the ceiling; 1.3 is rejected.)
  expressivity       `expressivity` query param (-2..2, GA, Beta behavior).
                     Asserts each end of the range plus the default is ACCEPTED
                     and yields non-silent audio. Exists because the param
                     shipped missing from the pricing catalog's known_params, so
                     the shim 400'd a documented feature the stem supports.
  expressivity_out_of_range
                     NEGATIVE control: expressivity=3 must be rejected, not
                     clamped — this is what establishes the accepted range.
  configure_speed    mid-stream `Configure{speed}` must come back as
                     ConfigureSuccess and the following turn must still produce
                     audio.
  interrupt          `Interrupt` mid-turn ⇒ SpeechInterrupted whose
                     text_spoken + text_remaining reconstruct the turn input.
  unknown_param      NEGATIVE control: a bogus query param must be REJECTED, not
                     ignored. A pass here means an unpriced parameter cannot slip
                     through and be served without being billed for.
  concurrent_5       5 simultaneous sessions. Flux TTS batches on the GPU, so
                     single-session success says nothing about the batched path.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e2e.e2e_test_common import (  # noqa: E402
    EXPRESSIVITY_MAX,
    EXPRESSIVITY_MIN,
    EXPRESSIVITY_OUT_OF_RANGE,
    LONG_TEXT,
    MAX_SPEED,
    REFERENCE_PHRASES,
    REFERENCE_TEXT,
    SAMPLE_RATE,
    audio_verdict,
    check_session_totals,
    print_summary_table,
    select_scenarios,
)
from flux_tts_client import (  # noqa: E402
    DEFAULT_VOICE,
    FluxTtsStream,
    ensure_env_credentials,
    bidi_endpoint_uri,
    make_client,
)

logger = logging.getLogger("flux-tts-e2e")


async def _one_turn(client, endpoint: str, voice: str, text, cid: int = 0, **params):
    """Open a session, speak `text` (str or list of chunks), flush, wait, close."""
    s = FluxTtsStream(client, endpoint, connection_id=cid)
    await s.start(voice=voice, **params)
    chunks = [text] if isinstance(text, str) else list(text)
    for c in chunks:
        await s.speak(c)
    await s.flush()
    await s.wait_for_flushed(timeout=30)
    await s.wait_for_turn(timeout=120)
    await s.finish()
    return s


def _base_row(name: str, s, started: float) -> dict:
    ok, note, stats = audio_verdict(bytes(s.audio_bytes), SAMPLE_RATE)
    if s.errors:
        ok, note = False, f"server error: {s.errors[0]}"
    return {
        "scenario": name,
        "ok": ok,
        "notes": note,
        "bytes": len(s.audio_bytes),
        "duration_s": stats.get("duration_s", 0.0),
        "rms": stats.get("rms", 0.0),
        "billable_chars": s.billable_chars,
        "elapsed_s": time.monotonic() - started,
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

async def sc_basic(client, ep, voice):
    t0 = time.monotonic()
    s = await _one_turn(client, ep, voice, REFERENCE_TEXT)
    row = _base_row("basic", s, t0)
    if row["ok"] and s.connected is None:
        row["ok"], row["notes"] = False, "no Connected frame"
    elif row["ok"]:
        row["notes"] = f"model={s.connected.get('model_name')} ttfa={s.ttfa_s:.2f}s"
    return row


async def sc_turn_accounting(client, ep, voice):
    t0 = time.monotonic()
    s = await _one_turn(client, ep, voice, REFERENCE_TEXT)
    row = _base_row("turn_accounting", s, t0)
    if not row["ok"]:
        return row
    if not s.turns:
        row["ok"], row["notes"] = False, "no SpeechMetadata for a flushed turn"
        return row
    mismatch = check_session_totals(s.turns, s.session_metadata)
    if mismatch:
        row["ok"], row["notes"] = False, f"totals mismatch: {mismatch}"
        return row

    # Assert on `input_character_count`, NOT `billable_character_count`.
    #
    # Measured 2026-08-10: `billable_character_count` comes back as 0 on every
    # `/v2/speak` streaming turn (and inside `SpeechInterrupted.metadata`) while
    # `input_character_count` is correct — filed separately. It does NOT affect
    # SageMaker billing, which counts characters independently and meters
    # `char_count * 45 / 1000` (verified against the metering emissions, e.g.
    # char_count=743 -> 33 units over the streaming transport), so this field
    # being zero costs nothing. Asserting on it would pin a known bug and mask a
    # real regression in the number billing DOES depend on.
    want = len(REFERENCE_TEXT)
    got_input = sum(t.get("input_character_count", 0) for t in s.turns)
    if not (0.5 * want <= got_input <= 1.5 * want):
        row["ok"] = False
        row["notes"] = f"input_character_count {got_input} implausible vs {want} sent"
        return row

    billable = s.billable_chars
    row["notes"] = f"turns={len(s.turns)} input={got_input}/{want}"
    if billable == 0:
        row["notes"] += " (upstream: billable_character_count=0)"
    return row


async def sc_multi_turn(client, ep, voice):
    t0 = time.monotonic()
    s = FluxTtsStream(client, ep)
    await s.start(voice=voice)
    for phrase in ("First turn, all done. ", "Second turn, also done."):
        await s.speak(phrase)
        await s.flush()
        await s.wait_for_flushed(timeout=30)
        await s.wait_for_turn(timeout=120)
    await s.finish()
    row = _base_row("multi_turn", s, t0)
    if not row["ok"]:
        return row
    if len(s.turns) != 2:
        row["ok"] = False
        row["notes"] = f"expected 2 SpeechMetadata, got {len(s.turns)}"
    elif len(set(s.speech_started)) < 2:
        row["ok"] = False
        row["notes"] = f"expected 2 distinct speech_ids, got {s.speech_started}"
    else:
        row["notes"] = f"turns={len(s.turns)} ids={len(set(s.speech_started))}"
    return row


async def sc_incremental_speak(client, ep, voice):
    t0 = time.monotonic()
    s = await _one_turn(client, ep, voice, REFERENCE_PHRASES)
    row = _base_row("incremental_speak", s, t0)
    if row["ok"]:
        row["notes"] = f"{len(REFERENCE_PHRASES)} Speak msgs, 1 turn"
    return row


async def sc_long_turn(client, ep, voice):
    t0 = time.monotonic()
    s = await _one_turn(client, ep, voice, LONG_TEXT)
    row = _base_row("long_turn", s, t0)
    if row["ok"] and len(s.turns) != 1:
        row["ok"] = False
        row["notes"] = f"expected 1 SpeechMetadata for one turn, got {len(s.turns)}"
    return row


async def sc_speed(client, ep, voice):
    """`speed` as a query param must be ACCEPTED *and* take effect.

    `speed` is a supported Flux TTS control, valid over **0.85-1.15**. Accepting
    the parameter is not enough on its own — a server that parsed it and ignored
    it would also return audio — so this synthesizes the SAME text twice, at the
    default rate and at the fastest supported rate, and requires the faster
    render to be measurably shorter.

    At 1.15 the expected duration ratio is ~1/1.15 = 0.87. The 0.95 gate is
    deliberately loose: the exact ratio depends on how the model redistributes
    pauses, so this asserts the direction and rough magnitude of the effect, not
    a precise multiplier.
    """
    t0 = time.monotonic()
    base = await _one_turn(client, ep, voice, REFERENCE_TEXT)
    fast = await _one_turn(client, ep, voice, REFERENCE_TEXT, speed=MAX_SPEED)
    row = _base_row("speed", fast, t0)
    if not row["ok"]:
        return row

    base_ok, _, base_stats = audio_verdict(bytes(base.audio_bytes), SAMPLE_RATE)
    if not base_ok:
        row["ok"] = False
        row["notes"] = "default-speed control render failed; cannot compare"
        return row

    base_s = base_stats.get("duration_s", 0.0)
    fast_s = row["duration_s"]
    if base_s <= 0 or fast_s <= 0:
        row["ok"] = False
        row["notes"] = f"unusable durations (default={base_s:.2f}s fast={fast_s:.2f}s)"
        return row

    ratio = fast_s / base_s
    row["notes"] = f"default={base_s:.2f}s speed{MAX_SPEED}={fast_s:.2f}s ratio={ratio:.2f}"
    if ratio > 0.95:
        row["ok"] = False
        row["notes"] += " — speed accepted but audio not shorter (ignored?)"
    return row


async def sc_expressivity(client, ep, voice):
    """`expressivity` must be ACCEPTED across its documented range on streaming too.

    This exists because the param shipped BROKEN: `expressivity` is a GA
    `/v2/speak` control that the container's stem fully implements, but it was
    missing from the pricing catalog's `tts.known_params`, and
    REJECT_UNKNOWN_PARAMS is on in every image — so the shim answered
    400 `unsupported_parameter` before stem ever saw it. Nothing caught that
    because nothing tested it.

    Over SageMaker bidi a pre-upgrade rejection is especially unfriendly: the 400
    happens before the WebSocket exists, so a client can simply hang rather than
    see an error (the same shape as the known cross-language-negative hang). That
    makes an explicit accept-check on this transport worth having separately from
    the batch one.

    Deliberately NOT asserted: that the audio differs audibly from the default.
    `expressivity` is Beta and non-default values raise the hallucination risk, so
    unlike `speed` — whose effect on duration is measurable — there is no stable
    audio property to gate on. Non-silent audio at each end of the range is.
    """
    t0 = time.monotonic()
    tried, failed = [], []
    last = None
    for value in (EXPRESSIVITY_MIN, 0, EXPRESSIVITY_MAX):
        try:
            res = await _one_turn(
                client, ep, voice, "Short expressivity probe.", expressivity=value
            )
            ok, note, _ = audio_verdict(bytes(res.audio_bytes), SAMPLE_RATE)
            tried.append(f"{value}:{'ok' if ok else note}")
            if not ok:
                failed.append(str(value))
            last = res
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "unsupported_parameter" in msg or "unsupported parameter" in msg.lower():
                tried.append(f"{value}:REJECTED-as-unknown-param")
            else:
                tried.append(f"{value}:{type(e).__name__}")
            failed.append(str(value))

    if last is None:
        return {
            "scenario": "expressivity",
            "ok": False,
            "notes": "every expressivity value failed — " + " ".join(tried),
            "bytes": 0,
            "duration_s": 0.0,
            "rms": 0.0,
            "chars": 0,
            "elapsed_s": time.monotonic() - t0,
        }

    row = _base_row("expressivity", last, t0)
    row["ok"] = not failed
    row["notes"] = (
        f"accepted {EXPRESSIVITY_MIN}..{EXPRESSIVITY_MAX}: " + " ".join(tried)
        if not failed
        else f"failed for {','.join(failed)} — " + " ".join(tried)
    )
    return row


async def sc_expressivity_out_of_range(client, ep, voice):
    """NEGATIVE control — `expressivity` outside -2..2 must be REJECTED on streaming.

    Pinning the rejection is what establishes the range `expressivity` relies on.
    A clamp would synthesize at a setting the caller never requested; silent
    acceptance would mean an unvalidated value reaching the model.

    Note the pass condition is "the session did not produce audio", not "an
    exception was raised": on this transport a pre-upgrade 400 can surface as a
    hang or a bare close rather than a clean error, so requiring a specific
    exception type would make this flaky for the wrong reason.
    """
    t0 = time.monotonic()
    rejected, detail = False, ""
    try:
        res = await _one_turn(
            client, ep, voice, "Short probe.", expressivity=EXPRESSIVITY_OUT_OF_RANGE
        )
        n = len(bytes(res.audio_bytes))
        rejected = n == 0
        detail = (
            f"ACCEPTED out-of-range expressivity={EXPRESSIVITY_OUT_OF_RANGE}, got {n}B"
            if n
            else "no audio (rejected)"
        )
    except Exception as e:  # noqa: BLE001
        rejected = True
        msg = str(e).upper()
        detail = "EXPRESSIVITY_OUT_OF_RANGE" if "OUT_OF_RANGE" in msg else f"{type(e).__name__}"
    return {
        "scenario": "expressivity_out_of_range",
        "ok": rejected,
        "notes": f"rejected as expected ({detail})" if rejected else detail,
        "bytes": 0,
        "duration_s": 0.0,
        "rms": 0.0,
        "chars": 0,
        "elapsed_s": time.monotonic() - t0,
    }


async def sc_configure_speed(client, ep, voice):
    """Mid-stream `Configure{speed}` must be ACKNOWLEDGED and not break the session.

    `Configure` is answered by the server once the engine reports back: a valid
    speed is held until the next turn boundary and acknowledged with
    `ConfigureSuccess`, an invalid one comes back as `ConfigureFailure`. Since
    speed is a supported control, the fastest supported rate must be accepted.

    PASS = ConfigureSuccess AND audio still flowed for the turn that follows it.
    A `ConfigureFailure` or a silent no-answer both fail: a server that never
    answers would leave a voice agent waiting forever.
    """
    t0 = time.monotonic()
    s = FluxTtsStream(client, ep)
    await s.start(voice=voice)
    await s.speak("This first part is at the default rate. ")
    await s.flush()
    await s.wait_for_turn(timeout=120)
    await s.configure(speed=MAX_SPEED)
    await asyncio.sleep(0.5)
    await s.speak("This second part follows the Configure message.")
    await s.flush()
    await s.wait_for_turn(timeout=120)
    await s.finish()
    row = _base_row("configure_speed", s, t0)
    if not row["ok"]:
        return row
    if s.configure_ok:
        row["notes"] = f"ConfigureSuccess: applied={s.configure_ok[-1]}"
    elif s.configure_failed:
        row["ok"] = False
        row["notes"] = f"ConfigureFailure for speed={MAX_SPEED}: {s.configure_failed[0].get('code')}"
    elif s.errors:
        row["ok"] = False
        row["notes"] = f"server Error: {s.errors[0][:50]}"
    else:
        row["ok"] = False
        row["notes"] = "no ConfigureSuccess/ConfigureFailure/Error — server never answered"
    return row


async def sc_interrupt(client, ep, voice):
    t0 = time.monotonic()
    s = FluxTtsStream(client, ep)
    await s.start(voice=voice)
    await s.speak(LONG_TEXT)
    await s.flush()
    # Let some audio land, then barge in mid-turn.
    for _ in range(100):
        if s.audio_frames > 0:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)
    # No speech_id: `Interrupt` carries `playback_offset` and nothing else, and it
    # rejects unknown fields — it always targets the active turn. Sending one made
    # the whole frame unparseable (`[MESSAGE-0000]`).
    await s.interrupt(playback_offset_ms=500)
    await s.wait_for_turn(timeout=60)
    await s.finish()
    row = _base_row("interrupt", s, t0)
    if not row["ok"]:
        return row
    if not s.interrupted:
        row["ok"], row["notes"] = False, "no SpeechInterrupted after Interrupt"
        return row
    ev = s.interrupted[0]
    spoken = ev.get("text_spoken", "")
    remaining = ev.get("text_remaining", "")
    if not remaining:
        # An empty `text_remaining` is only legitimate when no `playback_offset`
        # was supplied (the server then declines to report spoken/remaining at
        # all). We DO supply one, so empty means the accounting did not populate.
        #
        # Not treated as a hard failure only because `text_remaining` can also be
        # legitimately empty if the interrupt lands after synthesis finished. The
        # notes carry the offset so the two are distinguishable by eye.
        #
        # Historical note: this used to report an "upstream barge-in accounting"
        # gap, on the evidence that `audio_played_ms` came back 0 with both text
        # fields empty. That was self-inflicted — the client was attaching a
        # `speech_id` to `Interrupt`, which rejects unknown fields, so the frame
        # never parsed. With the field removed the accounting populates correctly.
        row["notes"] = (
            f"SpeechInterrupted with no text_remaining (played={ev.get('audio_played_ms')}ms)"
        )
        return row
    # spoken + remaining should reconstruct the turn input modulo the server's
    # text cleaning, so compare on a whitespace-insensitive basis.
    joined = "".join((spoken + remaining).split())
    want = "".join(LONG_TEXT.split())
    if joined != want:
        row["ok"] = False
        row["notes"] = (
            f"spoken+remaining != input ({len(joined)} vs {len(want)} chars)"
        )
    else:
        row["notes"] = f"played={ev.get('audio_played_ms')}ms spoken={len(spoken)}ch"
    return row


async def sc_unknown_param(client, ep, voice):
    """NEGATIVE control — a param outside the catalog allowlist must be rejected.

    Passing means we get an error — a 400 `unsupported_parameter` for a param
    with no rate, or a 400 for a param `/v2/speak` does not define — surfacing
    either as a handshake failure or a server Error frame. FAILING here means an
    unknown param was silently accepted, which is the silent-under-billing
    condition the check exists to prevent.
    """
    t0 = time.monotonic()
    rejected = False
    detail = ""
    try:
        s = await _one_turn(
            client, ep, voice, "Short probe.", **{"definitely_not_a_param": "1"}
        )
        if s.errors:
            rejected, detail = True, s.errors[0][:60]
        elif not s.audio_bytes:
            rejected, detail = True, "no audio, connection refused/closed"
    except Exception as e:  # noqa: BLE001 — a handshake 400 arrives as an exception
        rejected, detail = True, type(e).__name__
    return {
        "scenario": "unknown_param",
        "ok": rejected,
        "notes": (
            f"rejected as expected ({detail})"
            if rejected
            else "UNKNOWN PARAM WAS ACCEPTED — reject gate is not working"
        ),
        "bytes": 0,
        "duration_s": 0.0,
        "rms": 0.0,
        "billable_chars": 0,
        "elapsed_s": time.monotonic() - t0,
    }


async def sc_concurrent_5(client, ep, voice):
    t0 = time.monotonic()
    results = await asyncio.gather(
        *(
            _one_turn(client, ep, voice, REFERENCE_TEXT, cid=i)
            for i in range(5)
        ),
        return_exceptions=True,
    )
    good, notes = 0, []
    total_bytes = 0
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            notes.append(f"c{i}:{type(r).__name__}")
            continue
        ok, note, _ = audio_verdict(bytes(r.audio_bytes), SAMPLE_RATE)
        total_bytes += len(r.audio_bytes)
        if ok and not r.errors:
            good += 1
        else:
            notes.append(f"c{i}:{note or r.errors[0][:30]}")
    return {
        "scenario": "concurrent_5",
        "ok": good == 5,
        "notes": f"{good}/5 ok" + ("; " + ", ".join(notes[:3]) if notes else ""),
        "bytes": total_bytes,
        "duration_s": 0.0,
        "rms": 0.0,
        "billable_chars": 0,
        "elapsed_s": time.monotonic() - t0,
    }


SCENARIOS = {
    "basic": sc_basic,
    "turn_accounting": sc_turn_accounting,
    "multi_turn": sc_multi_turn,
    "incremental_speak": sc_incremental_speak,
    "long_turn": sc_long_turn,
    "speed": sc_speed,
    "expressivity": sc_expressivity,
    "configure_speed": sc_configure_speed,
    "interrupt": sc_interrupt,
    "concurrent_5": sc_concurrent_5,
    # --- negatives LAST. A rejected request yields no NATS usage payload, so the
    # shim fails it closed and records an `http_nats_grace_window` health trigger;
    # three inside the rolling window flip the endpoint to degraded and every
    # later scenario fails with an unrelated 424. Measured on the batch battery
    # 2026-08-13, where two added negatives did exactly that. Ordering the
    # negatives last makes the cost harmless, since nothing runs after them and
    # they pass on either a clean rejection or a fail-closed one.
    "unknown_param": sc_unknown_param,
    "expressivity_out_of_range": sc_expressivity_out_of_range,
}


async def main_async(args) -> int:
    ensure_env_credentials(args.region)
    client = make_client(args.region, args.fips)
    # Printed, not asserted — the log is the proof of which endpoint was used.
    print(f"Bidi URL:    {bidi_endpoint_uri(args.region, args.fips)}")
    print(f"FIPS:        {'yes (--fips)' if args.fips else 'no'}")
    names = select_scenarios(list(SCENARIOS), args.scenarios)
    rows = []
    for name in names:
        print(f"\n--- {name} ---", flush=True)
        try:
            rows.append(await SCENARIOS[name](client, args.endpoint, args.voice))
        except Exception as e:  # noqa: BLE001
            logger.exception("scenario %s raised", name)
            rows.append(
                {
                    "scenario": name,
                    "ok": False,
                    "notes": f"raised {type(e).__name__}: {e}"[:80],
                    "bytes": 0,
                    "duration_s": 0.0,
                    "rms": 0.0,
                    "billable_chars": 0,
                    "elapsed_s": 0.0,
                }
            )
    print()
    _, failed = print_summary_table(rows)
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("endpoint", nargs="?", help="SageMaker endpoint name")
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"default: {DEFAULT_VOICE}")
    p.add_argument("--scenarios", default=None, help="comma-separated subset")
    p.add_argument("--fips", action="store_true",
                   help="Route AWS calls to the FIPS 140-3 endpoints (bidi streaming on "
                        "runtime-fips.sagemaker.<region>.amazonaws.com:8443). OFF by "
                        "default. Selects AWS endpoints only — says nothing about the "
                        "container's own crypto.")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.list:
        for n in SCENARIOS:
            print(n)
        return 0
    if not args.endpoint:
        p.error("endpoint is required unless --list")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

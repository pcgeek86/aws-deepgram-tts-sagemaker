#!/usr/bin/env python3
"""Flux TTS BATCH e2e battery — `POST /invocations` → `POST /v2/speak`.

    AWS_PROFILE=shared-dev uv run e2e/e2e_test_batch.py <endpoint> --region us-east-2
    uv run e2e/e2e_test_batch.py <endpoint> --list

Flux TTS serves batch and streaming off the SAME `streaming` image (the
bidirectional-streaming LABEL declares the WebSocket capability but does not
preclude `/invocations`), so this battery runs against the same endpoint as
`e2e_test_streaming.py` — there is no separate batch endpoint to deploy.

The target path comes from `CustomAttributes`, not the URL: the `/invocations`
handler reads `x-amzn-sagemaker-custom-attributes` as `<path>?<query>` and
proxies there, defaulting to `v1/listen` when the header is absent. So a batch
TTS call with no CustomAttributes is a 400 about audio, which reads like a broken
endpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config as BotoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e2e.e2e_test_common import (  # noqa: E402
    EXPRESSIVITY_FRACTIONAL,
    EXPRESSIVITY_MAX,
    EXPRESSIVITY_MIN,
    EXPRESSIVITY_OUT_OF_RANGE,
    LONG_TEXT,
    MAX_SPEED,
    REFERENCE_TEXT,
    SAMPLE_RATE,
    audio_verdict,
    print_summary_table,
    select_scenarios,
)
from flux_tts_client import DEFAULT_VOICE, invoke_batch  # noqa: E402

logger = logging.getLogger("flux-tts-e2e-batch")


def _row(name: str, audio: bytes, meta: dict, started: float, extra: str = "") -> dict:
    ok, note, stats = audio_verdict(audio, SAMPLE_RATE)
    return {
        "scenario": name,
        "ok": ok,
        "notes": note or extra,
        "bytes": len(audio),
        "duration_s": stats.get("duration_s", 0.0),
        "rms": stats.get("rms", 0.0),
        "billable_chars": "",
        "elapsed_s": time.monotonic() - started,
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def sc_basic(rt, ep, voice):
    t0 = time.monotonic()
    audio, meta = invoke_batch(rt, ep, REFERENCE_TEXT, voice)
    return _row("basic", audio, meta, t0, extra=f"ct={meta['content_type']}")


def sc_long_text(rt, ep, voice):
    t0 = time.monotonic()
    audio, meta = invoke_batch(rt, ep, LONG_TEXT, voice)
    return _row("long_text", audio, meta, t0)


def sc_speed(rt, ep, voice):
    """`speed` must be ACCEPTED *and* take effect on the batch surface too.

    `speed` is a supported Flux TTS control, valid over **0.85–1.15** (see
    `speed_out_of_range`). Accepting the parameter is not a strong enough
    assertion on its own — a server that parsed it and ignored it would also
    return audio — so this synthesizes the SAME text at the default rate and at
    the fastest supported rate and requires the faster render to be measurably
    shorter.

    At 1.15 the expected duration ratio is ~1/1.15 ≈ 0.87. The 0.95 gate is
    deliberately loose: the exact ratio depends on how the model redistributes
    pauses, so this asserts the direction and rough magnitude of the effect, not
    a precise multiplier.
    """
    t0 = time.monotonic()
    try:
        base_audio, _ = invoke_batch(rt, ep, REFERENCE_TEXT, voice)
        fast_audio, fast_meta = invoke_batch(rt, ep, REFERENCE_TEXT, voice, speed=MAX_SPEED)
    except Exception as e:  # noqa: BLE001
        return {
            "scenario": "speed",
            "ok": False,
            "notes": f"speed={MAX_SPEED} rejected: {type(e).__name__}: {str(e)[:60]}",
            "bytes": 0,
            "duration_s": 0.0,
            "rms": 0.0,
            "billable_chars": "",
            "elapsed_s": time.monotonic() - t0,
        }

    row = _row("speed", fast_audio, fast_meta, t0)
    if not row["ok"]:
        return row

    base_ok, _, base_stats = audio_verdict(base_audio, SAMPLE_RATE)
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


def sc_speed_out_of_range(rt, ep, voice):
    """NEGATIVE control — `speed` outside 0.85–1.15 must be REJECTED, not clamped.

    Measured 2026-08-13: `speed=1.3` returns 400 `'speed' must be between 0.85
    and 1.15.` Clamping instead would silently synthesize at a rate the caller
    did not ask for, so the rejection is the desirable behavior and worth
    pinning — it is also what tells us the accepted range, which `speed` above
    depends on.
    """
    t0 = time.monotonic()
    rejected, detail = False, ""
    try:
        audio, _ = invoke_batch(rt, ep, "Short probe.", voice, speed=1.3)
        detail = f"ACCEPTED out-of-range speed, returned {len(audio)}B" if audio else "empty body"
        rejected = not audio
    except Exception as e:  # noqa: BLE001
        rejected = True
        detail = "must be between 0.85 and 1.15" if "0.85" in str(e) else f"{type(e).__name__}"
    return {
        "scenario": "speed_out_of_range",
        "ok": rejected,
        "notes": f"rejected as expected ({detail})" if rejected else detail,
        "bytes": 0,
        "duration_s": 0.0,
        "rms": 0.0,
        "billable_chars": "",
        "elapsed_s": time.monotonic() - t0,
    }


def sc_expressivity(rt, ep, voice):
    """`expressivity` must be ACCEPTED across its whole documented range.

    This scenario exists because the param was *shipped broken*: `expressivity`
    is a GA `/v2/speak` control that the container's stem fully implements, but it
    was missing from the pricing catalog's `tts.known_params`, and
    REJECT_UNKNOWN_PARAMS is on in every image — so the shim returned
    400 `unsupported_parameter` before the request ever reached stem. Nothing
    caught it because nothing tested it. This is that test.

    Both ends of the range plus the default are exercised, since the failure mode
    being guarded against is an allowlist/range mismatch rather than anything
    audio-specific. Deliberately NOT asserted: that the audio sounds different
    from the default. `expressivity` is Beta and non-default values raise the
    hallucination risk, so "is it more animated" is not a stable gate — unlike
    `speed`, whose effect on duration is directly measurable.
    """
    t0 = time.monotonic()
    tried, failed = [], []
    last_audio, last_meta = b"", {}
    for value in (EXPRESSIVITY_MIN, 0, EXPRESSIVITY_MAX):
        try:
            audio, meta = invoke_batch(rt, ep, "Short expressivity probe.", voice, expressivity=value)
            ok, note, _ = audio_verdict(audio, SAMPLE_RATE)
            tried.append(f"{value}:{'ok' if ok else note}")
            if not ok:
                failed.append(str(value))
            last_audio, last_meta = audio, meta
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # Name the original bug explicitly: this is what a catalog missing
            # `expressivity` looks like from the client side.
            if "unsupported_parameter" in msg or "unsupported parameter" in msg.lower():
                tried.append(f"{value}:REJECTED-as-unknown-param")
            else:
                tried.append(f"{value}:{type(e).__name__}")
            failed.append(str(value))

    row = _row("expressivity", last_audio, last_meta, t0)
    row["ok"] = not failed
    row["notes"] = (
        f"accepted {EXPRESSIVITY_MIN}..{EXPRESSIVITY_MAX}: " + " ".join(tried)
        if not failed
        else f"failed for {','.join(failed)} — " + " ".join(tried)
    )
    return row


def sc_expressivity_out_of_range(rt, ep, voice):
    """NEGATIVE control — `expressivity` outside -2..2 must be rejected.

    Documented as a 400 `EXPRESSIVITY_OUT_OF_RANGE`. Pinning the rejection is what
    establishes the accepted range that `expressivity` above relies on; a server
    that clamped instead would synthesize at a setting the caller never asked for.
    """
    t0 = time.monotonic()
    rejected, detail = False, ""
    try:
        audio, _ = invoke_batch(
            rt, ep, "Short probe.", voice, expressivity=EXPRESSIVITY_OUT_OF_RANGE
        )
        detail = (
            f"ACCEPTED out-of-range expressivity={EXPRESSIVITY_OUT_OF_RANGE}, "
            f"returned {len(audio)}B"
        ) if audio else "empty body"
        rejected = not audio
    except Exception as e:  # noqa: BLE001
        rejected = True
        msg = str(e)
        detail = "EXPRESSIVITY_OUT_OF_RANGE" if "OUT_OF_RANGE" in msg.upper() else f"{type(e).__name__}"
    return {
        "scenario": "expressivity_out_of_range",
        "ok": rejected,
        "notes": f"rejected as expected ({detail})" if rejected else detail,
        "bytes": 0,
        "duration_s": 0.0,
        "rms": 0.0,
        "billable_chars": "",
        "elapsed_s": time.monotonic() - t0,
    }


def sc_expressivity_fractional(rt, ep, voice):
    """NEGATIVE control — a FRACTIONAL `expressivity` must be rejected.

    Separate from `expressivity_out_of_range` on purpose: 1.5 is numerically
    *inside* -2..2, so it exercises a different validation path and a different
    documented error (`EXPRESSIVITY_INCREMENT_INVALID`). stem parses the param as
    `f64` rather than `i8` specifically so a fractional value reaches this check
    instead of failing as an opaque deserialization error, which is a behavior
    worth keeping honest.
    """
    t0 = time.monotonic()
    rejected, detail = False, ""
    try:
        audio, _ = invoke_batch(
            rt, ep, "Short probe.", voice, expressivity=EXPRESSIVITY_FRACTIONAL
        )
        detail = (
            f"ACCEPTED fractional expressivity={EXPRESSIVITY_FRACTIONAL}, "
            f"returned {len(audio)}B"
        ) if audio else "empty body"
        rejected = not audio
    except Exception as e:  # noqa: BLE001
        rejected = True
        msg = str(e).upper()
        if "INCREMENT_INVALID" in msg:
            detail = "EXPRESSIVITY_INCREMENT_INVALID"
        elif "OUT_OF_RANGE" in msg:
            # Still a rejection, but via the wrong code — worth surfacing rather
            # than hiding behind a green row.
            detail = "rejected but as OUT_OF_RANGE, expected INCREMENT_INVALID"
        else:
            detail = f"{type(e).__name__}"
    return {
        "scenario": "expressivity_fractional",
        "ok": rejected,
        "notes": f"rejected as expected ({detail})" if rejected else detail,
        "bytes": 0,
        "duration_s": 0.0,
        "rms": 0.0,
        "billable_chars": "",
        "elapsed_s": time.monotonic() - t0,
    }


def sc_wav_container(rt, ep, voice):
    """`container=wav` — a batch-only param (streaming rejects it), so this also
    confirms the batch query surface really is the batch one."""
    t0 = time.monotonic()
    audio, meta = invoke_batch(rt, ep, REFERENCE_TEXT, voice, container="wav")
    row = _row("wav_container", audio, meta, t0)
    if row["ok"]:
        from e2e.e2e_test_common import sniff_container

        got = sniff_container(audio)
        if got != "wav":
            row["ok"] = False
            row["notes"] = f"asked for container=wav, got {got!r}"
        else:
            row["notes"] = "wav header present"
    return row


def sc_unknown_param(rt, ep, voice):
    """NEGATIVE control — see the streaming battery's note. A bogus param must be
    a 400, not silently accepted."""
    t0 = time.monotonic()
    rejected, detail = False, ""
    try:
        audio, _ = invoke_batch(
            rt, ep, "Short probe.", voice, **{"definitely_not_a_param": "1"}
        )
        if not audio:
            rejected, detail = True, "empty body"
    except Exception as e:  # noqa: BLE001
        rejected, detail = True, f"{type(e).__name__}: {str(e)[:50]}"
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
        "billable_chars": "",
        "elapsed_s": time.monotonic() - t0,
    }


def sc_aura_model_rejected(rt, ep, voice):
    """NEGATIVE control — `/v2/speak` serves Flux voices only. An Aura model must
    come back as an endpoint-specific error, not as audio from some fallback
    voice."""
    t0 = time.monotonic()
    rejected, detail = False, ""
    try:
        audio, _ = invoke_batch(rt, ep, "Short probe.", "aura-2-thalia-en")
        if not audio:
            rejected, detail = True, "empty body"
    except Exception as e:  # noqa: BLE001
        rejected, detail = True, f"{type(e).__name__}"
    return {
        "scenario": "aura_model_rejected",
        "ok": rejected,
        "notes": (
            f"rejected as expected ({detail})"
            if rejected
            else "aura-2 model was SERVED on /v2/speak — wrong-tier billing risk"
        ),
        "bytes": 0,
        "duration_s": 0.0,
        "rms": 0.0,
        "billable_chars": "",
        "elapsed_s": time.monotonic() - t0,
    }


def sc_concurrent_5(rt, ep, voice):
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [
            ex.submit(invoke_batch, rt, ep, REFERENCE_TEXT, voice) for _ in range(5)
        ]
        good, notes, total = 0, [], 0
        for i, f in enumerate(futs):
            try:
                audio, _ = f.result()
                total += len(audio)
                ok, note, _ = audio_verdict(audio, SAMPLE_RATE)
                if ok:
                    good += 1
                else:
                    notes.append(f"r{i}:{note}")
            except Exception as e:  # noqa: BLE001
                notes.append(f"r{i}:{type(e).__name__}")
    return {
        "scenario": "concurrent_5",
        "ok": good == 5,
        "notes": f"{good}/5 ok" + ("; " + ", ".join(notes[:3]) if notes else ""),
        "bytes": total,
        "duration_s": 0.0,
        "rms": 0.0,
        "billable_chars": "",
        "elapsed_s": time.monotonic() - t0,
    }


# ORDER MATTERS, and not for the usual reasons. Every scenario whose request is
# rejected BY STEM costs the endpoint one health trigger: stem publishes no NATS
# usage payload for a request it refused, so the shim waits out its 200 ms grace
# window, fails the request closed (5xx, "returned without metering emission") and
# records an `http_nats_grace_window` trigger. THREE of those inside the rolling
# window flip the endpoint to degraded — `/ping` goes 503, SageMaker stops routing,
# and every scenario after that fails with a 424 that has nothing to do with what
# it was testing.
#
# Measured 2026-08-13: adding expressivity_out_of_range + expressivity_fractional
# took this battery from 2 stem-rejected requests to 4 and tripped exactly that.
# concurrent_5 and the entire streaming battery that followed failed, which read
# like a product regression and was not one.
#
# So: every stem-rejecting negative runs LAST, after everything that needs a
# healthy endpoint. Degradation is recoverable, so paying for it at the very end
# costs nothing — the negatives themselves still pass either way, since a
# fail-closed 5xx and a clean stem 400 are both "rejected" as far as they assert.
#
# `unknown_param` is deliberately NOT in that group: the shim rejects an
# off-allowlist param BEFORE proxying, so no NATS wait happens and it is free.
#
# If you run this battery and the streaming one back-to-back against ONE endpoint,
# leave a recovery gap between them, or the streaming run inherits the degraded
# state this battery's tail induces.
SCENARIOS = {
    # --- positives + load: need a healthy endpoint ---
    "basic": sc_basic,
    "long_text": sc_long_text,
    "speed": sc_speed,
    "expressivity": sc_expressivity,
    "wav_container": sc_wav_container,
    "concurrent_5": sc_concurrent_5,
    # --- free negative: rejected by the shim pre-proxy, no NATS wait ---
    "unknown_param": sc_unknown_param,
    # --- stem-rejecting negatives: each costs one degraded trigger, so LAST ---
    "speed_out_of_range": sc_speed_out_of_range,
    "expressivity_out_of_range": sc_expressivity_out_of_range,
    "expressivity_fractional": sc_expressivity_fractional,
    "aura_model_rejected": sc_aura_model_rejected,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("endpoint", nargs="?")
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--scenarios", default=None)
    p.add_argument("--fips", action="store_true",
                   help="Route AWS calls to the FIPS 140-3 endpoints (runtime-fips). OFF "
                        "by default. Selects AWS endpoints only — says nothing about the "
                        "container's own crypto.")
    p.add_argument("--list", action="store_true")
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

    rt = boto3.client("sagemaker-runtime", region_name=args.region,
                      config=BotoConfig(use_fips_endpoint=True) if args.fips else None)
    print(f"Runtime URL: {rt.meta.endpoint_url}")
    print(f"FIPS:        {'yes' if '-fips.' in rt.meta.endpoint_url else 'no'}")
    rows = []
    for name in select_scenarios(list(SCENARIOS), args.scenarios):
        print(f"\n--- {name} ---", flush=True)
        try:
            rows.append(SCENARIOS[name](rt, args.endpoint, args.voice))
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
                    "billable_chars": "",
                    "elapsed_s": 0.0,
                }
            )
    print()
    _, failed = print_summary_table(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

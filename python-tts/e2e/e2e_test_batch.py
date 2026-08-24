#!/usr/bin/env python3
"""
End-to-end correctness + feature-coverage test for a batch (REST) SageMaker TTS endpoint.

Calls `sagemaker-runtime.invoke_endpoint` synchronously with the Deepgram
`/v1/speak` path + query in `CustomAttributes` and a JSON `{"text": "..."}`
body — the synchronous mirror of how the STT batch driver invokes `/v1/listen`.
The response Body is the synthesized audio; each scenario validates that audio
**self-contained** (no second endpoint): non-empty, correct container/codec,
non-silent for linear16, the requested sample rate, and — for speed control —
that duration changes monotonically with `speed`.

Feature coverage per the Deepgram TTS docs (June 2026 audit):
  model (voice), encoding (linear16/mp3/flac/mulaw/alaw/opus/aac), sample_rate,
  bit_rate (mp3/opus/aac), container (none/wav/ogg), speed (0.7–1.5),
  inline IPA pronunciation override, the 2000-char text limit, mip_opt_out, tag.
See https://developers.deepgram.com/docs/tts-media-output-settings and
https://developers.deepgram.com/docs/tts-voice-controls

Pass / fail
-----------
Most scenarios PASS when the invoke succeeds and the returned audio satisfies
the scenario's assertions (bytes / container / sample_rate / non-silent RMS).
Two special kinds:
  - `expect_failure=True` (text over the 2000-char limit) PASSES when the
    endpoint rejects the request (e.g. HTTP 413).
  - `tolerated_error_substring` PASS-WITH-NOTE when the endpoint returns a known
    "not supported by this bundle" error (e.g. a voice/encoding the deployed
    model doesn't serve) — surfaces the deployment gap without false-failing.

Endpoint prerequisites
----------------------
- A pre-recorded / REST-capable SageMaker TTS endpoint is InService and
  reachable. The endpoint serves `/v1/speak` via the CustomAttributes path.

Usage
-----
    uv run e2e/e2e_test_batch.py your-tts-endpoint --region us-east-2
    uv run e2e/e2e_test_batch.py --list
    uv run e2e/e2e_test_batch.py your-tts-endpoint --scenarios basic,encoding_mp3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlparse

import boto3
from botocore.config import Config as BotoConfig

# Set once from --fips in main(). Applied PER CLIENT rather than by exporting
# AWS_USE_FIPS_ENDPOINT, which would also redirect the IAM Identity Center portal
# botocore uses to mint SSO credentials — portal.sso-fips.<region>.amazonaws.com
# does not exist, so a process-wide switch kills credential resolution before the
# run reaches SageMaker at all (a warm credential cache hides it, making the
# failure look intermittent).
_FIPS = False


def _client(session, service, region=None):
    """Build a boto3 client with the run's FIPS endpoint setting applied."""
    kwargs = {"config": BotoConfig(use_fips_endpoint=True)} if _FIPS else {}
    if region is not None:
        kwargs["region_name"] = region
    return session.client(service, **kwargs)
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test_common import (
    IPA_TEXT,
    SILENCE_RMS_FLOOR,
    SUPPORTED_LANGUAGES,
    alt_language_voice,
    default_voice,
    featured_voices,
    language_supported,
    linear16_duration_and_rms,
    parse_wav,
    print_summary_table,
    reference_text,
    sniff_container,
    voice_language,
)

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-2"
DEFAULT_LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class TTSScenario:
    """One batch TTS scenario.

    `params` are appended as `/v1/speak` query parameters. The validation
    fields below describe what the returned audio must satisfy:

      - `expect_container`: required magic-byte container ('wav'|'ogg'|'flac'|
        'mpeg'|'raw').
      - `check_rms`: require non-silent linear16 audio (RMS above the floor).
      - `expect_sample_rate`: required WAV sample rate (linear16+wav only).
      - `speed_compare`: special — synthesize the same text at 0.7 / 1.0 / 1.5
        and assert duration decreases monotonically as speed increases.
      - `expect_failure`: the request is expected to be rejected (negative test).
      - `tolerated_error_substring`: PASS-WITH-NOTE on this known error.
    """
    name: str
    description: str
    text: str | None = None
    voice: str | None = None
    params: dict[str, str] = field(default_factory=dict)
    concurrency: int = 1
    requests: int = 1
    expect_container: str | None = None
    check_rms: bool = False
    expect_sample_rate: int | None = None
    min_bytes: int = 64
    speed_compare: bool = False
    expect_failure: bool = False
    tolerated_error_substring: str | None = None
    # None = no restriction (default). Set ONLY when the endpoint hard-rejects
    # outside this language set — see `language_supported()` in
    # e2e_test_common.
    supported_languages: list[str] | None = None
    notes: str = ""


def default_scenarios(language: str, voice_coverage_n: int = 3) -> list[TTSScenario]:
    """Build the per-language scenario set (see streaming module docstring for
    the multi-voice coverage rationale)."""
    scenarios: list[TTSScenario] = [
        # ---- Coverage / load ----
        TTSScenario(
            name="basic",
            description="linear16/wav baseline — non-empty, non-silent audio",
            params={"encoding": "linear16", "container": "wav"},
            expect_container="wav",
            check_rms=True,
        ),
        TTSScenario(
            name="default_format",
            description="no encoding param — records the server's default output format",
            min_bytes=2000,
            notes="documents the REST default (the docs conflict mp3 vs linear16); see container= in notes",
        ),
        TTSScenario(
            name="concurrent_5",
            description="5 concurrent synthesize requests (linear16/wav)",
            params={"encoding": "linear16", "container": "wav"},
            concurrency=5,
            requests=5,
            check_rms=True,
        ),
        # ---- Encoding / container matrix ----
        TTSScenario(
            name="encoding_linear16_wav",
            description="encoding=linear16 container=wav (explicit)",
            params={"encoding": "linear16", "container": "wav"},
            expect_container="wav",
            check_rms=True,
        ),
        TTSScenario(
            name="encoding_linear16_raw",
            description="encoding=linear16 container=none (bare PCM)",
            params={"encoding": "linear16", "container": "none"},
            expect_container="raw",
            check_rms=True,
            notes="no RIFF header; RMS computed on raw PCM",
        ),
        TTSScenario(
            name="encoding_mp3",
            description="encoding=mp3 (MPEG audio)",
            params={"encoding": "mp3"},
            expect_container="mpeg",
            tolerated_error_substring="encoding",
        ),
        TTSScenario(
            name="encoding_flac",
            description="encoding=flac",
            params={"encoding": "flac"},
            expect_container="flac",
            tolerated_error_substring="encoding",
        ),
        TTSScenario(
            name="encoding_opus_ogg",
            description="encoding=opus (Ogg-wrapped)",
            params={"encoding": "opus"},
            expect_container="ogg",
            tolerated_error_substring="encoding",
        ),
        TTSScenario(
            name="encoding_mulaw_wav",
            description="encoding=mulaw sample_rate=8000 container=wav (telephony)",
            params={"encoding": "mulaw", "sample_rate": "8000", "container": "wav"},
            expect_container="wav",
            tolerated_error_substring="encoding",
            notes="8-bit companded; RMS skipped",
        ),
        TTSScenario(
            name="encoding_aac",
            description="encoding=aac (raw AAC stream — bytes-only check)",
            params={"encoding": "aac"},
            min_bytes=2000,
            tolerated_error_substring="encoding",
            notes="AAC is returned as a raw codec stream (no container magic bytes)",
        ),
        # ---- Sample rate ----
        TTSScenario(
            name="sample_rate_48000",
            description="encoding=linear16 sample_rate=48000 container=wav",
            params={"encoding": "linear16", "sample_rate": "48000", "container": "wav"},
            expect_container="wav",
            expect_sample_rate=48000,
            check_rms=True,
        ),
        TTSScenario(
            name="sample_rate_16000",
            description="encoding=linear16 sample_rate=16000 container=wav",
            params={"encoding": "linear16", "sample_rate": "16000", "container": "wav"},
            expect_container="wav",
            expect_sample_rate=16000,
            check_rms=True,
        ),
        # ---- Bit rate ----
        TTSScenario(
            name="bit_rate_mp3_32000",
            description="encoding=mp3 bit_rate=32000",
            params={"encoding": "mp3", "bit_rate": "32000"},
            expect_container="mpeg",
            tolerated_error_substring="encoding",
        ),
        # ---- Voice controls ----
        TTSScenario(
            name="speed_duration",
            description="speed 0.7 vs 1.0 vs 1.5 — duration must shrink as speed rises",
            params={"encoding": "linear16", "container": "wav"},
            speed_compare=True,
            tolerated_error_substring="bad request",
            notes="duration ∝ 1/speed when supported; older bundles may 400 (PASS-WITH-NOTE)",
        ),
        TTSScenario(
            name="pronunciation_ipa",
            description="inline IPA pronunciation override (Aura-2 voice control)",
            text=IPA_TEXT,
            params={"encoding": "linear16", "container": "wav"},
            check_rms=True,
            tolerated_error_substring="inline control",
            notes="well-formed inline IPA (escaped braces); produces audio on bundles "
                  "that support inline controls, else MODEL_DOES_NOT_SUPPORT_INLINE_CONTROLS "
                  "(PASS-WITH-NOTE)",
        ),
        # ---- Limits (negative) ----
        TTSScenario(
            name="text_limit_exceeded",
            description="text > 2000 chars — expect rejection (HTTP 413)",
            text=("Deepgram text to speech stress test sentence number. " * 60),  # ~3100 chars
            expect_failure=True,
            notes="2000-char per-request limit",
        ),
        # ---- Passthrough flags (smoke) ----
        TTSScenario(
            name="mip_opt_out",
            description="mip_opt_out=true (model-improvement opt-out; smoke)",
            params={"mip_opt_out": "true"},
            check_rms=True,
        ),
        TTSScenario(
            name="tag",
            description="tag=e2e (usage-reporting label; smoke)",
            params={"tag": "e2e"},
            check_rms=True,
        ),
    ]

    # Per-language multi-voice coverage — one row per featured voice (after the
    # default voice, which is already exercised by `basic`). Each must produce
    # non-silent linear16 audio; a single FAIL points at a specific missing or
    # broken voice in the deployed bundle.
    voices = featured_voices(language, voice_coverage_n)
    default = default_voice(language)
    for v in voices:
        if v.model == default:
            continue
        scenarios.append(TTSScenario(
            name=f"voice_{v.model}",
            description=f"voice={v.model} ({v.accent} {v.gender}) — bundle must serve this {v.language} voice",
            voice=v.model,
            params={"encoding": "linear16", "container": "wav"},
            expect_container="wav",
            check_rms=True,
            notes=f"language-coverage probe for {v.language}",
        ))

    # Cross-language negative — confirms the bundle is monolingual.
    alt = alt_language_voice(language)
    if alt is not None:
        scenarios.append(TTSScenario(
            name="voice_wrong_language",
            description=f"voice={alt.model} ({alt.language}) — expected to error on a {language}-only bundle",
            voice=alt.model,
            params={"encoding": "linear16", "container": "wav"},
            tolerated_error_substring="model",
            notes=f"monolingual-bundle negative; PASS-WITH-NOTE on '{alt.language}' voice rejection",
        ))

    # Aura-1 legacy voice check is only meaningful for English bundles (Aura-1
    # is English-only) — always included so a non-English run shows a visible
    # SKIP row (see `language_supported` in e2e_test_common) instead of the
    # scenario silently not existing in the list.
    scenarios.append(TTSScenario(
        name="voice_aura1_asteria",
        description="model=aura-asteria-en (legacy Aura-1 default voice)",
        voice="aura-asteria-en",
        params={"encoding": "linear16", "container": "wav"},
        check_rms=True,
        tolerated_error_substring="model",
        supported_languages=["en"],
        notes="PASS-WITH-NOTE if Aura-1 isn't bundled with this Aura-2 deploy",
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------

def _build_custom_attributes(voice: str, params: dict[str, str]) -> str:
    """Build the `v1/speak?...` CustomAttributes string."""
    parts = [f"model={quote(voice)}"]
    for k, v in params.items():
        parts.append(f"{quote(str(k))}={quote(str(v))}")
    return "v1/speak?" + "&".join(parts)


def _speak_once(
    session: boto3.Session,
    region: str,
    endpoint: str,
    text: str,
    custom_attributes: str,
) -> tuple[float, bytes | None, str | None, str | None]:
    """One synchronous synthesize. Returns (elapsed, audio_bytes, content_type, error)."""
    sm = _client(session, "sagemaker-runtime", region)
    start = time.monotonic()
    try:
        resp = sm.invoke_endpoint(
            EndpointName=endpoint,
            Body=json.dumps({"text": text}).encode("utf-8"),
            ContentType="application/json",
            Accept="*/*",
            CustomAttributes=custom_attributes,
        )
        elapsed = time.monotonic() - start
        audio = resp["Body"].read()
        return elapsed, audio, resp.get("ContentType"), None
    except ClientError as e:
        return time.monotonic() - start, None, None, str(e)


def _split_s3_uri(s3_uri: str) -> tuple[str, str]:
    p = urlparse(s3_uri)
    return p.netloc, p.path.lstrip("/")


class PreflightError(RuntimeError):
    """Raised when the async preflight detects an IAM/S3 misconfiguration that
    would otherwise make invoke_endpoint_async queue forever with no error."""


def preflight_async_iam(session, region, endpoint, bucket, upload_prefix) -> None:
    """Fail-fast before any async invoke if the endpoint's execution role can't
    read the input or write the output / failure prefixes (SageMaker's async
    dispatcher uses the *endpoint's* role, not the caller's — a missing
    permission silently queues the job for the full timeout)."""
    sm = _client(session, "sagemaker", region)
    iam = _client(session, "iam")
    ep = sm.describe_endpoint(EndpointName=endpoint)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    model = sm.describe_model(ModelName=cfg["ProductionVariants"][0]["ModelName"])
    exec_role = model["ExecutionRoleArn"]
    out_cfg = cfg["AsyncInferenceConfig"]["OutputConfig"]
    out_b, out_k = _split_s3_uri(out_cfg["S3OutputPath"])
    fail_b, fail_k = _split_s3_uri(out_cfg["S3FailurePath"])
    checks = [
        ("s3:GetObject", f"arn:aws:s3:::{bucket}/{upload_prefix.strip('/')}/probe.json"),
        ("s3:PutObject", f"arn:aws:s3:::{out_b}/{out_k.strip('/')}/probe.out"),
        ("s3:PutObject", f"arn:aws:s3:::{fail_b}/{fail_k.strip('/')}/probe.out"),
    ]
    try:
        denials = []
        for action, resource in checks:
            r = iam.simulate_principal_policy(
                PolicySourceArn=exec_role, ActionNames=[action], ResourceArns=[resource])
            if r["EvaluationResults"][0]["EvalDecision"] != "allowed":
                denials.append(f"{action} on {resource}")
        if denials:
            raise PreflightError(
                f"exec role {exec_role} cannot perform required S3 ops:\n  " + "\n  ".join(denials))
        logger.info("preflight: exec-role S3 permissions OK")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation"):
            logger.warning("preflight: caller lacks iam:SimulatePrincipalPolicy — skipping exec-role check")
        else:
            raise


def _async_speak_once(session, region, endpoint, text, custom, ctx) -> tuple[float, bytes | None, str | None, str | None]:
    """Upload the text to S3, invoke_endpoint_async, poll for the output audio.
    Returns (elapsed, audio_bytes, output_uri, error) — output_uri in the
    content_type slot so the caller can surface where the audio landed."""
    s3 = _client(session, "s3", region)
    sm = _client(session, "sagemaker-runtime", region)
    start = time.monotonic()
    key = f"{ctx['upload_prefix'].strip('/')}/{uuid.uuid4()}.json"
    try:
        s3.put_object(Bucket=ctx["bucket"], Key=key,
                      Body=json.dumps({"text": text}).encode("utf-8"),
                      ContentType="application/json")
    except ClientError as e:
        return time.monotonic() - start, None, None, f"input upload failed: {e}"
    input_s3 = f"s3://{ctx['bucket']}/{key}"
    try:
        r = sm.invoke_endpoint_async(
            EndpointName=endpoint, InputLocation=input_s3,
            ContentType="application/json", Accept="*/*", CustomAttributes=custom,
            InvocationTimeoutSeconds=ctx["invocation_timeout_s"],
            InferenceId=uuid.uuid4().hex,
        )
    except ClientError as e:
        return time.monotonic() - start, None, None, str(e)
    out_b, out_k = _split_s3_uri(r["OutputLocation"])
    fail_b, fail_k = _split_s3_uri(r["FailureLocation"])
    deadline = start + ctx["invocation_timeout_s"] + 60
    while time.monotonic() < deadline:
        try:
            body = s3.get_object(Bucket=out_b, Key=out_k)["Body"].read()
            return time.monotonic() - start, body, r["OutputLocation"], None
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") not in ("404", "NoSuchKey", "NotFound"):
                return time.monotonic() - start, None, None, f"output head error: {e}"
        try:
            body = s3.get_object(Bucket=fail_b, Key=fail_k)["Body"].read()
            return (time.monotonic() - start, None, None,
                    f"endpoint failure: {body.decode(errors='replace')[:200]}")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") not in ("404", "NoSuchKey", "NotFound"):
                return time.monotonic() - start, None, None, f"failure head error: {e}"
        time.sleep(ctx["poll_interval_s"])
    return time.monotonic() - start, None, None, f"timed out polling {r['OutputLocation']}"


def _synth(session, region, endpoint, text, voice, params, *, mode="sync", async_ctx=None):
    custom = _build_custom_attributes(voice, params)
    if mode == "async":
        return _async_speak_once(session, region, endpoint, text, custom, async_ctx)
    return _speak_once(session, region, endpoint, text, custom)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_audio(scenario: TTSScenario, audio: bytes, requested_sr: int) -> tuple[bool, list[str], dict]:
    """Apply the scenario's audio assertions. Returns (ok, notes, stats)."""
    notes: list[str] = []
    stats: dict = {"bytes": len(audio)}
    checks: list[bool] = [len(audio) >= scenario.min_bytes]
    if len(audio) < scenario.min_bytes:
        notes.append(f"BYTES<{scenario.min_bytes} (got {len(audio)})")

    container = sniff_container(audio)
    notes.append(f"container={container}")
    if scenario.expect_container:
        match = container == scenario.expect_container
        checks.append(match)
        if not match:
            notes.append(f"EXPECTED container={scenario.expect_container}")

    encoding = scenario.params.get("encoding", "linear16")
    if scenario.expect_sample_rate is not None:
        try:
            w = parse_wav(audio)
            stats["sample_rate"] = w["sample_rate"]
            sr_ok = w["sample_rate"] == scenario.expect_sample_rate
            checks.append(sr_ok)
            notes.append(f"sample_rate={w['sample_rate']}{'' if sr_ok else ' EXPECTED ' + str(scenario.expect_sample_rate)}")
        except Exception as e:  # noqa: BLE001
            checks.append(False)
            notes.append(f"WAV parse failed: {e}")

    if scenario.check_rms and encoding == "linear16":
        try:
            info = linear16_duration_and_rms(audio, requested_sr)
            stats["rms"] = round(info["rms"], 1)
            stats["duration_s"] = round(info["duration_s"], 3)
            rms_ok = info["rms"] > SILENCE_RMS_FLOOR
            checks.append(rms_ok)
            notes.append(f"rms={info['rms']:.0f}{'' if rms_ok else ' (SILENT!)'}")
            notes.append(f"dur={info['duration_s']:.2f}s")
        except Exception as e:  # noqa: BLE001
            checks.append(False)
            notes.append(f"PCM analysis failed: {e}")

    return all(checks), notes, stats


def run_scenario(
    scenario: TTSScenario,
    *,
    session: boto3.Session,
    region: str,
    endpoint: str,
    voice: str,
    text_default: str,
    mode: str = "sync",
    async_ctx: dict | None = None,
) -> dict:
    text = scenario.text if scenario.text is not None else text_default
    eff_voice = scenario.voice or voice
    requested_sr = int(scenario.params.get("sample_rate", 24000) or 24000)

    def synth(t, v, p):
        return _synth(session, region, endpoint, t, v, p, mode=mode, async_ctx=async_ctx)

    start = time.monotonic()

    # --- Special: speed-vs-duration monotonicity. ---
    if scenario.speed_compare:
        durations: dict[str, float] = {}
        err = None
        for speed in ("0.7", "1.0", "1.5"):
            params = {**scenario.params, "speed": speed}
            _, audio, _, e = synth(text, eff_voice, params)
            if e:
                err = e
                break
            info = linear16_duration_and_rms(audio or b"", requested_sr)
            durations[speed] = info["duration_s"]
        elapsed = time.monotonic() - start
        if err:
            tol = scenario.tolerated_error_substring
            if tol and tol.lower() in err.lower():
                return _row(scenario, True, elapsed, [f"DEPLOYMENT-GAP: '{tol}' — pass-with-note"])
            return _row(scenario, False, elapsed, [f"error: {err[:120]}"], error=err)
        d07, d10, d15 = durations["0.7"], durations["1.0"], durations["1.5"]
        # Slower speed → longer audio. Allow small slack for synthesis variance.
        ok = d07 > d10 > d15 > 0
        notes = [f"dur@0.7={d07:.2f}s", f"dur@1.0={d10:.2f}s", f"dur@1.5={d15:.2f}s",
                 "monotonic↓" if ok else "NOT MONOTONIC"]
        return _row(scenario, ok, elapsed, notes, duration_s=d10)

    # --- Single or concurrent synthesize. ---
    results: list[tuple[float, bytes | None, str | None, str | None]] = []
    if scenario.concurrency == 1:
        results.append(synth(text, eff_voice, scenario.params))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency) as ex:
            futures = [
                ex.submit(synth, text, eff_voice, scenario.params)
                for _ in range(scenario.requests)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
    elapsed = time.monotonic() - start

    failures = [r for r in results if r[3] is not None]
    successes = [r for r in results if r[3] is None and r[1] is not None]
    failure_text = " ".join(r[3] for r in failures if r[3]).lower()

    # --- Negative test: rejection expected. ---
    if scenario.expect_failure:
        ok = len(failures) > 0
        notes = ["expected-failure"]
        if not ok:
            notes.append("NO REJECTION (expected one)")
        else:
            notes.append(f"err='{failures[0][3][:80]}'")
        return _row(scenario, ok, elapsed, notes)

    # --- Tolerated bundle/format gap. ---
    if scenario.tolerated_error_substring and failures and scenario.tolerated_error_substring.lower() in failure_text:
        return _row(scenario, True, elapsed,
                    [f"DEPLOYMENT-GAP: '{scenario.tolerated_error_substring}' — pass-with-note"])

    if not successes:
        first = failures[0][3] if failures else "unknown"
        return _row(scenario, False, elapsed,
                    [f"all {len(results)} request(s) failed: {first[:120]}"], error=first)

    audio = successes[0][1] or b""
    ok, notes, stats = _validate_audio(scenario, audio, requested_sr)
    if len(results) > 1:
        notes.insert(0, f"reqs={len(results)} ok={len(successes)} fail={len(failures)}")
    if failures:
        ok = False
        notes.append(f"PARTIAL FAILURE x{len(failures)}: {failures[0][3][:80]}")
    if scenario.notes:
        notes.append(scenario.notes)
    return _row(scenario, ok, elapsed, notes,
                bytes_=stats.get("bytes"), rms=stats.get("rms"), duration_s=stats.get("duration_s"))


def _row(scenario, ok, elapsed, notes, *, bytes_=None, rms=None, duration_s=None, error=None) -> dict:
    return {
        "scenario": scenario.name,
        "ok": ok,
        "bytes": bytes_,
        "rms": rms,
        "duration_s": duration_s,
        "elapsed_s": elapsed,
        "notes": " ".join(notes),
        "error": error,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end correctness + feature test for a batch (REST) SageMaker "
            "TTS endpoint. Synthesizes a known sentence across the full TTS "
            "parameter surface and validates the returned audio self-contained "
            "(bytes / container / sample rate / non-silent RMS / speed→duration)."
        )
    )
    p.add_argument("endpoint_name", nargs="?", default=None,
                   help="TTS SageMaker endpoint name (required unless --list)")
    p.add_argument("--mode", choices=["sync", "async"], default="sync",
                   help="Transport: `sync` = invoke_endpoint (REST); `async` = "
                        "invoke_endpoint_async (S3 in/out). A TTS endpoint serves one "
                        "or the other based on whether its config has AsyncInferenceConfig.")
    p.add_argument("--bucket", default=None,
                   help="S3 bucket for async input uploads. Required with --mode async.")
    p.add_argument("--upload-prefix", default="tts-e2e-async-input",
                   help="S3 key prefix for async input uploads (default: tts-e2e-async-input)")
    p.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    p.add_argument("--language", default=None,
                   help=f"Bundle language (one of: {', '.join(SUPPORTED_LANGUAGES)}). "
                        "Selects the reference text + featured voices. If omitted, "
                        "derived from the --voice suffix (e.g. -es), else falls back "
                        f"to '{DEFAULT_LANGUAGE}'.")
    p.add_argument("--voice", default=None,
                   help="Default TTS voice. If omitted, the first featured voice for "
                        "the selected language is used (e.g. aura-2-thalia-en for en, "
                        "aura-2-celeste-es for es).")
    p.add_argument("--voice-coverage-n", type=int, default=3,
                   help="How many language-matched voices to cover individually "
                        "(default: 3). Each gets its own scenario row.")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="Log directory (default: /tmp/dg-sagemaker-e2e/tts-batch/<ts>)")
    p.add_argument("--scenarios", default="", metavar="NAME,NAME,...",
                   help="Comma-separated subset of scenario names (default: all). See --list.")
    p.add_argument("--fips", action="store_true",
                   help="Route AWS calls to the FIPS 140-3 endpoints (api-fips control "
                        "plane, runtime-fips invoke, s3-fips for --mode async). OFF by "
                        "default. Selects AWS endpoints only — says nothing about the "
                        "container's own crypto.")
    p.add_argument("--list", action="store_true", help="List scenarios and exit")
    p.add_argument("--invocation-timeout-s", type=int, default=600,
                   help="(async) per-invocation timeout passed to SageMaker (default: 600)")
    p.add_argument("--poll-interval-s", type=float, default=3.0,
                   help="(async) seconds between S3 polls for each result (default: 3.0)")
    p.add_argument("--skip-preflight", action="store_true",
                   help="(async) skip the exec-role IAM/S3 preflight check")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def _resolve_language_and_voice(args) -> tuple[str, str]:
    """Pick the final (language, voice) pair from the CLI args.

    Precedence: --language explicit > derive from --voice suffix > DEFAULT_LANGUAGE.
    """
    if args.language:
        if args.language not in SUPPORTED_LANGUAGES:
            raise SystemExit(
                f"ERROR: --language {args.language!r} not supported "
                f"(known: {', '.join(SUPPORTED_LANGUAGES)})"
            )
        language = args.language
    elif args.voice:
        lang_from_voice = voice_language(args.voice)
        language = lang_from_voice or DEFAULT_LANGUAGE
    else:
        language = DEFAULT_LANGUAGE
    voice = args.voice or default_voice(language)
    return language, voice


def main() -> int:
    def _activate_fips(on: bool) -> None:
        """Must run before ANY client is built — including the sts credential
        probe further down. Set after that point, part of the run silently used
        the standard endpoints while the log claimed FIPS."""
        global _FIPS
        _FIPS = on

    args = _make_parser().parse_args()
    _activate_fips(args.fips)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.list:
        # For --list, use the default language so the output is deterministic
        # without requiring an endpoint or language flag.
        scenarios = default_scenarios(DEFAULT_LANGUAGE, args.voice_coverage_n)
        print(f"Available scenarios (shown for language={DEFAULT_LANGUAGE!r}; "
              f"voice rows are per-language):")
        for s in scenarios:
            langs = f"  langs={s.supported_languages}" if s.supported_languages else ""
            print(f"  {s.name:<32} {s.description}{langs}")
        return 0

    if not args.endpoint_name:
        print("ERROR: endpoint_name is required (run with --list to see scenarios).", file=sys.stderr)
        return 1

    if args.mode == "async" and not args.bucket:
        print("ERROR: --bucket is required with --mode async (S3 input destination).", file=sys.stderr)
        return 1

    language, voice = _resolve_language_and_voice(args)
    scenarios = default_scenarios(language, args.voice_coverage_n)

    if args.scenarios:
        wanted = {tok.strip() for tok in args.scenarios.split(",") if tok.strip()}
        unknown = wanted - {s.name for s in scenarios}
        if unknown:
            print(f"ERROR: unknown scenario(s): {sorted(unknown)}", file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s.name in wanted]

    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        workdir = Path(tempfile.gettempdir()) / "dg-sagemaker-e2e" / "tts-batch" / ts
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        session = boto3.Session(region_name=args.region)
        _client(session, "sts").get_caller_identity()
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"ERROR: AWS credentials missing: {e}", file=sys.stderr)
        return 2

    async_ctx = None
    if args.mode == "async":
        async_ctx = {
            "bucket": args.bucket,
            "upload_prefix": args.upload_prefix,
            "invocation_timeout_s": args.invocation_timeout_s,
            "poll_interval_s": args.poll_interval_s,
        }
        if not args.skip_preflight:
            try:
                preflight_async_iam(session, args.region, args.endpoint_name,
                                    args.bucket, args.upload_prefix)
            except PreflightError as e:
                print(f"ERROR: async preflight failed:\n{e}", file=sys.stderr)
                return 2

    print("=" * 80)
    print(f"Endpoint:    {args.endpoint_name}")
    print(f"Mode:        {args.mode}")
    print(f"Region:      {args.region}")
    print(f"FIPS:        {'yes (--fips)' if args.fips else 'no'}")
    print(f"Language:    {language}")
    print(f"Voice:       {voice}")
    if args.mode == "async":
        print(f"Bucket:      {args.bucket}")
    print(f"Workdir:     {workdir}")
    print("=" * 80)
    print()

    rows: list[dict] = []
    for scenario in scenarios:
        print(f"--> {scenario.name}  ({scenario.description})")
        if not language_supported(scenario.supported_languages, language):
            reason = (f"language not supported: scenario supports "
                      f"{scenario.supported_languages}, run language={language!r}")
            print(f"    SKIP  {reason}")
            rows.append({
                "scenario": scenario.name, "ok": True, "skipped": True,
                "elapsed_s": 0.0, "notes": f"SKIPPED ({reason})",
            })
            continue
        row = run_scenario(
            scenario,
            session=session,
            region=args.region,
            endpoint=args.endpoint_name,
            voice=voice,
            text_default=reference_text(language),
            mode=args.mode,
            async_ctx=async_ctx,
        )
        rows.append(row)
        flag = "SKIP" if row.get("skipped") else ("PASS" if row["ok"] else "FAIL")
        print(f"    {flag}  elapsed={row['elapsed_s']:.1f}s  {row['notes']}")

    print()
    passed, failed = print_summary_table(rows)
    (workdir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nFull results: {workdir / 'results.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
End-to-end correctness + red-team test for batch-mode SageMaker STT endpoints.

A batch endpoint is configured as **either** sync or async — never both — via
its EndpointConfig. This driver mirrors that: pass `--mode sync` or
`--mode async`, and only the matching scenarios run.

**Sync (`--mode sync`)** — uses `sagemaker-runtime.invoke_endpoint` against
the 25-second `spacewalk.wav` sample (under the 25 MB body limit). Five
scenarios cover basic single-request, concurrency, and the
diarize / keyterms / redact feature flags.

**Async (`--mode async`)** — uses `sagemaker-runtime.invoke_endpoint_async`
against a multiplied long-form variant (~15 min, ~76 MB) — comfortably over
the sync cap but well under the 1 GiB async limit. Four scenarios cover
basic single-request, 4× concurrency, diarize, and summarize (requires
fathom in the bundle).

For each invocation the script validates the returned transcript against the
known reference text via Word Error Rate. Designed as the definitive
correctness gate for a batch endpoint before promoting it.

Endpoint prerequisites
----------------------
- A pre-recorded SageMaker endpoint (manifest `mode: batch`) is InService.
- For `--mode async`: the EndpointConfig carries an `AsyncInferenceConfig`
  block with `OutputConfig.S3OutputPath` + `S3FailurePath`, and the
  execution role has read on the input prefix + write on the output / failures
  prefixes. `--bucket` is required (where this script uploads input WAVs).

Usage
-----
    # sync endpoint:
    uv run e2e_test_batch.py your-sync-endpoint --mode sync --region us-east-2

    # async endpoint:
    uv run e2e_test_batch.py your-async-endpoint --mode async \\
      --bucket your-async-bucket --region us-east-2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Same-directory imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test_common import (
    SPACEWALK_REFERENCE_TEXT,
    download_sample,
    expected_text_for_loops,
    fmt_wer,
    language_supported,
    multiply_wav,
    print_summary_table,
    validate_pcm16,
    wer,
)

# Set once from --fips in main(); read by aws_client() for every client built
# in this run. OFF by default — the standard endpoints are used unless asked.
_FIPS = False


def aws_client(session: "boto3.Session", service: str, region: str | None = None):
    """boto3 client honoring this run's --fips setting.

    FIPS is applied PER CLIENT rather than by exporting
    AWS_USE_FIPS_ENDPOINT for the whole process. That variable also redirects
    the IAM Identity Center portal botocore calls to mint SSO credentials, and
    portal.sso-fips.<region>.amazonaws.com does not exist (NXDOMAIN in
    us-east-2 and us-west-2, checked 2026-08-20) — so with an SSO profile whose
    role credentials are not already cached, a process-wide switch dies during
    credential resolution with an EndpointConnectionError naming an SSO URL,
    before it ever reaches SageMaker. (A warm cache masks it, which makes the
    failure look intermittent.) Scoping FIPS to the client sidesteps it
    entirely: the credential chain keeps its normal endpoints.
    """
    cfg = BotoConfig(use_fips_endpoint=True) if _FIPS else None
    kw = {"config": cfg}
    if region is not None:
        kw["region_name"] = region
    return session.client(service, **kw)


DEFAULT_REGION = "us-east-2"
DEFAULT_UPLOAD_PREFIX = "stt-e2e-batch-input"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class BatchScenario:
    """One batch scenario, scoped to features supported on nova-3
    pre-recorded per https://developers.deepgram.com/docs/ (May 2026 audit).

    `bundle_component` documents the model component a bundle must include
    for the scenario to succeed; when omitted, only the ASR weights are
    required. `tolerated_error_substring` lets the scenario PASS-WITH-NOTE
    when the endpoint returns a specific known error (e.g. "entity
    detection" or "summarize" feature unavailable) — surfaces bundle gaps
    without false-failing.
    """
    name: str
    description: str
    transport: str  # "sync" | "async"
    use_long_form: bool = False
    concurrency: int = 1
    requests: int = 1
    custom_params: dict[str, str] = field(default_factory=dict)
    wer_threshold: float = 0.05
    presence_check: str | None = None
    presence_in_full_response: bool = False  # presence check against whole JSON, not just transcript
    bundle_component: str | None = None
    tolerated_error_substring: str | None = None
    # Negative scenario: the endpoint MUST reject. PASS iff an invocation fails
    # with an error containing this substring AND nothing succeeds. Distinct from
    # `tolerated_error_substring`, which only *tolerates* an error (and would
    # falsely PASS if the request were silently served). Used to verify the
    # reject-unknown-params gate (a request with an off-allowlist param → 400).
    expect_error_substring: str | None = None
    # None = no restriction (default). Set ONLY when the endpoint hard-rejects
    # (a real error, not a graceful warning/no-op) outside this language set —
    # see `language_supported()` in e2e_test_common for the exact semantics
    # and why most "English only" docs badges do NOT belong here.
    supported_languages: list[str] | None = None
    notes: str = ""


def default_scenarios(mode: str) -> list[BatchScenario]:
    """Return scenarios appropriate for the configured endpoint mode.

    Sync and async batch endpoints serve different invocation surfaces; an
    endpoint is configured as one or the other via its EndpointConfig, so we
    filter by mode here rather than trying to run incompatible scenarios.

    Feature coverage matrix per the docs (nova-3, pre-recorded):

      Always-supported by the API on pre-recorded nova-3:
        punctuate, smart_format, numerals, measurements, dictation,
        diarize (v1), diarize_model=v2, keyterm, replace (find/replace),
        filler_words, utterances, utt_split, paragraphs,
        profanity_filter, search
        NB. filler_words is only available on the Nova, Nova-2, and
        Nova-3 *general* models — NOT on specialized models such as
        nova-3-medical, so the filler_words scenario is N/A for those
        models even on pre-recorded; see
        https://developers.deepgram.com/docs/filler-words
      Bundle-component dependent (model-catalog families that EXIST but
      are not yet added to FEATURE_COMPONENTS for batch — these
      scenarios will FAIL on the current MP bundles until we add them):
        - smart_format (with entity-aware formatting) + redact /
          detect_entities → semantic_tagger
          (uuid 06bc8f36-e59b-43b0-a1a9-47281c893ee8,
           descriptor name="default", mode="batch")
          NB. an earlier note claimed the model catalog had no
          batch NER model — that was wrong; semantic_tagger and
          broccolizer (6946b038-...) both exist at batch mode.
        - search → g2p
          (uuid 89555db3-b00f-4808-a6af-2db8b73acf14,
           descriptor name="g2p", no mode tag)
        - summarize / topics / intents / sentiment → fathom
          (uuid 67875a7f-..., already in FEATURE_COMPONENTS for batch)
      NOT supported on pre-recorded (excluded here):
        interim_results (streaming-only)
    """
    if mode not in ("sync", "async"):
        raise ValueError(f"mode must be 'sync' or 'async' (got {mode!r})")
    all_scenarios = _all_scenarios()
    return [s for s in all_scenarios if s.transport == mode]


def _all_scenarios() -> list[BatchScenario]:
    return [
        # ============ Sync (synchronous InvokeEndpoint) ============
        # ---- Coverage / load ----
        BatchScenario(
            name="sync_25s",
            description="sync invoke, 25 s file, defaults",
            transport="sync",
        ),
        BatchScenario(
            name="sync_25s_concurrent_5",
            description="5 concurrent sync invokes, 25 s file",
            transport="sync",
            concurrency=5,
            requests=5,
        ),
        # ---- Formatting features ----
        BatchScenario(
            name="sync_25s_diarize_v1",
            description="sync + diarize=true (v1, default body unchanged)",
            transport="sync",
            custom_params={"diarize": "true"},
        ),
        BatchScenario(
            name="sync_25s_diarize_v2",
            description="sync + diarize_model=v2 (pre-recorded only)",
            transport="sync",
            custom_params={"diarize_model": "v2"},
            notes=(
                "diarize_model=v2 is pre-recorded only; streaming returns 400. "
                "Must be sent WITHOUT diarize=true — the API (2026-06, post "
                "1.192) rejects the combo with 400 'diarize_model cannot be "
                "used together with diarize or diarize_version'."
            ),
        ),
        BatchScenario(
            name="sync_25s_smart_format",
            description="sync + smart_format=true (entity-aware formatting)",
            transport="sync",
            custom_params={"smart_format": "true"},
            bundle_component="semantic_tagger (uuid 06bc8f36-... mode=batch)",
            notes=(
                "the API's smart_format implicitly requests entity detection "
                "when format_entity_tags=true; needs the batch NER model"
            ),
        ),
        BatchScenario(
            name="sync_25s_filler_words",
            description="sync + filler_words=true (ums/uhs preserved)",
            transport="sync",
            custom_params={"filler_words": "true"},
            wer_threshold=1.0,  # filler text differs from clean ref
            presence_check="um",
            notes="filler tokens preserved; WER skipped",
        ),
        BatchScenario(
            name="sync_25s_profanity_filter",
            description="sync + profanity_filter=true (clip has none; smoke)",
            transport="sync",
            custom_params={"profanity_filter": "true"},
        ),
        BatchScenario(
            name="sync_25s_paragraphs",
            description="sync + paragraphs=true (pre-recorded only)",
            transport="sync",
            custom_params={"paragraphs": "true"},
            notes="paragraphs are pre-recorded only",
        ),
        BatchScenario(
            name="sync_25s_utterances",
            description="sync + utterances=true",
            transport="sync",
            custom_params={"utterances": "true"},
        ),
        BatchScenario(
            name="sync_25s_utt_split",
            description="sync + utt_split=0.8 (utterance split window)",
            transport="sync",
            custom_params={"utterances": "true", "utt_split": "0.8"},
            notes="utt_split sets silence threshold (s) between utterances",
        ),
        BatchScenario(
            name="sync_25s_punctuate",
            description="sync + punctuate=true (default; explicit verification)",
            transport="sync",
            custom_params={"punctuate": "true"},
            presence_check=".",
        ),
        BatchScenario(
            name="sync_25s_numerals",
            description="sync + numerals=true (digit substitution)",
            transport="sync",
            custom_params={"numerals": "true"},
            notes="digit substitution; clip has few numbers — smoke",
        ),
        BatchScenario(
            name="sync_25s_measurements",
            description="sync + measurements=true (unit/measure substitution)",
            transport="sync",
            custom_params={"measurements": "true"},
            notes="unit substitution; clip has no measurements — smoke",
        ),
        BatchScenario(
            name="sync_25s_dictation",
            description="sync + dictation=true (spoken punctuation -> chars)",
            transport="sync",
            custom_params={"dictation": "true"},
            wer_threshold=1.0,
            notes="dictation transforms spoken punctuation cues",
        ),
        # ---- Custom vocabulary ----
        BatchScenario(
            name="sync_25s_keyterm",
            description="sync + keyterm (spacewalk,female; nova-3 boost)",
            transport="sync",
            custom_params={"keyterm": "spacewalk", "keyterm2": "female"},
            presence_check="spacewalk",
            notes="nova-3 only — `keyterm`, NOT `keywords`",
        ),
        BatchScenario(
            name="sync_25s_replace",
            description="sync + replace=spacewalk:moonwalk",
            transport="sync",
            custom_params={"replace": "spacewalk:moonwalk"},
            wer_threshold=1.0,
            presence_check="moonwalk",
            notes="content swap; WER skipped",
        ),
        BatchScenario(
            name="sync_25s_search",
            description="sync + search=spacewalk (acoustic phonetic match)",
            transport="sync",
            custom_params={"search": "spacewalk"},
            presence_in_full_response=True,
            presence_check="spacewalk",
            bundle_component="g2p (uuid 89555db3-...)",
            notes="search needs g2p (grapheme-to-phoneme) — NOT yet in FEATURE_COMPONENTS",
        ),
        # ---- Bundle-component-dependent (will FAIL until bundle adds these) ----
        BatchScenario(
            name="sync_25s_redact_name",
            description="sync + redact=name (requires batch entity detection; English-classified transcript only)",
            transport="sync",
            custom_params={"redact": "name"},
            wer_threshold=1.0,
            bundle_component="semantic_tagger (uuid 06bc8f36-... mode=batch)",
            supported_languages=["en", "multi"],
            notes=(
                "confirmed 2026-07-14: entity-based redaction only routes to "
                "the engine when the TRANSCRIPT is "
                "classified Englishness::All — 'multi' passes because "
                "language-detect on English audio resolves to English, but a "
                "forced non-English language hard-500s ('Redaction unavailable') "
                "regardless of bundle content — auto-skipped outside en/multi"
            ),
        ),
        BatchScenario(
            name="sync_25s_summarize",
            description="sync + summarize=v2 (English only — hard 400 otherwise)",
            transport="sync",
            custom_params={"summarize": "v2"},
            wer_threshold=1.0,
            bundle_component="fathom",
            presence_in_full_response=True,
            presence_check="summary",
            tolerated_error_substring="summarize",
            supported_languages=["en"],
            notes=(
                "confirmed 2026-07-14: the API hard-rejects with 400 "
                '"Summarization v2 not supported for non-English languages" '
                "for any non-English/multi request, regardless of bundle "
                "content — auto-skipped outside --language en"
            ),
        ),
        BatchScenario(
            name="sync_25s_topics",
            description="sync + topics=true (English; requires LLM/fathom-style component)",
            transport="sync",
            custom_params={"topics": "true"},
            wer_threshold=1.0,
            bundle_component="fathom",
            tolerated_error_substring="topic",
            notes="English only",
        ),
        BatchScenario(
            name="sync_25s_intents",
            description="sync + intents=true (English; requires LLM/fathom-style component)",
            transport="sync",
            custom_params={"intents": "true"},
            wer_threshold=1.0,
            bundle_component="fathom",
            tolerated_error_substring="intent",
            notes="English only",
        ),
        BatchScenario(
            name="sync_25s_sentiment",
            description="sync + sentiment=true (English; requires LLM/fathom-style component)",
            transport="sync",
            custom_params={"sentiment": "true"},
            wer_threshold=1.0,
            bundle_component="fathom",
            tolerated_error_substring="sentiment",
            notes="English only",
        ),
        # ---- Negative: reject-unknown-params gate (container 400s off-allowlist params) ----
        BatchScenario(
            name="sync_25s_reject_unknown_param",
            description="sync + bogus=true → expect 400 unsupported_parameter (not served)",
            transport="sync",
            custom_params={"bogus": "true"},
            expect_error_substring="unsupported_parameter",
            notes="reject-unknown-params: off-allowlist key must 400 before the API, not serve",
        ),
        BatchScenario(
            name="sync_25s_reject_unknown_param_falsy",
            description="sync + bogus=false → expect 400 (reject is value-independent)",
            transport="sync",
            custom_params={"bogus": "false"},
            expect_error_substring="unsupported_parameter",
            notes="value-independent: a falsy value does NOT exempt an unknown key",
        ),

        # ============ Async (InvokeEndpointAsync, S3 in/out) ============
        BatchScenario(
            name="async_25s",
            description="async invoke, 25 s file (short-form smoke)",
            transport="async",
        ),
        BatchScenario(
            name="async_15min",
            description="async invoke, ~15 min file (single)",
            transport="async",
            use_long_form=True,
        ),
        BatchScenario(
            name="async_15min_concurrent_4",
            description="4 concurrent async invokes, ~15 min file",
            transport="async",
            use_long_form=True,
            concurrency=4,
            requests=4,
        ),
        BatchScenario(
            name="async_15min_diarize_v2",
            description="async + diarize_model=v2",
            transport="async",
            use_long_form=True,
            custom_params={"diarize": "true", "diarize_model": "v2"},
        ),
        BatchScenario(
            name="async_15min_summarize",
            description="async + summarize=v2 (English only — hard 400 otherwise)",
            transport="async",
            use_long_form=True,
            custom_params={"summarize": "v2"},
            wer_threshold=1.0,
            bundle_component="fathom",
            presence_in_full_response=True,
            presence_check="summary",
            tolerated_error_substring="summarize",
            supported_languages=["en"],
            notes=(
                "confirmed 2026-07-14: the API hard-rejects with 400 "
                '"Summarization v2 not supported for non-English languages" '
                "for any non-English/multi request, regardless of bundle "
                "content — auto-skipped outside --language en"
            ),
        ),
        BatchScenario(
            name="async_15min_redact_name",
            description="async + redact=name (requires batch entity detection; English-classified transcript only)",
            transport="async",
            use_long_form=True,
            custom_params={"redact": "name"},
            wer_threshold=1.0,
            bundle_component="semantic_tagger (uuid 06bc8f36-... mode=batch)",
            supported_languages=["en", "multi"],
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_query(params: dict[str, str]) -> str:
    """Build the v1/listen?... custom-attributes string.

    `diarize` and `diarize_model` are mutually exclusive in the API
    (`diarize_model cannot be used together with diarize or
    diarize_version`), so the auto-added `diarize=false` default is
    suppressed when the caller already sets `diarize_model`. Same idea
    for any other future mutex; today only diarize/diarize_model
    matters.
    """
    parts = []
    has_diarize_model = "diarize_model" in params
    base = {
        "model": params.pop("model", "nova-3"),
        "language": params.pop("language", "en"),
        "punctuate": params.pop("punctuate", "true"),
    }
    if not has_diarize_model:
        base["diarize"] = params.pop("diarize", "false")
    elif "diarize" in params:
        # Caller passed both — drop the implicit one; the API would 400 anyway.
        params.pop("diarize")
    for k, v in base.items():
        parts.append(f"{k}={v}")
    for k, v in params.items():
        # Allow duplicate `keyterm` etc. via numbered keys (keyterm, keyterm2…).
        actual = k.rstrip("0123456789")
        parts.append(f"{quote(actual)}={quote(v)}")
    return "v1/listen?" + "&".join(parts)


def _extract_transcript(result: dict) -> tuple[str, float, float]:
    try:
        alt = result["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        return "", 0.0, 0.0
    transcript = alt.get("transcript", "") or ""
    confidence = float(alt.get("confidence", 0.0) or 0.0)
    duration = float(((result or {}).get("metadata") or {}).get("duration", 0.0) or 0.0)
    return transcript, confidence, duration


def _split_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


class PreflightError(RuntimeError):
    """Raised when the async preflight detects an IAM/S3 misconfiguration
    that would otherwise cause invoke_endpoint_async to queue forever
    with no surfaced error."""


def preflight_async_iam(
    session: boto3.Session,
    region: str,
    endpoint: str,
    bucket: str,
    upload_prefix: str,
) -> None:
    """Fail-fast before any S3 upload / invoke when the endpoint's
    execution role can't read input or write output / failures.

    SageMaker's async dispatcher reads InputLocation + writes OutputLocation
    using the *endpoint's execution role* (not the caller's). If that role
    lacks GetObject on the input or PutObject on the output/failure prefix,
    invoke_endpoint_async returns 200 and the job queues silently — no
    CloudWatch invocations, no container logs, ~30 min wasted.

    Primary path uses iam:SimulatePrincipalPolicy on the exec role.
    Falls back to a HeadBucket + zero-byte probe write/delete when the
    caller lacks iam:SimulatePrincipalPolicy. Final fallback is a WARN.
    """
    sm = aws_client(session, "sagemaker", region)
    iam = aws_client(session, "iam")
    ep = sm.describe_endpoint(EndpointName=endpoint)
    cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
    variant = cfg["ProductionVariants"][0]
    model = sm.describe_model(ModelName=variant["ModelName"])
    exec_role = model["ExecutionRoleArn"]
    out_cfg = cfg["AsyncInferenceConfig"]["OutputConfig"]
    out_bucket, out_key_prefix = _split_s3_uri(out_cfg["S3OutputPath"])
    fail_bucket, fail_key_prefix = _split_s3_uri(out_cfg["S3FailurePath"])
    checks: list[tuple[str, str]] = [
        ("s3:GetObject", f"arn:aws:s3:::{bucket}/{upload_prefix.strip('/')}/probe.wav"),
        ("s3:PutObject", f"arn:aws:s3:::{out_bucket}/{out_key_prefix.strip('/')}/probe.out"),
        ("s3:PutObject", f"arn:aws:s3:::{fail_bucket}/{fail_key_prefix.strip('/')}/probe.out"),
    ]
    logger.info("preflight: exec_role=%s checks=%s", exec_role, checks)
    try:
        denials: list[str] = []
        for action, resource in checks:
            r = iam.simulate_principal_policy(
                PolicySourceArn=exec_role, ActionNames=[action], ResourceArns=[resource],
            )
            decision = r["EvaluationResults"][0]["EvalDecision"]
            if decision != "allowed":
                matched = r["EvaluationResults"][0].get("MatchedStatements") or []
                denials.append(f"{action} on {resource}: {decision} (matched={matched})")
        if denials:
            raise PreflightError(
                f"exec role {exec_role} cannot perform required S3 ops:\n  "
                + "\n  ".join(denials)
                + "\nFix the role's S3 policy before retrying — invoke_endpoint_async "
                  "would otherwise queue silently."
            )
        logger.info("preflight: iam:SimulatePrincipalPolicy passed for all 3 checks")
        return
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation"):
            raise
        logger.info("preflight: caller lacks iam:SimulatePrincipalPolicy (%s); using probe fallback", code)
    s3 = aws_client(session, "s3", region)
    probe_key = f"_preflight-probe-{uuid.uuid4()}.tmp"
    for b, prefix in ((bucket, upload_prefix), (out_bucket, out_key_prefix), (fail_bucket, fail_key_prefix)):
        key = f"{prefix.strip('/')}/{probe_key}"
        try:
            s3.head_bucket(Bucket=b)
            s3.put_object(Bucket=b, Key=key, Body=b"")
            s3.delete_object(Bucket=b, Key=key)
        except ClientError as e:
            logger.warning(
                "preflight: caller-side probe on s3://%s/%s failed (%s); "
                "cannot verify exec-role perms — proceeding anyway",
                b, key, e,
            )
            return
    logger.info("preflight: caller-side probe writes succeeded (exec-role perms unverified)")


# ---------------------------------------------------------------------------
# Sync runner
# ---------------------------------------------------------------------------

def _sync_invoke_once(
    session: boto3.Session,
    region: str,
    endpoint: str,
    audio_bytes: bytes,
    custom_attributes: str,
) -> tuple[float, dict | None, str | None]:
    sm = aws_client(session, "sagemaker-runtime", region)
    start = time.monotonic()
    try:
        resp = sm.invoke_endpoint(
            EndpointName=endpoint,
            Body=audio_bytes,
            ContentType="audio/wav",
            Accept="application/json",
            CustomAttributes=custom_attributes,
        )
        elapsed = time.monotonic() - start
        body = resp["Body"].read()
        try:
            return elapsed, json.loads(body), None
        except json.JSONDecodeError as e:
            return elapsed, None, f"non-JSON response: {e}"
    except ClientError as e:
        return time.monotonic() - start, None, str(e)


def run_sync_scenario(
    scenario: BatchScenario,
    *,
    session: boto3.Session,
    region: str,
    endpoint: str,
    short_wav: Path,
    long_wav: Path,
    long_loops: int,
    model: str,
    language: str,
) -> dict:
    wav = long_wav if scenario.use_long_form else short_wav
    expected = (
        expected_text_for_loops(long_loops)
        if scenario.use_long_form
        else SPACEWALK_REFERENCE_TEXT
    )
    audio = wav.read_bytes()
    params = {"model": model, "language": language, **scenario.custom_params}
    custom = _build_query(params)

    start = time.monotonic()
    results: list[tuple[float, dict | None, str | None]] = []
    if scenario.concurrency == 1:
        results.append(_sync_invoke_once(session, region, endpoint, audio, custom))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency) as ex:
            futures = [
                ex.submit(_sync_invoke_once, session, region, endpoint, audio, custom)
                for _ in range(scenario.requests)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
    elapsed = time.monotonic() - start

    failures = [r for r in results if r[2] is not None]
    successes = [r for r in results if r[2] is None and r[1] is not None]

    # Negative scenario: the endpoint MUST reject (e.g. unsupported param → 400).
    # PASS iff a failure carrying the expected substring occurred AND nothing was
    # served (a success means the reject didn't fire). Checked first — for this
    # scenario type a "failure" is the pass condition.
    if scenario.expect_error_substring is not None:
        sub = scenario.expect_error_substring.lower()
        matched = any(sub in (r[2] or "").lower() for r in failures)
        ok = matched and not successes
        if ok:
            note = f"REJECTED as expected ('{scenario.expect_error_substring}')"
        elif successes:
            note = "EXPECTED REJECT but request was served — reject gate not firing"
        else:
            first = failures[0][2][:160] if failures else "no response"
            note = f"EXPECTED '{scenario.expect_error_substring}' but got: {first}"
        return {
            "scenario": scenario.name,
            "ok": ok,
            "wer": 0.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": (f"{scenario.notes} | " if scenario.notes else "") + note,
        }

    # PASS-WITH-NOTE: bundle missing component this scenario needs. The
    # endpoint returns a known error pattern we tolerate to surface the
    # bundle gap without false-failing.
    failure_text = " ".join(r[2] for r in failures if r[2]).lower()
    tolerated = (
        scenario.tolerated_error_substring is not None
        and scenario.tolerated_error_substring.lower() in failure_text
    )
    if tolerated:
        return {
            "scenario": scenario.name,
            "ok": True,
            "wer": 0.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": (
                f"BUNDLE-GAP: '{scenario.tolerated_error_substring}' "
                f"(needs {scenario.bundle_component} component) — pass-with-note"
            ),
        }

    if not successes:
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": f"all {len(results)} requests failed: {failures[0][2][:120] if failures else 'unknown'}",
            "error": "all failed",
        }

    transcript, conf, dur = _extract_transcript(successes[0][1] or {})
    w_ratio, s, d, i, n = wer(expected, transcript)

    notes = [
        f"reqs={len(results)}",
        f"success={len(successes)}",
        f"conf={conf:.1%}",
        f"audio_dur={dur:.1f}s",
    ]
    if failures:
        notes.append(f"failed={len(failures)}")
    if scenario.notes:
        notes.append(scenario.notes)

    presence_ok = True
    if scenario.presence_check:
        haystack = (
            json.dumps(successes[0][1] or {}).lower()
            if scenario.presence_in_full_response
            else transcript.lower()
        )
        presence_ok = scenario.presence_check.lower() in haystack
        notes.append(f"presence({scenario.presence_check})={'ok' if presence_ok else 'MISSING'}")

    ok = (
        not failures
        and w_ratio <= scenario.wer_threshold
        and presence_ok
    )

    return {
        "scenario": scenario.name,
        "ok": ok,
        "wer": w_ratio,
        "sdi": (s, d, i),
        "words": n,
        "elapsed_s": elapsed,
        "notes": " ".join(notes),
        "transcript_head": transcript[:200],
    }


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

def _upload_input(session: boto3.Session, region: str, wav: Path, bucket: str, prefix: str) -> str:
    s3 = aws_client(session, "s3", region)
    key = f"{prefix.strip('/')}/{uuid.uuid4()}.wav"
    s3.upload_file(str(wav), bucket, key, ExtraArgs={"ContentType": "audio/wav"})
    return f"s3://{bucket}/{key}"


def _async_invoke_once(
    session: boto3.Session,
    region: str,
    endpoint: str,
    input_s3: str,
    custom: str,
    invocation_timeout_s: int,
    poll_interval_s: float,
    inference_id: str,
) -> tuple[float, dict | None, str | None, str]:
    """Submit + poll one async invocation. Returns (elapsed, response, error, out_uri)."""
    sm = aws_client(session, "sagemaker-runtime", region)
    s3 = aws_client(session, "s3", region)
    start = time.monotonic()
    try:
        r = sm.invoke_endpoint_async(
            EndpointName=endpoint,
            InputLocation=input_s3,
            ContentType="audio/wav",
            Accept="application/json",
            CustomAttributes=custom,
            InvocationTimeoutSeconds=invocation_timeout_s,
            InferenceId=inference_id,
        )
    except ClientError as e:
        return time.monotonic() - start, None, str(e), ""

    out_uri = r["OutputLocation"]
    fail_uri = r["FailureLocation"]
    out_bucket, out_key = _split_s3_uri(out_uri)
    fail_bucket, fail_key = _split_s3_uri(fail_uri)
    deadline = start + invocation_timeout_s + 60
    while time.monotonic() < deadline:
        try:
            s3.head_object(Bucket=out_bucket, Key=out_key)
            body = s3.get_object(Bucket=out_bucket, Key=out_key)["Body"].read()
            try:
                return time.monotonic() - start, json.loads(body), None, out_uri
            except json.JSONDecodeError as e:
                return time.monotonic() - start, None, f"non-JSON: {e}", out_uri
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in ("404", "NoSuchKey", "NotFound"):
                return time.monotonic() - start, None, f"head error: {e}", out_uri
        try:
            s3.head_object(Bucket=fail_bucket, Key=fail_key)
            body = s3.get_object(Bucket=fail_bucket, Key=fail_key)["Body"].read()
            return (
                time.monotonic() - start, None,
                f"endpoint failure: {body.decode(errors='replace')[:200]}",
                fail_uri,
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in ("404", "NoSuchKey", "NotFound"):
                return time.monotonic() - start, None, f"failure head: {e}", fail_uri
        time.sleep(poll_interval_s)
    return time.monotonic() - start, None, f"timed out polling {out_uri}", out_uri


def run_async_scenario(
    scenario: BatchScenario,
    *,
    session: boto3.Session,
    region: str,
    endpoint: str,
    bucket: str,
    upload_prefix: str,
    short_wav: Path,
    long_wav: Path,
    long_loops: int,
    model: str,
    language: str,
    invocation_timeout_s: int,
    poll_interval_s: float,
) -> dict:
    wav = long_wav if scenario.use_long_form else short_wav
    expected = (
        expected_text_for_loops(long_loops)
        if scenario.use_long_form
        else SPACEWALK_REFERENCE_TEXT
    )
    params = {"model": model, "language": language, **scenario.custom_params}
    custom = _build_query(params)

    input_s3 = _upload_input(session, region, wav, bucket, upload_prefix)

    start = time.monotonic()
    results: list[tuple[float, dict | None, str | None, str]] = []
    prefix = f"e2e-{scenario.name}-{int(time.time())}"
    if scenario.concurrency == 1:
        results.append(_async_invoke_once(
            session, region, endpoint, input_s3, custom,
            invocation_timeout_s, poll_interval_s, f"{prefix}-001",
        ))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency) as ex:
            futures = [
                ex.submit(
                    _async_invoke_once, session, region, endpoint, input_s3, custom,
                    invocation_timeout_s, poll_interval_s, f"{prefix}-{i:03d}",
                )
                for i in range(scenario.requests)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
    elapsed = time.monotonic() - start

    failures = [r for r in results if r[2] is not None]
    successes = [r for r in results if r[2] is None and r[1] is not None]

    failure_text = " ".join(r[2] for r in failures if r[2]).lower()
    tolerated = (
        scenario.tolerated_error_substring is not None
        and scenario.tolerated_error_substring.lower() in failure_text
    )
    if tolerated:
        return {
            "scenario": scenario.name,
            "ok": True,
            "wer": 0.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": (
                f"BUNDLE-GAP: '{scenario.tolerated_error_substring}' "
                f"(needs {scenario.bundle_component} component) — pass-with-note"
            ),
            "input_s3": input_s3,
        }

    if not successes:
        first_err = failures[0][2] if failures else "unknown"
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": f"all {len(results)} async requests failed: {first_err[:120]}",
            "error": first_err,
            "input_s3": input_s3,
        }

    transcript, conf, dur = _extract_transcript(successes[0][1] or {})

    presence_ok = True
    presence_note = ""
    if scenario.presence_check:
        haystack = (
            json.dumps(successes[0][1] or {}).lower()
            if scenario.presence_in_full_response
            else transcript.lower()
        )
        presence_ok = scenario.presence_check.lower() in haystack
        presence_note = f"presence({scenario.presence_check})={'ok' if presence_ok else 'MISSING'}"

    w_ratio, s, d, i, n = wer(expected, transcript)
    notes = [
        f"reqs={len(results)}",
        f"success={len(successes)}",
        f"conf={conf:.1%}",
        f"audio_dur={dur:.1f}s",
    ]
    if failures:
        notes.append(f"failed={len(failures)}")
    if presence_note:
        notes.append(presence_note)
    if scenario.notes:
        notes.append(scenario.notes)
    notes.append(f"input={input_s3}")

    ok = not failures and w_ratio <= scenario.wer_threshold and presence_ok

    return {
        "scenario": scenario.name,
        "ok": ok,
        "wer": w_ratio,
        "sdi": (s, d, i),
        "words": n,
        "elapsed_s": elapsed,
        "notes": " ".join(notes),
        "transcript_head": transcript[:200],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end correctness test for a batch SageMaker STT endpoint. "
            "Downloads spacewalk.wav, multiplies it to ~15 min, runs the "
            "scenario set matching the endpoint's configured transport "
            "(--mode sync|async), validates each transcript against the "
            "known reference via WER."
        )
    )
    p.add_argument("endpoint_name", nargs="?", default=None, help="Batch SageMaker endpoint name")
    p.add_argument(
        "--mode",
        choices=["sync", "async"],
        default=None,
        help=(
            "Which scenario set to run, matched to the endpoint's "
            "configured transport. `sync` invokes `invoke_endpoint` against "
            "the short-form file; `async` uses `invoke_endpoint_async` with "
            "S3 in/out. A batch endpoint serves only one of these — pick "
            "the one its EndpointConfig was created with. Required unless "
            "--list."
        ),
    )
    p.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket for async input uploads. Required with --mode async; "
             "ignored when --mode sync.",
    )
    p.add_argument("--upload-prefix", default=DEFAULT_UPLOAD_PREFIX,
                   help=f"S3 key prefix for uploads (default: {DEFAULT_UPLOAD_PREFIX!r})")
    p.add_argument("--region", default=DEFAULT_REGION,
                   help=f"AWS region (default: {DEFAULT_REGION})")
    p.add_argument("--fips", action="store_true",
                   help="Run the battery against the FIPS 140-3 SageMaker endpoints "
                        "(runtime-fips./api-fips.sagemaker.<region>.amazonaws.com, "
                        "s3-fips). OFF by default. Not available in every region — see "
                        "https://docs.aws.amazon.com/general/latest/gr/rande.html"
                        "#FIPS-endpoints")
    p.add_argument("--model", default="nova-3", help="Deepgram model (default: nova-3)")
    p.add_argument("--language", default="en", help="Language code (default: en)")
    p.add_argument("--workdir", default=None, metavar="DIR",
                   help="Fixture + log directory (default: /tmp/dg-sagemaker-e2e/batch/<ts>)")
    p.add_argument("--target-long-form-s", type=float, default=900.0,
                   help="Target duration of the long-form multiplied WAV (default: 900 = 15 min)")
    p.add_argument("--scenarios", default="",
                   help="Comma-separated subset of scenarios. --list to see names.")
    p.add_argument("--list", action="store_true",
                   help="List scenarios for --mode (or all if --mode not set) and exit")
    p.add_argument("--wer-threshold", type=float, default=0.05)
    p.add_argument("--invocation-timeout-s", type=int, default=3600)
    p.add_argument("--poll-interval-s", type=float, default=5.0)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--skip-preflight", action="store_true",
                   help="(async only) skip the IAM/S3 preflight check on the endpoint's "
                        "execution role. Preflight is ON by default to fail-fast when the "
                        "exec role can't read input / write output, which otherwise hangs "
                        "the run silently for the full invocation-timeout.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p


def main() -> int:
    args = _make_parser().parse_args()
    global _FIPS
    _FIPS = args.fips           # must be set before the first aws_client() call
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.list:
        scenarios = (
            default_scenarios(args.mode) if args.mode in ("sync", "async")
            else _all_scenarios()
        )
        print("Available scenarios:")
        for s in scenarios:
            langs = f"  langs={s.supported_languages}" if s.supported_languages else ""
            print(f"  {s.name:<28} [{s.transport}]  {s.description}  "
                  f"(threshold={fmt_wer(s.wer_threshold)}){langs}")
        return 0

    if not args.mode:
        print("ERROR: --mode {sync|async} is required (run with --list to see scenarios).",
              file=sys.stderr)
        return 1
    if not args.endpoint_name:
        print("ERROR: endpoint_name is required (run with --list to see scenarios).",
              file=sys.stderr)
        return 1
    if args.mode == "async" and not args.bucket:
        print("ERROR: --bucket is required with --mode async (S3 input destination).",
              file=sys.stderr)
        return 1

    scenarios = default_scenarios(args.mode)

    if args.scenarios:
        wanted = {tok.strip() for tok in args.scenarios.split(",") if tok.strip()}
        unknown = wanted - {s.name for s in scenarios}
        if unknown:
            print(f"ERROR: unknown scenario(s) for --mode {args.mode}: {sorted(unknown)}",
                  file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s.name in wanted]

    if not scenarios:
        print("ERROR: no scenarios to run.", file=sys.stderr)
        return 1

    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        workdir = Path(tempfile.gettempdir()) / "dg-sagemaker-e2e" / "batch" / ts
    workdir.mkdir(parents=True, exist_ok=True)

    # AWS session
    try:
        session = boto3.Session(region_name=args.region)
        aws_client(session, "sts").get_caller_identity()
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"ERROR: AWS credentials missing: {e}", file=sys.stderr)
        return 2

    if args.mode == "async" and not args.skip_preflight:
        try:
            preflight_async_iam(
                session, args.region, args.endpoint_name, args.bucket, args.upload_prefix,
            )
        except PreflightError as e:
            print(f"ERROR: async preflight failed:\n{e}", file=sys.stderr)
            return 2
    elif args.mode == "async" and args.skip_preflight:
        logger.warning("preflight: SKIPPED via --skip-preflight (exec-role IAM unchecked)")

    short_wav = workdir / "spacewalk.wav"
    long_wav = workdir / "spacewalk-15min.wav"

    print("=" * 80)
    print(f"Endpoint:    {args.endpoint_name}")
    print(f"Mode:        {args.mode}")
    print(f"Region:      {args.region}")
    _rt_url = aws_client(boto3.Session(region_name=args.region),
                         "sagemaker-runtime", args.region).meta.endpoint_url
    print(f"Runtime URL: {_rt_url}")
    print(f"FIPS:        {'yes' if '-fips.' in _rt_url else 'no'}")
    if args.mode == "async":
        print(f"Bucket:      {args.bucket}")
    print(f"Workdir:     {workdir}")
    print("=" * 80)

    download_sample(short_wav, force=args.force_download)
    sr, ch, dur = validate_pcm16(short_wav)
    print(f"Sample:      {short_wav.name}  {sr} Hz  {ch}ch  {dur:.2f}s")
    runnable = [s for s in scenarios if language_supported(s.supported_languages, args.language)]
    if any(s.use_long_form for s in runnable):
        if not long_wav.exists() or args.force_download:
            loops = multiply_wav(short_wav, long_wav, args.target_long_form_s)
        else:
            _, _, long_dur = validate_pcm16(long_wav)
            loops = max(1, round(long_dur / dur))
        _, _, long_dur = validate_pcm16(long_wav)
        print(f"Long-form:   {long_wav.name}  {long_dur:.0f}s ({loops} loops) "
              f"({long_wav.stat().st_size / (1024*1024):.1f} MB)")
    else:
        loops = 1

    print()
    rows: list[dict] = []
    for scenario in scenarios:
        print(f"--> {scenario.name}  [{scenario.transport}]  ({scenario.description})")
        if not language_supported(scenario.supported_languages, args.language):
            reason = (f"language not supported: scenario supports "
                      f"{scenario.supported_languages}, run --language={args.language!r}")
            print(f"    SKIP  {reason}")
            rows.append({
                "scenario": scenario.name, "ok": True, "skipped": True,
                "wer": None, "sdi": (0, 0, 0), "words": 0, "elapsed_s": 0.0,
                "notes": f"SKIPPED ({reason})",
            })
            continue
        if scenario.transport == "sync":
            row = run_sync_scenario(
                scenario,
                session=session,
                region=args.region,
                endpoint=args.endpoint_name,
                short_wav=short_wav,
                long_wav=long_wav,
                long_loops=loops,
                model=args.model,
                language=args.language,
            )
        else:
            row = run_async_scenario(
                scenario,
                session=session,
                region=args.region,
                endpoint=args.endpoint_name,
                bucket=args.bucket,
                upload_prefix=args.upload_prefix,
                short_wav=short_wav,
                long_wav=long_wav,
                long_loops=loops,
                model=args.model,
                language=args.language,
                invocation_timeout_s=args.invocation_timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
        rows.append(row)
        flag = "SKIP" if row.get("skipped") else ("PASS" if row["ok"] else "FAIL")
        wer_str = "-" if row.get("skipped") else fmt_wer(row["wer"])
        print(f"    {flag}  WER={wer_str}  elapsed={row['elapsed_s']:.1f}s  {row['notes']}")

    print()
    passed, failed = print_summary_table(rows, wer_threshold=args.wer_threshold)
    (workdir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nFull results: {workdir / 'results.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

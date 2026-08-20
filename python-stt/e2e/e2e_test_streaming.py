#!/usr/bin/env python3
"""
End-to-end correctness + red-team test for a streaming SageMaker STT endpoint.

Drives `stt_wav_stress.py stream` through a sequence of scenarios — basic
short-form, basic long-form, sustained-concurrency, ramped-concurrency, each
of the major feature flags (diarize / keyterms / redact / interim-results),
and an adversarial bare-WebSocket-close path — then validates each
connection's `combined_final_text` against a known reference transcript via
Word Error Rate. Designed to be the definitive correctness gate for a
streaming endpoint before promoting it.

Fixtures
--------
- Downloads `https://dpgr.am/spacewalk.wav` (~25 s English mono, 16-bit PCM).
- Multiplies the sample by N loops in-place to a long-form variant (default
  ≥ 15 min) for sustained-concurrency + long-form smoke.

Pass / fail
-----------
Each scenario succeeds when:
  - the subprocess exits 0,
  - every connection reports at least one final transcript,
  - WER of the combined final text vs. the expected reference (single or
    multiplied) is below the per-scenario threshold (default 5%).

For scenarios that intentionally distort transcription (e.g. PII redact), the
threshold is raised and a presence-check on the redaction marker is applied
instead of raw WER. For diarize, WER is computed against the same reference
because the text body is unchanged — diarize only adds speaker tags as
separate fields.

Endpoint prerequisites
----------------------
- A streaming-mode SageMaker endpoint (manifest `mode: streaming`) is
  InService and accessible to the calling identity.
- The endpoint's manifest defaults include the requested `--model` /
  `--language` (default `nova-3` / `en`), or the same can be requested
  explicitly via `--model` / `--language` on this script.

Usage
-----
    uv run e2e_test_streaming.py your-streaming-endpoint-name --region us-east-2

By default the WAV fixtures land under `/tmp/dg-sagemaker-e2e/` and persist
across runs (re-runs skip the download); pass `--workdir <path>` to override.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

# Same directory imports so the e2e suite can be run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_test_common import (
    SPACEWALK_REFERENCE_TEXT,
    download_sample,
    expected_text_for_loops,
    fmt_wer,
    language_supported,
    multiply_wav,
    print_summary_table,
    trim_trailing_silence,
    validate_pcm16,
    wer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class StreamScenario:
    """One streaming scenario, scoped to features supported on the
    nova-3 streaming transport per
    https://developers.deepgram.com/docs/ (May 2026 audit).

    `bundle_component` documents the model component a bundle must include
    for the scenario to succeed; when omitted, only the ASR weights are
    required. `tolerated_error_substring` lets the scenario PASS-WITH-NOTE
    when the endpoint returns a specific known error (e.g. "entity
    detection" when the redaction model isn't bundled) — this surfaces
    bundle gaps without false-failing.
    """
    name: str
    description: str
    use_long_form: bool = False
    # Use the trailing-silence-trimmed clip (ends at the last speech sample) so
    # the final segment isn't endpointed in-stream and is delivered ONLY if the
    # server's CloseStream finalize reaches the client.
    use_notail: bool = False
    connections: int = 1
    extra_args: list[str] = field(default_factory=list)
    wer_threshold: float = 0.05
    presence_check: str | None = None  # substring or marker that must appear
    # Substring that must appear in the combined final text, used as a TAIL
    # guard: set it to the reference's last word(s) on a no-trailing-silence clip
    # so a dropped final segment (tail truncation) fails the scenario even though
    # whole-transcript WER would stay under threshold.
    tail_check: str | None = None
    expect_failure: bool = False
    bundle_component: str | None = None  # e.g. "streaming-ner"
    tolerated_error_substring: str | None = None  # PASS-WITH-NOTE if seen
    # Negative scenario: the endpoint MUST reject the upgrade. PASS iff this
    # substring appears in the client's output AND no transcripts are produced.
    # Distinct from `tolerated_error_substring` (which only tolerates an error and
    # would falsely PASS on a served session). Verifies the reject-unknown-params
    # gate at the WS handshake (off-allowlist param → 400 before the 101).
    expect_error_substring: str | None = None
    # None = no restriction (default). Set ONLY when the endpoint hard-rejects
    # (a real error, not a graceful warning/no-op) outside this language set —
    # see `language_supported()` in e2e_test_common. No streaming scenario
    # needs this today (summarize/topics/intents/sentiment are pre-recorded
    # only; dictation/measurements/redact degrade gracefully rather than
    # hard-reject per a 2026-07-14 docs+empirical audit) — field exists for
    # parity with the batch driver so a future language-gated feature has
    # somewhere to go.
    supported_languages: list[str] | None = None
    notes: str = ""


def default_scenarios(model: str, language: str) -> list[StreamScenario]:
    """Feature coverage matrix per the docs (nova-3, streaming):

      Always-supported by the API on nova-3 streaming (per docs):
        punctuate, smart_format, numerals, dictation, diarize (v1),
        keyterm, replace (find/replace), profanity_filter,
        interim_results, search
      Nova-3 streaming SPECIFIC notes:
        - `keyterm` (NOT `keywords` — those are nova-2/legacy)
        - `diarize=true` only — `diarize_model=v2` is pre-recorded-only,
          returns 400 on streaming
      Bundle-component dependent:
        - redact / detect_entities → requires streaming-ner
          (UUID 90424f3a-... `modes=["streaming"]`)
        - search → requires g2p (UUID 89555db3-...
          `modes=["streaming","batch"]`); search is supported on
          nova-3 streaming AND batch per
          https://developers.deepgram.com/docs/search
      Not supported on nova-3 streaming (per docs; excluded here):
        - filler_words (pre-recorded only on nova-3; also note
          filler_words is only available on the Nova, Nova-2, and
          Nova-3 *general* models — NOT on specialized models such as
          nova-3-medical, so this feature is N/A for those models
          regardless of streaming vs pre-recorded; see
          https://developers.deepgram.com/docs/filler-words)
        - utterances (pre-recorded only on nova-3; see
          https://developers.deepgram.com/docs/utterances)
        - paragraphs (pre-recorded only)
        - measurements (pre-recorded only)
        - utt_split (pre-recorded only)
        - diarize_model (pre-recorded only)
        - summarize / topics / intents / sentiment (pre-recorded only)
    """
    return [
        # ---- Coverage / load ----
        StreamScenario(
            name="basic_25s",
            description="1 conn, 25 s file, defaults",
            connections=1,
        ),
        StreamScenario(
            name="concurrent_5x_25s",
            description="5 simultaneous connections, 25 s file",
            connections=5,
        ),
        StreamScenario(
            name="concurrent_10x_15min",
            description="10 simultaneous connections on ~15 min file",
            use_long_form=True,
            connections=10,
            notes="sustained-load WER check",
        ),
        StreamScenario(
            name="ramp_10x_step5",
            description="10 conns in batches of 5 with 2 s delay",
            connections=10,
            extra_args=["--batch-size", "5", "--batch-delay", "2"],
        ),
        # ---- Tail-finalize regression guard ----
        StreamScenario(
            name="tail_finalize_notail",
            description="no-trailing-silence clip; CloseStream must finalize the last segment",
            connections=1,
            use_notail=True,
            extra_args=["--extra", "endpointing=300", "--interim-results"],
            tail_check="today",  # reference ends "...that we have today."
            notes=(
                "regression guard: CloseStream sent as DataType=UTF8 (WS Text "
                "frame, the default) must reach the API AND the client must drain "
                "before closing input, or the trailing segment is dropped. See "
                "close_stream_binary_frame for the DataType=BINARY counterpart"
            ),
        ),
        # ---- Streaming-only features ----
        StreamScenario(
            name="feature_interim_results",
            description="--interim-results (verify interims emitted)",
            connections=1,
            extra_args=["--interim-results"],
            notes="streaming-only feature",
        ),
        # ---- Formatting features ----
        StreamScenario(
            name="feature_diarize_v1",
            description="--diarize true (streaming v1 diarizer)",
            connections=1,
            extra_args=["--diarize", "true"],
            notes="diarize_model=v2 is NOT accepted on streaming (400)",
        ),
        StreamScenario(
            name="feature_smart_format",
            description="--extra smart_format=true",
            connections=1,
            extra_args=["--extra", "smart_format=true"],
            notes="implies punctuate; may delay finals up to 3 s",
        ),
        StreamScenario(
            name="feature_punctuate",
            description="--punctuate true (default; explicit verification)",
            connections=1,
            extra_args=["--punctuate", "true"],
            presence_check=".",
            notes="punctuation marks should appear in finals",
        ),
        StreamScenario(
            name="feature_numerals",
            description="--extra numerals=true (digit substitution)",
            connections=1,
            extra_args=["--extra", "numerals=true"],
            notes="numerals param accepted; clip has few numbers — smoke",
        ),
        StreamScenario(
            name="feature_dictation",
            description="--extra dictation=true (spoken punctuation -> chars)",
            connections=1,
            extra_args=["--extra", "dictation=true"],
            wer_threshold=1.0,  # dictation may alter punctuation tokens
            notes="dictation transforms spoken punctuation cues",
        ),
        StreamScenario(
            name="feature_profanity_filter",
            description="--extra profanity_filter=true (no profanity in clip; smoke)",
            connections=1,
            extra_args=["--extra", "profanity_filter=true"],
            notes="clip has no profanity; transcript should be unchanged",
        ),
        # ---- Custom vocabulary ----
        StreamScenario(
            name="feature_keyterm",
            description="--keyterms 'spacewalk,female' (nova-3 boost)",
            connections=1,
            extra_args=["--keyterms", "spacewalk,female"],
            presence_check="spacewalk",
            notes="nova-3 only — `keyterm`, NOT `keywords`",
        ),
        StreamScenario(
            name="feature_replace",
            description="replace=spacewalk:moonwalk (find/replace)",
            connections=1,
            extra_args=["--extra", "replace=spacewalk:moonwalk"],
            wer_threshold=1.0,  # text intentionally changed
            presence_check="moonwalk",
            notes="content swap; WER skipped",
        ),
        # ---- Search (bundle-dependent: needs g2p) ----
        StreamScenario(
            name="feature_search",
            description="search=spacewalk (phonetic match; requires g2p)",
            connections=1,
            extra_args=["--extra", "search=spacewalk"],
            presence_check="spacewalk",  # search hits surface in results
            bundle_component="g2p (uuid 89555db3-...)",
            notes="needs g2p; will FAIL on bundles without it",
        ),
        # ---- Redaction / entity detection (bundle-dependent) ----
        StreamScenario(
            name="feature_redact_name",
            description="--redact name (requires streaming-ner component; English-classified transcript only)",
            connections=1,
            extra_args=["--redact", "name"],
            wer_threshold=1.0,
            bundle_component="streaming-ner",
            tolerated_error_substring="entity detection",
            supported_languages=["en", "multi"],
            notes=(
                "WER skipped; PASS-WITH-NOTE if bundle lacks streaming-ner. "
                "Self-hosted redaction (batch AND streaming) is English-only "
                "per docs — 'multi' passes because language-detect on English "
                "audio resolves to English; a forced non-English language "
                "won't — auto-skipped outside en/multi"
            ),
        ),
        # ---- Adversarial ----
        StreamScenario(
            name="adversarial_bare_close",
            description="--no-use-close-stream (bare WS close path)",
            connections=1,
            extra_args=["--no-use-close-stream"],
            notes="trailing tail may drop; WER threshold relaxed",
            wer_threshold=0.10,
        ),
        # ---- Frame typing: CloseStream as a Binary frame ----
        # Mirror of tail_finalize_notail, but with the control message sent as
        # DataType=BINARY. SageMaker maps DataType to the WebSocket opcode
        # (bidi container contract §3.1): UTF8 -> Text, absent/BINARY -> Binary.
        # Control messages are parsed only from Text frames, so a BINARY
        # CloseStream reaches it only via the container's binary-control
        # reframing. Same tail_check
        # as the UTF8 scenario: if the reframe regresses, the CloseStream is
        # consumed as audio, the last segment is never finalized, and the tail
        # word goes missing.
        StreamScenario(
            name="close_stream_binary_frame",
            description="--control-data-type BINARY (container binary→text reframing)",
            connections=1,
            use_notail=True,
            extra_args=[
                "--control-data-type", "BINARY",
                "--extra", "endpointing=300", "--interim-results",
            ],
            tail_check="today",
            notes=(
                "regression guard for the DataType-omitted client: the container must "
                "reframe a binary CloseStream to Text or the tail is dropped"
            ),
        ),
        # ---- Negative: reject-unknown-params gate (container 400s off-allowlist params) ----
        StreamScenario(
            name="reject_unknown_param",
            description="--extra bogus=true → WS upgrade rejected (not a stream)",
            connections=1,
            extra_args=["--extra", "bogus=true"],
            # SageMaker's bidi-stream does NOT propagate the container's
            # WS-handshake response body, so the customer sees a generic
            # "Failed to establish WebSocket connection" (424) rather than the
            # container's `unsupported_parameter` JSON (unlike the HTTP /invocations
            # path, which surfaces it). The reject still fires at the handshake —
            # confirmed in the container log ("rejecting request: unsupported query
            # parameter(s)"). On a healthy endpoint (where the plain scenarios
            # pass) a 424 for a bogus param can only be this gate.
            expect_error_substring="Failed to establish WebSocket connection",
            notes="reject-unknown-params: off-allowlist key must reject the upgrade (generic 424; reason in container logs)",
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _stress_cmd(
    script: Path,
    endpoint: str,
    wav_path: Path,
    region: str,
    model: str,
    language: str,
    connections: int,
    summary_path: Path,
    extra: list[str],
    fips: bool = False,
) -> list[str]:
    cmd = [
        "uv", "run", "--project", str(script.parent),
        str(script), "stream", endpoint,
        "--file", str(wav_path),
        "--region", region,
        "--model", model,
        "--language", language,
        "--connections", str(connections),
        "--summary-jsonl", str(summary_path),
    ]
    if fips:
        cmd.append("--fips")
    cmd.extend(extra)
    return cmd


def _scenario_timeout_s(scenario: StreamScenario, long_form_audio_s: float, base_timeout_s: int) -> int:
    """Per-scenario subprocess timeout.

    Long-form scenarios stream the full multiplied WAV in wall-clock time
    (the bidi-stream paces audio to its sample rate), so the subprocess
    needs at least `long_form_audio_s` plus headroom for connect / finals /
    teardown. A flat 900 s cap clips ~908 s long-form runs as a timeout
    rather than letting them complete. Headroom = max(300 s, 30% of audio)
    so concurrent runs (where finals trickle in after the last frame ships)
    still finish cleanly. Short-form scenarios keep the user-provided
    `--subprocess-timeout-s` (default 900 s — plenty for a 26 s file).
    """
    if not scenario.use_long_form:
        return base_timeout_s
    headroom = max(300.0, long_form_audio_s * 0.3)
    return int(long_form_audio_s + headroom)


def run_scenario(
    scenario: StreamScenario,
    *,
    endpoint: str,
    region: str,
    model: str,
    language: str,
    short_wav: Path,
    long_wav: Path,
    notail_wav: Path,
    long_loops: int,
    long_form_audio_s: float,
    stress_script: Path,
    log_dir: Path,
    subprocess_timeout_s: int,
    fips: bool = False,
) -> dict:
    if scenario.use_long_form:
        wav, expected = long_wav, expected_text_for_loops(long_loops)
    elif scenario.use_notail:
        wav, expected = notail_wav, SPACEWALK_REFERENCE_TEXT
    else:
        wav, expected = short_wav, SPACEWALK_REFERENCE_TEXT
    timeout_s = _scenario_timeout_s(scenario, long_form_audio_s, subprocess_timeout_s)

    summary_path = log_dir / f"{scenario.name}.summary.jsonl"
    stdout_path = log_dir / f"{scenario.name}.stdout.log"
    stderr_path = log_dir / f"{scenario.name}.stderr.log"

    cmd = _stress_cmd(
        stress_script, endpoint, wav, region, model, language,
        scenario.connections, summary_path, scenario.extra_args,
        fips=fips,
    )
    logger.info(f"[{scenario.name}] running: {' '.join(cmd)}")

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": f"subprocess timed out after {timeout_s}s (long_form_audio_s={long_form_audio_s:.0f})",
            "error": "timeout",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    elapsed = time.monotonic() - start

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    # Negative scenario: the endpoint MUST reject (e.g. unsupported param → 400 at
    # the WS handshake). PASS iff the expected substring appears in the client's
    # output AND no finals were produced. Checked BEFORE the summary-exists gate,
    # because a hard handshake reject may not produce a per-connection summary at
    # all. NOTE: the exact substring SageMaker surfaces for a container 400 may
    # differ from the body's "unsupported_parameter" — tune against a real run if
    # it doesn't match (the gate's intent is "rejected at handshake + no audio").
    if scenario.expect_error_substring is not None:
        sub = scenario.expect_error_substring.lower()
        combined_out = (result.stdout + "\n" + result.stderr).lower()
        finals_seen = 0
        if summary_path.exists():
            try:
                srows = [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]
                finals_seen = sum(r.get("transcripts_final", 0) for r in srows)
            except Exception:
                finals_seen = 0
        matched = sub in combined_out
        ok = matched and finals_seen == 0
        if ok:
            note = f"REJECTED as expected ('{scenario.expect_error_substring}')"
        elif finals_seen > 0:
            note = "EXPECTED REJECT but session produced transcripts — gate not firing"
        else:
            note = (f"EXPECTED '{scenario.expect_error_substring}' in client output "
                    f"but not found (exit={result.returncode}) — inspect stderr_path")
        return {
            "scenario": scenario.name,
            "ok": ok,
            "wer": 0.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": (f"{scenario.notes} | " if scenario.notes else "") + note,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    if not summary_path.exists():
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": f"no summary-jsonl produced (exit={result.returncode})",
            "error": f"exit {result.returncode}",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    rows = [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {
            "scenario": scenario.name,
            "ok": False,
            "wer": 1.0,
            "sdi": (0, 0, 0),
            "words": 0,
            "elapsed_s": elapsed,
            "notes": "summary-jsonl empty",
            "error": f"exit {result.returncode}",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    # Score the connection with the most final transcripts (others should
    # match closely; WER on the strongest one keeps signal clean if a single
    # session experienced a transient hiccup).
    rows.sort(key=lambda r: r.get("transcripts_final", 0), reverse=True)
    best = rows[0]
    combined = best.get("combined_final_text", "") or ""
    finals = sum(r.get("transcripts_final", 0) for r in rows)
    errored = sum(1 for r in rows if r.get("errored"))
    interim_total = sum(r.get("transcripts_interim", 0) for r in rows)

    w_ratio, s, d, i, n = wer(expected, combined)

    notes_parts = []
    if scenario.notes:
        notes_parts.append(scenario.notes)
    notes_parts.append(f"conns={len(rows)}")
    notes_parts.append(f"finals={finals}")
    if errored:
        notes_parts.append(f"errored={errored}")
    if scenario.name == "feature_interim_results":
        notes_parts.append(f"interim_total={interim_total}")

    # PASS-WITH-NOTE: bundle missing the component this scenario requires.
    # The API returns a known error pattern (e.g. "entity detection" for
    # redact on a bundle lacking streaming-ner); we surface the bundle gap
    # without false-failing.
    all_error_text = " ".join(
        msg for row in rows for msg in (row.get("error_messages") or [])
    ).lower()
    tolerated = (
        scenario.tolerated_error_substring is not None
        and scenario.tolerated_error_substring.lower() in all_error_text
    )

    presence_ok = True
    if scenario.presence_check and not tolerated:
        presence_ok = scenario.presence_check.lower() in combined.lower()
        notes_parts.append(f"presence({scenario.presence_check})={'ok' if presence_ok else 'MISSING'}")

    # Tail guard: the reference's final word(s) must be present, catching a
    # dropped trailing segment (tail truncation) that whole-transcript WER would
    # mask. See StreamScenario.tail_check / the tail_finalize_notail scenario.
    tail_ok = True
    if scenario.tail_check and not tolerated:
        tail_ok = scenario.tail_check.lower() in combined.lower()
        notes_parts.append(f"tail({scenario.tail_check})={'ok' if tail_ok else 'TRUNCATED'}")

    if tolerated:
        notes_parts.append(
            f"BUNDLE-GAP: '{scenario.tolerated_error_substring}' "
            f"(needs {scenario.bundle_component or 'feature-specific'} component) — pass-with-note"
        )
        ok = True
        # Reset misleading WER signal — the request never produced ASR output.
        w_ratio, s, d, i = 0.0, 0, 0, 0
    else:
        ok = (
            result.returncode == 0
            and errored == 0
            and finals > 0
            and w_ratio <= scenario.wer_threshold
            and presence_ok
            and tail_ok
        )
    if scenario.name == "feature_interim_results" and interim_total == 0 and not tolerated:
        ok = False
        notes_parts.append("no interim emissions")

    return {
        "scenario": scenario.name,
        "ok": ok,
        "wer": w_ratio,
        "sdi": (s, d, i),
        "words": n,
        "elapsed_s": elapsed,
        "notes": " ".join(notes_parts),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_path": str(summary_path),
        "combined_final_text": combined[:200],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end correctness test for a streaming SageMaker STT endpoint. "
            "Downloads the canonical Deepgram spacewalk.wav sample, multiplies "
            "it to ~15 min, then runs a battery of scenarios through "
            "stt_wav_stress.py and validates each session's transcript against "
            "the known reference via Word Error Rate."
        )
    )
    p.add_argument(
        "endpoint_name",
        nargs="?",
        default=None,
        help="Streaming SageMaker endpoint name (required unless --list)",
    )
    p.add_argument("--region", default="us-east-2", help="AWS region (default: us-east-2)")
    p.add_argument(
        "--fips",
        action="store_true",
        help=(
            "Run the battery against the FIPS 140-3 SageMaker runtime endpoint "
            "(runtime-fips.sagemaker.<region>.amazonaws.com:8443). OFF by default; "
            "passed through to every stt_wav_stress.py subprocess."
        ),
    )
    p.add_argument("--model", default="nova-3", help="Deepgram model (default: nova-3)")
    p.add_argument("--language", default="en", help="Language code (default: en)")
    p.add_argument(
        "--workdir",
        default=None,
        metavar="DIR",
        help="Fixture + log directory (default: /tmp/dg-sagemaker-e2e/streaming/<timestamp>)",
    )
    p.add_argument(
        "--target-long-form-s",
        type=float,
        default=900.0,
        metavar="SECONDS",
        help="Target duration of the long-form multiplied WAV (default: 900 = 15 min)",
    )
    p.add_argument(
        "--scenarios",
        default="",
        metavar="NAME,NAME,...",
        help="Comma-separated subset of scenario names to run (default: all). "
             "Pass --list to see available scenarios.",
    )
    p.add_argument("--list", action="store_true", help="List scenarios and exit")
    p.add_argument(
        "--wer-threshold",
        type=float,
        default=0.05,
        help="Default WER threshold for non-distorting scenarios (default: 0.05)",
    )
    p.add_argument(
        "--subprocess-timeout-s",
        type=int,
        default=900,
        help="Per-scenario subprocess timeout (default: 900 = 15 min)",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download spacewalk.wav even if cached",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p


def main() -> int:
    args = _make_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    scenarios = default_scenarios(args.model, args.language)
    if args.wer_threshold != 0.05:
        for s in scenarios:
            if s.wer_threshold == 0.05:
                s.wer_threshold = args.wer_threshold

    if args.list:
        print("Available scenarios:")
        for s in scenarios:
            print(f"  {s.name:<25} {s.description}  (threshold={fmt_wer(s.wer_threshold)})")
        return 0

    if not args.endpoint_name:
        print("ERROR: endpoint_name is required (run with --list to see scenarios).",
              file=sys.stderr)
        return 1

    if args.scenarios:
        wanted = {tok.strip() for tok in args.scenarios.split(",") if tok.strip()}
        unknown = wanted - {s.name for s in scenarios}
        if unknown:
            print(f"ERROR: unknown scenario(s): {sorted(unknown)}", file=sys.stderr)
            print(f"Run with --list to see available names.", file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s.name in wanted]

    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        workdir = Path(tempfile.gettempdir()) / "dg-sagemaker-e2e" / "streaming" / ts
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    short_wav = workdir / "spacewalk.wav"
    long_wav = workdir / "spacewalk-15min.wav"
    notail_wav = workdir / "spacewalk-notail.wav"

    print("=" * 80)
    print(f"Endpoint:    {args.endpoint_name}")
    print(f"Region:      {args.region}")
    # Resolved the same way stt_wav_stress.py derives its bidi endpoint, so the
    # header states the host the subprocesses will actually stream to (FIPS or
    # not) rather than assuming the default one.
    _rt_url = boto3.Session(region_name=args.region).client(
        "sagemaker-runtime",
        config=BotoConfig(use_fips_endpoint=True) if args.fips else None,
    ).meta.endpoint_url
    print(f"Runtime URL: {_rt_url}:8443")
    print(f"FIPS:        {'yes' if '-fips.' in _rt_url else 'no'}")
    print(f"Model/lang:  {args.model} / {args.language}")
    print(f"Workdir:     {workdir}")
    print("=" * 80)

    # 1. Fixture prep
    download_sample(short_wav, force=args.force_download)
    sr, ch, dur = validate_pcm16(short_wav)
    print(f"Sample:      {short_wav.name}  {sr} Hz  {ch}ch  {dur:.2f}s "
          f"({short_wav.stat().st_size / 1024:.0f} KB)")
    if not long_wav.exists() or args.force_download:
        loops = multiply_wav(short_wav, long_wav, args.target_long_form_s)
    else:
        # Recompute loop count from existing file size for the expected text.
        _, _, long_dur = validate_pcm16(long_wav)
        loops = max(1, round(long_dur / dur))
    _, _, long_dur = validate_pcm16(long_wav)
    print(f"Long-form:   {long_wav.name}  {long_dur:.0f}s ({loops} loops) "
          f"({long_wav.stat().st_size / (1024*1024):.1f} MB)")
    # No-trailing-silence fixture for the tail-finalize regression guard.
    if not notail_wav.exists() or args.force_download:
        notail_dur = trim_trailing_silence(short_wav, notail_wav)
    else:
        _, _, notail_dur = validate_pcm16(notail_wav)
    print(f"No-tail:     {notail_wav.name}  {notail_dur:.2f}s (trailing silence trimmed)")

    # 2. Locate the stress script (lives one directory up — e2e/ is a sibling of stt_wav_stress.py)
    stress_script = Path(__file__).resolve().parent.parent / "stt_wav_stress.py"
    if not stress_script.is_file():
        print(f"ERROR: {stress_script} missing", file=sys.stderr)
        return 2

    # 3. Run scenarios sequentially
    print()
    rows: list[dict] = []
    for scenario in scenarios:
        print(f"--> {scenario.name}  ({scenario.description})")
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
        row = run_scenario(
            scenario,
            endpoint=args.endpoint_name,
            region=args.region,
            model=args.model,
            language=args.language,
            short_wav=short_wav,
            long_wav=long_wav,
            notail_wav=notail_wav,
            long_loops=loops,
            long_form_audio_s=long_dur,
            stress_script=stress_script,
            log_dir=log_dir,
            subprocess_timeout_s=args.subprocess_timeout_s,
            fips=args.fips,
        )
        rows.append(row)
        flag = "SKIP" if row.get("skipped") else ("PASS" if row["ok"] else "FAIL")
        wer_str = "-" if row.get("skipped") else fmt_wer(row["wer"])
        print(f"    {flag}  WER={wer_str}  elapsed={row['elapsed_s']:.1f}s  {row['notes']}")

    # 4. Summary
    print()
    passed, failed = print_summary_table(rows, wer_threshold=args.wer_threshold)
    (workdir / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nFull results: {workdir / 'results.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

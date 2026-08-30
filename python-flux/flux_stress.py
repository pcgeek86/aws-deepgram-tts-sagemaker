#!/usr/bin/env python3
"""
Deepgram Flux STT SageMaker Stress Test

Streams audio to multiple simultaneous bidirectional connections to a Deepgram
Flux model deployed on SageMaker.  Supports two input modes:

  file        – Stream a WAV file at real-time pace (repeatable load testing).
  microphone  – Capture live microphone input via PyAudio.

Flux uses the /v2/listen endpoint and a turn-based message protocol with
integrated end-of-turn detection, replacing the /v1/listen channel/alternatives
format used by Nova models.

Supported Flux input messages:
  - Binary audio frames (raw PCM bytes)
  - Configure   (update thresholds/keyterms mid-stream)
  - CloseStream (gracefully terminate the stream)

Note: Flux has no JSON ``{"type":"KeepAlive"}`` control message — that is a
Nova-only mechanism. Flux keeps a connection alive at the WebSocket layer
(standard WS ping). The SageMaker bidirectional-stream transport exposes no
WS-ping primitive (only UTF8/BINARY payload parts), so to hold a connection
open through a silence gap this client sends a minimal binary frame (a tiny
silent PCM chunk); any frame resets the server's idle timer. See
``send_keepalive_ping`` / ``--idle-before-audio``.

Supported Flux output messages:
  - Connected        (stream established)
  - TurnInfo         (transcript updates; event field indicates state)
  - ConfigureSuccess (Configure acknowledged)
  - ConfigureFailure (Configure rejected)
  - Error            (fatal; connection terminated by server)
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable
import wave
from queue import Queue
from urllib.parse import quote
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    ModelStreamError,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme

# Configuration constants
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL = "flux-general-en"
FLUX_API_PATH = "v2/listen"
DEFAULT_MIC_SAMPLE_RATE = 16000

# Deepgram recommends 80ms audio chunks for Flux
AUDIO_CHUNK_MS = 80

# Minimal "keepalive ping" payload: a single 16-bit silent PCM sample (2 bytes,
# value 0). Flux has no JSON KeepAlive control message (Nova-only); to keep a
# connection alive through a silence gap we send the smallest valid binary frame
# we can. It must be sample-aligned (even byte count for linear16) so it doesn't
# corrupt the real audio stream, and silence so it has no transcript effect — the
# point is purely to send *a* frame and reset the server's idle timer.
KEEPALIVE_PING_BYTES = b"\x00\x00"

# Default cadence for keepalive pings during an idle hold when --keepalive-interval
# is not given. Well under typical 60s WS/HTTP2 idle timeouts.
DEFAULT_IDLE_KEEPALIVE_S = 10.0

logger = logging.getLogger(__name__)


@dataclass
class MidStreamPlan:
    """Optional mid-stream control-message schedule for a file-mode stream.

    Lets the e2e driver exercise Flux's distinctive in-band control messages
    (Configure / Finalize) and connection-liveness behaviour deterministically
    while a WAV streams:

    - ``idle_before_audio_s``: before streaming any audio, hold the connection
      open for N seconds of silence, sending only a tiny binary keepalive frame
      every ``keepalive_interval_s`` (or :data:`DEFAULT_IDLE_KEEPALIVE_S`). The
      audio that follows must still transcribe correctly — that is the
      end-to-end proof the connection survived the idle gap.
    - ``keepalive_interval_s``: cadence for the tiny binary keepalive frame
      (used during the idle hold above, and also interleaved while audio
      streams). NOT a JSON ``KeepAlive`` message — that is Nova-only.
    - ``reconfigure_after_s``: send a single ``Configure`` once N seconds of
      audio have streamed, applying any of the reconfigure_* fields. Useful for
      asserting ``ConfigureSuccess`` (valid change) or ``ConfigureFailure``
      (e.g. eager_eot_threshold > eot_threshold).
    - ``finalize_at_end``: send a ``Finalize`` after the last audio chunk to
      flush the final turn before ``CloseStream``.
    """
    keepalive_interval_s: float | None = None
    idle_before_audio_s: float | None = None
    reconfigure_after_s: float | None = None
    reconfigure_eot_threshold: float | None = None
    reconfigure_eager_eot_threshold: float | None = None
    reconfigure_eot_timeout_ms: int | None = None
    reconfigure_keyterms: list[str] | None = None
    reconfigure_language_hints: list[str] | None = None
    finalize_at_end: bool = False

    @property
    def active(self) -> bool:
        return (
            self.keepalive_interval_s is not None
            or self.idle_before_audio_s is not None
            or self.reconfigure_after_s is not None
            or self.finalize_at_end
        )


# ---------------------------------------------------------------------------
# Connection – Flux /v2/listen bidirectional stream
# ---------------------------------------------------------------------------

class DeepgramFluxConnection:
    """
    Represents a single bidirectional streaming connection to a Deepgram Flux
    model on SageMaker.

    Handles the Flux v2 protocol: binary audio input, JSON control messages,
    and TurnInfo-based transcript responses with integrated end-of-turn detection.
    """

    def __init__(self, connection_id: int, client: SageMakerRuntimeHTTP2Client, endpoint_name: str):
        self.connection_id = connection_id
        self.client = client
        self.endpoint_name = endpoint_name
        self.stream = None
        self.output_stream = None
        self.is_active = False
        self.response_task = None
        self.chunk_count = 0
        self.byte_count = 0
        self.turn_index = 0
        self.transcript_parts: list[str] = []
        self.close_requested = False
        self.fatal_error_handler: Callable[[str], None] | None = None

        # --- Structured per-connection telemetry (for --summary-jsonl + e2e) ---
        # Flux is turn-based: the authoritative final text for a turn is the
        # EndOfTurn transcript. We accumulate those for a WER-able combined
        # transcript, and count every TurnInfo event + control-message ack so
        # the e2e driver can assert on feature behaviour (eager events emitted,
        # Configure accepted, etc.).
        self.eot_transcripts: list[str] = []          # one per EndOfTurn
        self.event_counts: Counter[str] = Counter()    # TurnInfo event histogram
        self.languages_detected: set[str] = set()       # flux-general-multi only
        self.languages_hinted: set[str] = set()          # flux-general-multi only
        self.connected = False
        self.request_id: str | None = None
        self.configure_success = 0
        self.configure_failure = 0
        # The ConfigureSuccess ack echoes back the *full* applied config
        # (thresholds/keyterms/language_hints), and ConfigureFailure carries a
        # code/description. Counting acks only proves a response arrived;
        # capturing the payload lets the e2e driver verify the server actually
        # applied the requested change (or rejected it for the expected reason).
        self.configure_acks: list[dict] = []      # one per ConfigureSuccess
        self.configure_failures: list[dict] = []   # one per ConfigureFailure
        self.errored = False
        self.error_messages: list[str] = []
        self.session_start_at: float | None = None
        self.first_eot_at: float | None = None
        self.last_eot_at: float | None = None
        self.close_observed_at: float | None = None

    async def start_session(
        self,
        sample_rate: int,
        model: str = DEFAULT_MODEL,
        encoding: str = "linear16",
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keyterms: list[str] | None = None,
        language_hints: list[str] | None = None,
        extra: dict[str, str] | None = None,
    ):
        """
        Open a Flux /v2/listen bidirectional stream on SageMaker.

        Args:
            sample_rate: Sample rate of the audio (e.g. 16000).
            model: Flux model name (default: flux-general-en).
            encoding: Audio encoding (default: linear16).
            eot_threshold: Confidence threshold for EndOfTurn (0.5–0.9).
            eager_eot_threshold: Confidence for EagerEndOfTurn; must be ≤ eot_threshold.
            eot_timeout_ms: Max silence ms before forced EndOfTurn (500–10000).
            keyterms: List of keyterms for boosting recognition accuracy.
            language_hints: Language codes to bias recognition (multilingual models);
                emitted as repeated `language_hint=<code>` query params.
            extra: Arbitrary additional query parameters appended verbatim
                (e.g. {"mip_opt_out": "true", "tag": "e2e"}). Lets callers
                exercise Flux params this client has no dedicated flag for.
        """
        query_params: dict[str, str | int | float] = {
            "model": model,
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        if eot_threshold is not None:
            query_params["eot_threshold"] = eot_threshold
        if eager_eot_threshold is not None:
            query_params["eager_eot_threshold"] = eager_eot_threshold
        if eot_timeout_ms is not None:
            query_params["eot_timeout_ms"] = eot_timeout_ms
        # --extra wins over the typed defaults above when keys collide, so a
        # scenario can override e.g. encoding without a dedicated flag.
        if extra:
            query_params.update(extra)

        query_string = "&".join(f"{k}={quote(str(v))}" for k, v in query_params.items())

        # `keyterm` and `language_hint` are repeatable query params, so they are
        # appended as separate key=value pairs rather than folded into the dict.
        if keyterms:
            keyterm_params = "&".join(f"keyterm={quote(kt)}" for kt in keyterms)
            query_string = f"{query_string}&{keyterm_params}"
        if language_hints:
            hint_params = "&".join(f"language_hint={quote(lh)}" for lh in language_hints)
            query_string = f"{query_string}&{hint_params}"

        logger.debug(f"[Connection {self.connection_id}] Starting Flux session: {query_string}")
        self.session_start_at = time.monotonic()

        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path=FLUX_API_PATH,
            model_query_string=query_string,
        )

        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(stream_input)
        self.is_active = True

        output = await self.stream.await_output()
        self.output_stream = output[1]

        self.response_task = asyncio.create_task(self._process_responses())
        await asyncio.sleep(0.1)

        logger.info(f"[Connection {self.connection_id}] Session started")

    # -------------------------------------------------------------------------
    # Input messages
    # -------------------------------------------------------------------------

    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send a raw PCM audio frame to the stream."""
        if not self.is_active:
            return
        try:
            # Audio is BINARY — explicit rather than relying on the default, so
            # the contrast with the UTF8 control messages in _send_json is
            # visible at the call site.
            payload = RequestPayloadPart(bytes_=audio_bytes, data_type="BINARY")
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
            self.chunk_count += 1
            self.byte_count += len(audio_bytes)
            logger.debug(
                f"[Connection {self.connection_id}] Sent chunk {self.chunk_count} "
                f"({len(audio_bytes)} bytes)"
            )
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error sending audio chunk: {e}")
            self.is_active = False

    async def send_configure(
        self,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keyterms: list[str] | None = None,
        language_hints: list[str] | None = None,
    ):
        """
        Send a Configure message to update thresholds, keyterms, or language
        hints mid-stream (per the Flux Configure schema).

        Args:
            eot_threshold: New EndOfTurn confidence threshold.
            eager_eot_threshold: New EagerEndOfTurn confidence threshold.
            eot_timeout_ms: New silence timeout in milliseconds.
            keyterms: Replacement list of keyterms (replaces the existing list,
                not merged). Note the query-string key is singular `keyterm`,
                but the Configure-message field is plural `keyterms`.
            language_hints: Replacement list of language codes (flux-general-multi
                only). A non-empty list replaces; `[]` clears; ``None`` keeps the
                current hints. Query-string key is singular `language_hint`.
        """
        if not self.is_active:
            return
        message: dict = {"type": "Configure"}
        thresholds = {}
        if eot_threshold is not None:
            thresholds["eot_threshold"] = eot_threshold
        if eager_eot_threshold is not None:
            thresholds["eager_eot_threshold"] = eager_eot_threshold
        if eot_timeout_ms is not None:
            thresholds["eot_timeout_ms"] = eot_timeout_ms
        if thresholds:
            message["thresholds"] = thresholds
        if keyterms is not None:
            message["keyterms"] = keyterms
        if language_hints is not None:
            message["language_hints"] = language_hints

        await self._send_json(message)
        logger.debug(f"[Connection {self.connection_id}] Sent Configure: {message}")

    async def send_keepalive_ping(self):
        """Send a minimal binary frame to keep the connection alive during silence.

        Flux has no JSON ``KeepAlive`` control message (that is Nova-only) and
        the SageMaker bidirectional-stream transport exposes no WebSocket-ping
        primitive — only UTF8/BINARY payload parts. So we send the smallest
        valid binary audio frame we can (:data:`KEEPALIVE_PING_BYTES`, one silent
        16-bit sample); any frame resets the server's idle timer, and silence has
        no transcript effect.
        """
        if not self.is_active:
            return
        try:
            # BINARY on purpose: this is a silent audio sample, not a control
            # message — it resets the idle timer by being audio.
            payload = RequestPayloadPart(bytes_=KEEPALIVE_PING_BYTES, data_type="BINARY")
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
            self.byte_count += len(KEEPALIVE_PING_BYTES)
            logger.debug(f"[Connection {self.connection_id}] Sent keepalive ping")
        except Exception as e:
            logger.error(
                f"[Connection {self.connection_id}] Error sending keepalive ping: {e}"
            )
            self.is_active = False

    async def send_finalize(self):
        """
        Send a Finalize message to flush any buffered audio and complete the
        current turn immediately, regardless of end-of-turn detection state.
        """
        if not self.is_active:
            return
        await self._send_json({"type": "Finalize"})
        logger.debug(f"[Connection {self.connection_id}] Sent Finalize")

    async def send_close_stream(self):
        """Send a CloseStream message to gracefully terminate the stream."""
        if not self.is_active:
            return
        self.close_requested = True
        await self._send_json({"type": "CloseStream"})
        logger.debug(f"[Connection {self.connection_id}] Sent CloseStream")

    async def _send_json(self, message: dict):
        """Encode a JSON control message and write it to the input stream."""
        try:
            message_bytes = json.dumps(message).encode("utf-8")
            payload = RequestPayloadPart(bytes_=message_bytes, data_type="UTF8")
            event = RequestStreamEventPayloadPart(value=payload)
            await self.stream.input_stream.send(event)
        except Exception as e:
            logger.error(
                f"[Connection {self.connection_id}] Error sending {message.get('type')} message: {e}"
            )
            self.is_active = False

    # -------------------------------------------------------------------------
    # Output message processing
    # -------------------------------------------------------------------------

    async def _process_responses(self):
        """Receive and dispatch Flux /v2/listen server messages."""
        try:
            logger.debug(f"[Connection {self.connection_id}] Response processor started")

            while self.is_active:
                result = await self.output_stream.receive()
                if result is None:
                    logger.debug(f"[Connection {self.connection_id}] Stream closed by server")
                    break
                if result.value and result.value.bytes_:
                    self._handle_message(result.value.bytes_.decode("utf-8"))

            # Drain any remaining buffered responses after is_active goes False
            logger.debug(f"[Connection {self.connection_id}] Draining remaining responses...")
            drain_count = 0
            while drain_count < 20:
                try:
                    result = await asyncio.wait_for(self.output_stream.receive(), timeout=0.5)
                    if result is None:
                        break
                    if result.value and result.value.bytes_:
                        drain_count += 1
                        self._handle_message(result.value.bytes_.decode("utf-8"))
                except asyncio.TimeoutError:
                    break

            if drain_count:
                logger.debug(
                    f"[Connection {self.connection_id}] Drained {drain_count} buffered response(s)"
                )

        except ModelStreamError as e:
            if self.close_requested:
                logger.info(
                    f"[Connection {self.connection_id}] Stream closed after CloseStream: {e}"
                )
            else:
                message = f"[Connection {self.connection_id}] Error in response processor: {e}"
                self.is_active = False
                # Record the failure so it surfaces in summary()/--summary-jsonl —
                # a stream-level ModelStreamError (e.g. the server rejecting a
                # query param at connect) does not arrive as a JSON Error frame.
                self.errored = True
                self.error_messages.append(str(e))
                logger.error(message, exc_info=True)
                if self.fatal_error_handler:
                    self.fatal_error_handler(message)
        except Exception as e:
            self.errored = True
            self.error_messages.append(str(e))
            logger.error(
                f"[Connection {self.connection_id}] Error in response processor: {e}",
                exc_info=True,
            )

    def _handle_message(self, raw: str):
        """Parse and dispatch a single server-side Flux JSON message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[Connection {self.connection_id}] Non-JSON response: {raw!r}")
            return

        msg_type = msg.get("type")

        if msg_type == "Connected":
            self.connected = True
            request_id = msg.get("request_id", "")
            self.request_id = request_id or self.request_id
            seq = msg.get("sequence_id", 0)
            logger.info(
                f"[Connection {self.connection_id}] Connected "
                f"(request_id={request_id}, sequence_id={seq})"
            )

        elif msg_type == "TurnInfo":
            # request_id is also echoed on every TurnInfo — capture it in case
            # the Connected frame was missed.
            if not self.request_id and msg.get("request_id"):
                self.request_id = msg.get("request_id")
            self._handle_turn_info(msg)

        elif msg_type == "ConfigureSuccess":
            self.configure_success += 1
            thresholds = msg.get("thresholds", {})
            keyterms = msg.get("keyterms", [])
            language_hints = msg.get("language_hints", [])
            # Record the echoed config so the e2e driver can confirm the server
            # applied exactly what was requested — not merely that it acked.
            self.configure_acks.append({
                "thresholds": thresholds,
                "keyterms": keyterms,
                "language_hints": language_hints,
            })
            logger.info(
                f"[Connection {self.connection_id}] ConfigureSuccess "
                f"(thresholds={thresholds}, keyterms={keyterms})"
            )

        elif msg_type == "ConfigureFailure":
            self.configure_failure += 1
            code = msg.get("code", "")
            desc = msg.get("description", "")
            self.configure_failures.append({"code": code, "description": desc})
            logger.warning(
                f"[Connection {self.connection_id}] ConfigureFailure "
                f"[{code}]: {desc or 'check that eager_eot_threshold ≤ eot_threshold'}"
            )

        elif msg_type == "Error":
            code = msg.get("code", "unknown")
            desc = msg.get("description", "")
            logger.error(
                f"[Connection {self.connection_id}] Fatal server error [{code}]: {desc}"
            )
            self.errored = True
            self.error_messages.append(f"[{code}] {desc}".strip())
            self.is_active = False

        else:
            logger.debug(f"[Connection {self.connection_id}] Unhandled message type: {msg_type}")

    def _handle_turn_info(self, msg: dict):
        """
        Process a TurnInfo message and print transcript output.

        Flux TurnInfo event types:
          Update          – Periodic transcript update (~every 250ms), no state change.
          StartOfTurn     – User has started speaking; transcript may be empty.
          EagerEndOfTurn  – High likelihood the user has finished; non-empty transcript;
                            use for speculative LLM pre-processing.
          TurnResumed     – User resumed speaking after EagerEndOfTurn; discard speculative work.
          EndOfTurn       – Turn complete; transcript is final and matches last EagerEndOfTurn.
        """
        event = msg.get("event", "")
        transcript = msg.get("transcript", "")
        turn_index = msg.get("turn_index", self.turn_index)
        seq = msg.get("sequence_id", "?")
        audio_start = msg.get("audio_window_start", 0.0)
        audio_end = msg.get("audio_window_end", 0.0)
        eot_confidence = msg.get("end_of_turn_confidence")

        # Telemetry: histogram every event, and capture the multilingual
        # language fields (flux-general-multi only) for the language-hint tests.
        if event:
            self.event_counts[event] += 1
        langs = msg.get("languages")
        if langs:
            self.languages_detected.update(langs)
        langs_hinted = msg.get("languages_hinted")
        if langs_hinted:
            self.languages_hinted.update(langs_hinted)

        logger.debug(
            f"[Connection {self.connection_id}] TurnInfo event={event} "
            f"turn={turn_index} seq={seq} "
            f"audio=[{audio_start:.2f}s-{audio_end:.2f}s]"
        )

        if event == "Connected":
            # Flux sometimes sends a Connected-like event via TurnInfo on reconnects
            logger.info(f"[Connection {self.connection_id}] Stream ready")

        elif event == "StartOfTurn":
            self.transcript_parts = []
            logger.debug(f"[Connection {self.connection_id}] Turn {turn_index} started")

        elif event == "Update":
            if transcript.strip():
                print(
                    f"[Conn {self.connection_id}]   {transcript} [update]"
                )

        elif event == "EagerEndOfTurn":
            eot_str = f" ({eot_confidence:.1%})" if eot_confidence is not None else ""
            print(
                f"[Conn {self.connection_id}] ~ {transcript}{eot_str} [eager, turn {turn_index}]"
            )

        elif event == "TurnResumed":
            print(
                f"[Conn {self.connection_id}]   ... resumed [turn {turn_index}]"
            )

        elif event == "EndOfTurn":
            eot_str = f" ({eot_confidence:.1%})" if eot_confidence is not None else ""
            if transcript.strip():
                print(
                    f"[Conn {self.connection_id}] ✓ {transcript}{eot_str} [turn {turn_index}]"
                )
                # The EndOfTurn transcript is the authoritative final text for
                # the turn — accumulate it for a WER-able combined transcript.
                self.eot_transcripts.append(transcript.strip())
                now = time.monotonic()
                if self.first_eot_at is None:
                    self.first_eot_at = now
                self.last_eot_at = now
            self.turn_index = turn_index + 1

        else:
            if transcript.strip():
                print(f"[Conn {self.connection_id}] [{event}] {transcript}")

    # -------------------------------------------------------------------------
    # Structured summary
    # -------------------------------------------------------------------------

    def summary(self) -> dict:
        """Per-connection structured summary for the e2e drivers + load triage.

        ``combined_final_text`` joins every EndOfTurn transcript in order — the
        WER-able final text for the whole stream. The event histogram and
        Configure-ack counts let the e2e driver assert feature behaviour
        (eager events emitted, Configure accepted/rejected, etc.).
        """
        def _delta(a, b):
            return round(b - a, 4) if (a is not None and b is not None) else None

        return {
            "connection_id": self.connection_id,
            "request_id": self.request_id,
            "connected": self.connected,
            "errored": self.errored,
            "error_messages": list(self.error_messages),
            "chunks_sent": self.chunk_count,
            "bytes_sent": self.byte_count,
            "turns_eot": self.event_counts.get("EndOfTurn", 0),
            "turns_eager": self.event_counts.get("EagerEndOfTurn", 0),
            "turns_resumed": self.event_counts.get("TurnResumed", 0),
            "starts_of_turn": self.event_counts.get("StartOfTurn", 0),
            "updates": self.event_counts.get("Update", 0),
            "event_counts": dict(self.event_counts),
            "configure_success": self.configure_success,
            "configure_failure": self.configure_failure,
            "configure_acks": list(self.configure_acks),
            "configure_failures": list(self.configure_failures),
            "languages_detected": sorted(self.languages_detected),
            "languages_hinted": sorted(self.languages_hinted),
            "combined_final_text": " ".join(self.eot_transcripts).strip(),
            "first_eot_latency_s": _delta(self.session_start_at, self.first_eot_at),
            "session_duration_s": _delta(self.session_start_at, self.close_observed_at),
        }

    # -------------------------------------------------------------------------
    # Session teardown
    # -------------------------------------------------------------------------

    async def _close_input_stream(self):
        """Close the input (request) stream, logging but not raising on error."""
        try:
            await self.stream.input_stream.close()
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error closing input stream: {e}")

    async def end_session(self):
        """
        Gracefully close the Flux stream.

        Well-behaved close ordering (mirrors stt_wav_stress.py): send CloseStream,
        drain the response stream until the server finalizes the trailing turn and
        closes it on its own, and ONLY THEN close our input stream. Closing the
        input stream is rendered to the SageMaker container as a WebSocket Close;
        doing it immediately after CloseStream races the server's finalize and
        drops the trailing transcript. This connection's is_active is held True
        across the drain (the client-level is_active is the stop signal) so
        _process_responses keeps reading turns until the clean close.
        """
        if self.close_observed_at is None:
            self.close_observed_at = time.monotonic()

        if not self.is_active:
            return

        logger.debug(f"[Connection {self.connection_id}] Ending session")

        # 1. CloseStream tells Deepgram no more data is coming.
        await self.send_close_stream()

        # 2. Drain: wait for the response processor to read the server's final
        #    results and observe the clean close (receive() -> None), with a
        #    timeout. asyncio.wait (not wait_for) so the task isn't cancelled
        #    mid-await — awscrt raises InvalidStateError if a future is cancelled
        #    while its C-level body callback is in flight.
        if self.response_task and not self.response_task.done():
            done, _ = await asyncio.wait({self.response_task}, timeout=15.0)
            if not done:
                logger.warning(
                    f"[Connection {self.connection_id}] Timeout waiting for final responses (15s); forcing close"
                )
                self.is_active = False
                await self._close_input_stream()
                self.response_task.cancel()
                try:
                    await self.response_task
                except (asyncio.CancelledError, Exception):
                    pass
                logger.info(
                    f"[Connection {self.connection_id}] Session ended (forced) "
                    f"(sent {self.chunk_count} audio chunks, {self.turn_index} turn(s) completed)"
                )
                return
            logger.debug(f"[Connection {self.connection_id}] All responses received")

        # 3. Server already closed the output stream; close our input as cleanup.
        self.is_active = False
        await self._close_input_stream()

        logger.info(
            f"[Connection {self.connection_id}] Session ended "
            f"(sent {self.chunk_count} audio chunks, {self.turn_index} turn(s) completed)"
        )


# ---------------------------------------------------------------------------
# Shared base client
# ---------------------------------------------------------------------------

class BaseFluxClient:
    """
    Shared infrastructure for multi-connection Flux streaming clients.

    Subclasses implement the audio source (WAV file or microphone) and call
    initialize_connections() once self.sample_rate is populated.
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str = DEFAULT_REGION,
        num_connections: int = 1,
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.num_connections = num_connections
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self.client: SageMakerRuntimeHTTP2Client | None = None
        self.connections: list[DeepgramFluxConnection] = []
        self.is_active = False
        self.sample_rate: int | None = None  # must be set before initialize_connections()
        self.abort_reason: str | None = None

    def request_abort(self, message: str):
        """Stop the run after a fatal stream error and preserve the root cause."""
        if self.abort_reason is not None:
            return
        self.abort_reason = message
        self.is_active = False
        for conn in self.connections:
            conn.is_active = False

    def _initialize_client(self):
        """Resolve AWS credentials and build the SageMaker Runtime HTTP/2 client."""
        logger.debug("Initializing SageMaker client")

        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()

            if credentials is None:
                raise NoCredentialsError()

            frozen = credentials.get_frozen_credentials()
            os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                os.environ["AWS_SESSION_TOKEN"] = frozen.token

            caller_identity = session.client("sts").get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found. Configure via one of:")
            logger.error("  1. aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
            logger.error("  3. ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e

        config = Config(
            endpoint_uri=self.bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
        )
        self.client = SageMakerRuntimeHTTP2Client(config=config)
        logger.info("SageMaker client initialized")

    def verify_endpoint(self):
        """
        Verify the SageMaker endpoint exists and is InService before opening any
        bidirectional streams.

        Flux's ``invoke_endpoint_with_bidirectional_stream`` does not fail fast
        when the endpoint is missing from the target region — the HTTP/2 stream
        open simply wedges. A cheap DescribeEndpoint pre-flight turns that silent
        hang into a clear error (e.g. wrong --region).

        Raises:
            RuntimeError: If the endpoint does not exist or is not InService.
        """
        sm_client = boto3.Session(region_name=self.region).client("sagemaker")
        try:
            response = sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ValidationException":
                raise RuntimeError(
                    f"Endpoint '{self.endpoint_name}' does not exist in region '{self.region}'"
                ) from e
            raise

        status = response.get("EndpointStatus")
        if status != "InService":
            raise RuntimeError(
                f"Endpoint '{self.endpoint_name}' is not ready (status: {status})"
            )

        logger.info(f"Endpoint '{self.endpoint_name}' is InService")

    async def initialize_connections(
        self,
        model: str = DEFAULT_MODEL,
        encoding: str = "linear16",
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keyterms: list[str] | None = None,
        language_hints: list[str] | None = None,
        extra: dict[str, str] | None = None,
    ):
        """
        Start all bidirectional Flux streaming connections in parallel.

        Used by the microphone client (one shared capture broadcasts to all
        connections). The file client uses ``initialize_and_stream`` instead so
        each connection streams the file independently from the start.

        Args:
            model: Flux model variant (default: flux-general-en).
            encoding: Audio encoding (default: linear16).
            eot_threshold: EndOfTurn confidence threshold (0.5–0.9).
            eager_eot_threshold: EagerEndOfTurn threshold; must be ≤ eot_threshold.
            eot_timeout_ms: Silence timeout ms before forced EndOfTurn (500–10000).
            keyterms: Keyterms for recognition boosting.
            language_hints: Language codes (flux-general-multi only).
            extra: Arbitrary additional query parameters appended verbatim.
        """
        if self.sample_rate is None:
            raise RuntimeError("sample_rate must be set before calling initialize_connections()")

        if not self.client:
            self._initialize_client()

        logger.info(f"Initializing {self.num_connections} connection(s)...")

        for i in range(self.num_connections):
            conn = DeepgramFluxConnection(i + 1, self.client, self.endpoint_name)
            conn.fatal_error_handler = self.request_abort
            self.connections.append(conn)

        await asyncio.gather(*[
            conn.start_session(
                sample_rate=self.sample_rate,
                model=model,
                encoding=encoding,
                eot_threshold=eot_threshold,
                eager_eot_threshold=eager_eot_threshold,
                eot_timeout_ms=eot_timeout_ms,
                keyterms=keyterms or [],
                language_hints=language_hints or [],
                extra=extra,
            )
            for conn in self.connections
        ])

        self.is_active = True
        logger.info(f"All {self.num_connections} connection(s) ready")

    async def end_all_sessions(self):
        """Close all streaming sessions gracefully."""
        if not self.is_active and not self.connections:
            return

        logger.debug("Ending all sessions")
        self.is_active = False

        logger.info(f"Closing {len(self.connections)} connection(s)...")
        await asyncio.gather(*[c.end_session() for c in self.connections])
        logger.info("All sessions ended")


# ---------------------------------------------------------------------------
# File client
# ---------------------------------------------------------------------------

class FileFluxClient(BaseFluxClient):
    """
    Streams a WAV audio file in real-time to multiple simultaneous Deepgram
    Flux connections on AWS SageMaker.

    Audio is paced to the WAV file's sample rate so it arrives at the endpoint
    at the same cadence as live microphone input. The file can be looped for
    extended stress test runs. Audio chunks are sized to the Deepgram-recommended
    80ms window for optimal Flux performance.
    """

    def __init__(
        self,
        endpoint_name: str,
        wav_path: str,
        region: str = DEFAULT_REGION,
        num_connections: int = 1,
    ):
        super().__init__(endpoint_name, region, num_connections)
        self.wav_path = wav_path
        self.channels: int | None = None
        self.sample_width: int | None = None
        self.duration_seconds: float | None = None
        # Whole file loaded into memory once, then sliced into 80ms chunks per
        # connection so every connection streams an identical, independent copy
        # from the start (cf. STT) — gives clean per-connection WER + ramp.
        self.audio_bytes: bytes | None = None

    def open_wav(self) -> wave.Wave_read:
        """Open and validate the WAV file, populating audio metadata."""
        try:
            wf = wave.open(self.wav_path, "rb")
        except FileNotFoundError:
            logger.error(f"WAV file not found: {self.wav_path}")
            raise
        except wave.Error as e:
            logger.error(f"Invalid WAV file '{self.wav_path}': {e}")
            raise

        self.sample_rate = wf.getframerate()
        self.channels = wf.getnchannels()
        self.sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()
        self.duration_seconds = total_frames / self.sample_rate

        if self.sample_width != 2:
            wf.close()
            raise ValueError(
                f"WAV file must be 16-bit PCM (sample width 2 bytes). "
                f"Got {self.sample_width * 8}-bit audio. "
                "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )

        logger.info(
            f"WAV: {self.wav_path} | {self.sample_rate} Hz | "
            f"{self.channels}ch | {self.duration_seconds:.2f}s"
        )
        return wf

    def load_audio(self):
        """Load + validate the WAV once into memory for per-connection streaming."""
        wf = self.open_wav()
        try:
            self.audio_bytes = wf.readframes(wf.getnframes())
        finally:
            wf.close()

    async def initialize_and_stream(
        self,
        *,
        batch_size: int = 0,
        batch_delay: float = 0.0,
        model: str = DEFAULT_MODEL,
        encoding: str = "linear16",
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keyterms: list[str] | None = None,
        language_hints: list[str] | None = None,
        extra: dict[str, str] | None = None,
        loop: bool = False,
        plan: MidStreamPlan | None = None,
    ) -> list[asyncio.Task]:
        """
        Open every connection and stream the full WAV to each independently.

        Mirrors the STT stress client: connections open in batches of
        ``batch_size`` (gated by a semaphore so the CRT isn't swamped by
        simultaneous TCP handshakes), and each one starts streaming its own copy
        of the audio from the start as soon as it's open — so per-connection
        transcripts are complete and comparable even under ramped concurrency.

        Returns one asyncio.Task per connection (each runs open → stream).
        """
        if self.sample_rate is None or self.audio_bytes is None:
            raise RuntimeError("load_audio() must be called before initialize_and_stream()")
        if not self.client:
            self._initialize_client()

        effective_batch = batch_size if batch_size > 0 else self.num_connections
        num_batches = (self.num_connections + effective_batch - 1) // effective_batch
        logger.info(
            f"Initializing {self.num_connections} connection(s) in {num_batches} "
            f"batch(es) of up to {effective_batch} (delay {batch_delay}s between batches)..."
        )

        self.is_active = True
        streaming_tasks: list[asyncio.Task] = []
        setup_sem = asyncio.Semaphore(effective_batch)

        async def _open_and_stream(conn: DeepgramFluxConnection):
            async with setup_sem:
                try:
                    await conn.start_session(
                        sample_rate=self.sample_rate,
                        model=model,
                        encoding=encoding,
                        eot_threshold=eot_threshold,
                        eager_eot_threshold=eager_eot_threshold,
                        eot_timeout_ms=eot_timeout_ms,
                        keyterms=keyterms or [],
                        language_hints=language_hints or [],
                        extra=extra,
                    )
                except Exception as exc:
                    # Capture connect-time failures (e.g. HTTP 400 for an
                    # invalid param like language_hint on flux-general-en) so the
                    # summary still records them for the e2e driver.
                    logger.error(f"[Connection {conn.connection_id}] Failed to open: {exc}")
                    conn.errored = True
                    conn.error_messages.append(str(exc))
                    if conn.close_observed_at is None:
                        conn.close_observed_at = time.monotonic()
                    return
            try:
                await self._stream_audio_to_connection(conn, loop=loop, plan=plan)
            except Exception as exc:
                logger.error(f"[Connection {conn.connection_id}] Streaming failed: {exc}")
                conn.errored = True
                conn.error_messages.append(str(exc))

        try:
            for batch_start in range(0, self.num_connections, effective_batch):
                if not self.is_active:
                    break
                batch_end = min(batch_start + effective_batch, self.num_connections)
                for i in range(batch_start, batch_end):
                    conn = DeepgramFluxConnection(i + 1, self.client, self.endpoint_name)
                    conn.fatal_error_handler = self.request_abort
                    self.connections.append(conn)
                    streaming_tasks.append(
                        asyncio.create_task(_open_and_stream(conn), name=f"flux-conn-{i + 1}")
                    )
                logger.info(
                    f"Opening batch {batch_start // effective_batch + 1}/{num_batches}: "
                    f"connection(s) {batch_start + 1}–{batch_end}..."
                )
                if batch_end < self.num_connections and batch_delay > 0:
                    await asyncio.sleep(batch_delay)
        except Exception:
            self.is_active = False
            for task in streaming_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*streaming_tasks, return_exceptions=True)
            raise

        logger.info(f"All {self.num_connections} connection task(s) launched")
        return streaming_tasks

    async def _stream_audio_to_connection(
        self,
        conn: DeepgramFluxConnection,
        loop: bool = False,
        plan: MidStreamPlan | None = None,
    ):
        """Stream the in-memory WAV to a single connection, paced to real time.

        Honors an optional :class:`MidStreamPlan` (KeepAlive cadence, a one-shot
        mid-stream Configure, and a trailing Finalize) so the e2e driver can
        exercise Flux's in-band control messages deterministically.
        """
        assert self.audio_bytes is not None
        assert self.sample_rate is not None
        plan = plan or MidStreamPlan()
        frames_per_chunk = int(self.sample_rate * AUDIO_CHUNK_MS / 1000)
        bytes_per_frame = (self.sample_width or 2) * (self.channels or 1)
        bytes_per_chunk = frames_per_chunk * bytes_per_frame
        chunk_duration = frames_per_chunk / self.sample_rate  # seconds

        keepalive_interval = plan.keepalive_interval_s
        reconfigure_after = plan.reconfigure_after_s

        # Hold the connection open through a leading silence gap before any audio,
        # sending only tiny binary keepalive frames. If the connection dies during
        # this gap, the audio streamed afterwards won't transcribe — so a passing
        # WER on the audio below is the end-to-end proof it survived the idle.
        if plan.idle_before_audio_s:
            await self._hold_idle_with_keepalive(
                conn,
                plan.idle_before_audio_s,
                keepalive_interval or DEFAULT_IDLE_KEEPALIVE_S,
            )

        stream_start = time.monotonic()
        total_chunks = 0
        play_count = 0
        next_keepalive = keepalive_interval if keepalive_interval else None
        reconfigure_pending = reconfigure_after is not None

        while conn.is_active and self.is_active:
            play_count += 1
            for offset in range(0, len(self.audio_bytes), bytes_per_chunk):
                if not (conn.is_active and self.is_active):
                    break
                raw = self.audio_bytes[offset:offset + bytes_per_chunk]
                await conn.send_audio_chunk(raw)
                total_chunks += 1

                rel = time.monotonic() - stream_start

                # One-shot mid-stream Configure.
                if reconfigure_pending and reconfigure_after is not None and rel >= reconfigure_after:
                    await conn.send_configure(
                        eot_threshold=plan.reconfigure_eot_threshold,
                        eager_eot_threshold=plan.reconfigure_eager_eot_threshold,
                        eot_timeout_ms=plan.reconfigure_eot_timeout_ms,
                        keyterms=plan.reconfigure_keyterms,
                        language_hints=plan.reconfigure_language_hints,
                    )
                    reconfigure_pending = False

                # Periodic keepalive ping (tiny binary frame; not a JSON KeepAlive).
                if next_keepalive is not None and keepalive_interval and rel >= next_keepalive:
                    await conn.send_keepalive_ping()
                    next_keepalive += keepalive_interval

                # Pace delivery to real-time speed.
                sleep_time = (total_chunks * chunk_duration) - rel
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            if not loop:
                break

        # Flush the final turn before CloseStream when asked.
        if plan.finalize_at_end and conn.is_active:
            await conn.send_finalize()
            # Give the server a moment to emit the forced EndOfTurn before teardown.
            await asyncio.sleep(1.5)

        logger.debug(
            f"[Connection {conn.connection_id}] Finished streaming "
            f"({play_count} pass(es), {total_chunks} chunks)"
        )

    async def _hold_idle_with_keepalive(
        self,
        conn: DeepgramFluxConnection,
        duration_s: float,
        interval_s: float,
    ):
        """Hold ``conn`` open for ``duration_s`` of silence, sending a tiny binary
        keepalive frame every ``interval_s``.

        Used to verify a Flux connection survives a long idle gap. Flux relies on
        WebSocket ping for liveness (no JSON KeepAlive — that is Nova-only), but
        the SageMaker transport gives us no WS-ping primitive, so we send minimal
        binary frames instead; any frame resets the server's idle timer. Stops
        early if the connection drops (``conn.is_active`` goes False), so the
        downstream audio phase / assertions see the failure.
        """
        logger.info(
            f"[Connection {conn.connection_id}] Idle hold: {duration_s:.0f}s of "
            f"silence, keepalive ping every {interval_s:.0f}s"
        )
        deadline = time.monotonic() + duration_s
        while conn.is_active and self.is_active and time.monotonic() < deadline:
            await conn.send_keepalive_ping()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(interval_s, remaining))
        logger.info(
            f"[Connection {conn.connection_id}] Idle hold complete "
            f"(active={conn.is_active})"
        )


# ---------------------------------------------------------------------------
# Microphone client
# ---------------------------------------------------------------------------

class MicFluxClient(BaseFluxClient):
    """
    Captures live microphone audio via PyAudio and streams it to multiple
    simultaneous Deepgram Flux connections on AWS SageMaker.

    Audio is captured in 80ms chunks (matching Deepgram's recommended frame
    size) and broadcast in real-time to all active connections.
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str = DEFAULT_REGION,
        num_connections: int = 1,
        sample_rate: int = DEFAULT_MIC_SAMPLE_RATE,
        device_index: int | None = None,
    ):
        super().__init__(endpoint_name, region, num_connections)
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.audio_queue: Queue = Queue()
        self.pyaudio_instance = None
        self.audio_stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — enqueue captured audio for async broadcast."""
        try:
            import pyaudio
        except ImportError:
            pass
        if status:
            logger.warning(f"PyAudio stream status: {status}")
        self.audio_queue.put(in_data)
        try:
            import pyaudio as _pa
            return (None, _pa.paContinue)
        except ImportError:
            return (None, 0)

    async def start_microphone(self):
        """Initialize PyAudio and begin capturing microphone input."""
        try:
            import pyaudio
        except ImportError:
            print("ERROR: pyaudio is required for microphone input.")
            print("Install it with: uv add pyaudio")
            print("On macOS you may also need: brew install portaudio")
            raise

        logger.info("Initializing microphone")
        self.pyaudio_instance = pyaudio.PyAudio()

        logger.debug("Available audio input devices:")
        for i in range(self.pyaudio_instance.get_device_count()):
            info = self.pyaudio_instance.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                marker = " *" if self.device_index is not None and i == self.device_index else ""
                logger.debug(f"  [{i}] {info['name']}{marker}")

        frames_per_chunk = int(self.sample_rate * AUDIO_CHUNK_MS / 1000)

        open_kwargs: dict = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=frames_per_chunk,
            stream_callback=self._audio_callback,
        )
        if self.device_index is not None:
            open_kwargs["input_device_index"] = self.device_index

        try:
            self.audio_stream = self.pyaudio_instance.open(**open_kwargs)
            self.audio_stream.start_stream()
            logger.info("Microphone started — speak now!")
        except Exception as e:
            logger.error(
                f"Failed to open microphone: {e}. "
                "Try listing devices with --list-devices or specifying --device INDEX."
            )
            raise

    async def stream_microphone_audio(self):
        """Broadcast audio chunks from the PyAudio queue to all active connections."""
        chunk_count = 0
        try:
            while self.is_active:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get()
                    chunk_count += 1

                    active = [c for c in self.connections if c.is_active]
                    if not active:
                        logger.warning("All connections have failed")
                        self.is_active = False
                        break

                    await asyncio.gather(*[c.send_audio_chunk(audio_chunk) for c in active])
                    logger.debug(f"Broadcast chunk {chunk_count} to {len(active)} connection(s)")
                else:
                    await asyncio.sleep(0.005)
        except Exception as e:
            logger.error(f"Error broadcasting microphone audio: {e}")
            raise
        finally:
            logger.info(f"Broadcasted {chunk_count} microphone audio chunks total")

    def stop_microphone(self):
        """Stop and release PyAudio resources."""
        if self.audio_stream:
            logger.debug("Stopping microphone stream")
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pyaudio_instance:
            logger.debug("Terminating PyAudio")
            self.pyaudio_instance.terminate()
        logger.info("Microphone stopped")

    async def end_all_sessions(self):
        """Stop microphone then close all Flux sessions."""
        self.stop_microphone()
        await super().end_all_sessions()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by both the file and microphone subcommands."""
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="Number of simultaneous Flux connections (default: 1)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Flux model variant (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--eot-threshold",
        type=float,
        default=None,
        metavar="0.5-0.9",
        help="EndOfTurn confidence threshold (default: Flux server default of 0.7)",
    )
    parser.add_argument(
        "--eager-eot-threshold",
        type=float,
        default=None,
        metavar="0.3-0.9",
        help=(
            "EagerEndOfTurn confidence threshold; enables eager events when set. "
            "Must be ≤ --eot-threshold."
        ),
    )
    parser.add_argument(
        "--eot-timeout-ms",
        type=int,
        default=None,
        metavar="500-10000",
        help="Max silence (ms) before a forced EndOfTurn (default: 5000)",
    )
    parser.add_argument(
        "--keyterms",
        default="",
        metavar="TERM1,TERM2",
        help="Comma-separated list of keyterms to boost recognition accuracy",
    )
    parser.add_argument(
        "--encoding",
        default="linear16",
        help="Audio encoding query param (default: linear16). "
             "Flux accepts linear16/linear32/mulaw/alaw/opus/ogg-opus.",
    )
    parser.add_argument(
        "--language-hints",
        default="",
        metavar="en,es",
        help="Comma-separated language codes to bias recognition. Only valid with "
             "the multilingual model (flux-general-multi); sending to flux-general-en "
             "returns HTTP 400.",
    )
    parser.add_argument(
        "--profanity-filter",
        default=None,
        choices=["true", "false"],
        help="Enable profanity filtering (default: server default of false)",
    )
    parser.add_argument(
        "--extra",
        default="",
        metavar="k=v&k2=v2",
        help="Extra Flux query parameters appended verbatim (e.g. "
             "'mip_opt_out=true&tag=e2e'). Overrides typed defaults on key collision.",
    )
    parser.add_argument(
        "--summary-jsonl",
        default=None,
        metavar="PATH",
        help="Write a per-connection JSON summary (one object per line) for "
             "programmatic inspection by the e2e drivers.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop automatically after this many seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the DescribeEndpoint != InService pre-flight check. Use this "
             "when the endpoint is in Updating status but the old variant is still "
             "serving traffic (SageMaker blue/green); the stream open will fail on "
             "its own if the endpoint truly isn't reachable.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )


def _validate_common_args(args) -> str | None:
    """Return an error message string, or None if arguments are valid."""
    if args.connections < 1:
        return "--connections must be a positive integer (minimum 1)"
    if args.eot_threshold is not None and not (0.5 <= args.eot_threshold <= 0.9):
        return "--eot-threshold must be between 0.5 and 0.9"
    if args.eager_eot_threshold is not None and not (0.3 <= args.eager_eot_threshold <= 0.9):
        return "--eager-eot-threshold must be between 0.3 and 0.9"
    if (
        args.eot_threshold is not None
        and args.eager_eot_threshold is not None
        and args.eager_eot_threshold > args.eot_threshold
    ):
        return "--eager-eot-threshold must be ≤ --eot-threshold"
    return None


def _print_banner(args, extra_lines: list[str]):
    print("=" * 60)
    print("Deepgram Flux STT SageMaker Stress Test")
    print("=" * 60)
    print(f"Endpoint:      {args.endpoint_name}")
    for line in extra_lines:
        print(line)
    print(f"Connections:   {args.connections}")
    print(f"Model:         {args.model}")
    print(f"Region:        {args.region}")
    print(f"Chunk Size:    {AUDIO_CHUNK_MS}ms")
    if args.eot_threshold is not None:
        print(f"EoT Threshold: {args.eot_threshold}")
    if args.eager_eot_threshold is not None:
        print(f"Eager EoT:     {args.eager_eot_threshold}")
    if args.eot_timeout_ms is not None:
        print(f"EoT Timeout:   {args.eot_timeout_ms}ms")
    keyterms = _split_csv(args.keyterms)
    if keyterms:
        print(f"Keyterms:      {', '.join(keyterms)}")
    if getattr(args, "encoding", None) and args.encoding != "linear16":
        print(f"Encoding:      {args.encoding}")
    language_hints = _split_csv(getattr(args, "language_hints", ""))
    if language_hints:
        print(f"Lang hints:    {', '.join(language_hints)}")
    if getattr(args, "profanity_filter", None) is not None:
        print(f"Profanity:     {args.profanity_filter}")
    if getattr(args, "extra", ""):
        print(f"Extra params:  {args.extra}")
    print("=" * 60)


def _split_csv(value: str | None) -> list[str]:
    """Split a comma-separated CLI value into a clean list (empty → [])."""
    if not value:
        return []
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _parse_extra(raw: str) -> dict[str, str]:
    """Parse a `k=v&k2=v2` --extra string into a dict. Raises ValueError on bad form."""
    extra: dict[str, str] = {}
    for pair in (raw or "").split("&"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--extra entry '{pair}' must be in k=v form")
        k, v = pair.split("=", 1)
        extra[k.strip()] = v.strip()
    return extra


def _write_summary_jsonl(client: BaseFluxClient, path: str):
    """Write one JSON summary object per connection to `path` (for the e2e drivers)."""
    try:
        with open(path, "w") as f:
            for conn in client.connections:
                f.write(json.dumps(conn.summary()) + "\n")
        print(f"Per-connection summary written: {path}")
    except OSError as e:
        logger.error(f"Could not write --summary-jsonl {path}: {e}")


def _print_stream_summary(client: BaseFluxClient):
    """Print an aggregate stream summary across all connections."""
    conns = client.connections
    errored = [c for c in conns if c.errored]
    total_eot = sum(c.event_counts.get("EndOfTurn", 0) for c in conns)
    total_eager = sum(c.event_counts.get("EagerEndOfTurn", 0) for c in conns)
    cfg_ok = sum(c.configure_success for c in conns)
    cfg_fail = sum(c.configure_failure for c in conns)
    print("\n" + "=" * 60)
    print("STREAM SUMMARY")
    print("=" * 60)
    print(f"Total connections:  {len(conns)}")
    print(f"Errored:            {len(errored)}")
    print(f"EndOfTurn total:    {total_eot}")
    print(f"EagerEndOfTurn:     {total_eager}")
    if cfg_ok or cfg_fail:
        print(f"Configure ok/fail:  {cfg_ok}/{cfg_fail}")
    if errored:
        msgs: dict[str, int] = {}
        for c in errored:
            for m in c.error_messages:
                msgs[m] = msgs.get(m, 0) + 1
        print("Errors:")
        for m, n in msgs.items():
            print(f"  {'(x%d) ' % n if n > 1 else ''}{m}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _mid_stream_plan_from_args(args) -> MidStreamPlan:
    """Build a MidStreamPlan from the file-subcommand control-message flags."""
    return MidStreamPlan(
        keepalive_interval_s=getattr(args, "keepalive_interval", None),
        idle_before_audio_s=getattr(args, "idle_before_audio", None),
        reconfigure_after_s=getattr(args, "reconfigure_after", None),
        reconfigure_eot_threshold=getattr(args, "reconfigure_eot_threshold", None),
        reconfigure_eager_eot_threshold=getattr(args, "reconfigure_eager_eot_threshold", None),
        reconfigure_eot_timeout_ms=getattr(args, "reconfigure_eot_timeout_ms", None),
        reconfigure_keyterms=_split_csv(getattr(args, "reconfigure_keyterms", "")) or None,
        reconfigure_language_hints=_split_csv(getattr(args, "reconfigure_language_hints", "")) or None,
        finalize_at_end=getattr(args, "finalize_at_end", False),
    )


async def run_file(args) -> int:
    keyterms = _split_csv(args.keyterms)
    language_hints = _split_csv(args.language_hints)
    try:
        extra = _parse_extra(args.extra)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    if args.profanity_filter is not None:
        extra.setdefault("profanity_filter", args.profanity_filter)

    batch_size = getattr(args, "batch_size", 0) or 0
    batch_delay = getattr(args, "batch_delay", 0.0) or 0.0

    client = FileFluxClient(
        endpoint_name=args.endpoint_name,
        wav_path=args.file,
        region=args.region,
        num_connections=args.connections,
    )

    # Fail fast on a missing/not-ready endpoint (e.g. wrong --region) instead of
    # wedging in the bidirectional-stream open.
    if not args.skip_verify:
        try:
            client.verify_endpoint()
        except (RuntimeError, ClientError, NoCredentialsError, PartialCredentialsError) as e:
            print(f"ERROR: {e}")
            return 1
    else:
        logger.info(f"Skipping endpoint verification (--skip-verify) for '{args.endpoint_name}'")

    # Validate + load the WAV once before printing the banner.
    try:
        client.load_audio()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    limit_str = (
        f"{args.duration}s"
        if args.duration
        else ("until file ends (looping)" if args.loop else "until file ends or Ctrl+C")
    )

    banner_extras = [
        f"Input:         file ({args.file})",
        f"File Duration: {client.duration_seconds:.2f}s",
        f"Sample Rate:   {client.sample_rate} Hz",
        f"Channels:      {client.channels}",
        f"Loop:          {'yes' if args.loop else 'no'}",
        f"Limit:         {limit_str}",
    ]
    if getattr(args, "idle_before_audio", None):
        ka = getattr(args, "keepalive_interval", None) or DEFAULT_IDLE_KEEPALIVE_S
        banner_extras.append(
            f"Idle hold:     {args.idle_before_audio:.0f}s silence before audio "
            f"(keepalive ping every {ka:.0f}s)"
        )
    _print_banner(args, banner_extras)

    plan = _mid_stream_plan_from_args(args)

    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)
    exit_code = 0

    try:
        streaming_tasks = await client.initialize_and_stream(
            batch_size=batch_size,
            batch_delay=batch_delay,
            model=args.model,
            encoding=args.encoding,
            eot_threshold=args.eot_threshold,
            eager_eot_threshold=args.eager_eot_threshold,
            eot_timeout_ms=args.eot_timeout_ms,
            keyterms=keyterms,
            language_hints=language_hints,
            extra=extra,
            loop=args.loop,
            plan=plan if plan.active else None,
        )

        print("\n" + "=" * 60)
        print(f"LIVE TRANSCRIPTION - {args.connections} Flux Connection(s)")
        print("  ✓  = EndOfTurn  |  ~  = EagerEndOfTurn  |  [update] = interim")
        if args.duration:
            print(f"   (Running for {args.duration}s, or press Ctrl+C to stop early)")
        else:
            print("   (Press Ctrl+C to stop)")
        print("=" * 60 + "\n")

        all_streaming = asyncio.gather(*streaming_tasks, return_exceptions=True)
        if args.duration:
            try:
                await asyncio.wait_for(all_streaming, timeout=args.duration)
            except asyncio.TimeoutError:
                print(f"\nDuration of {args.duration}s reached, stopping...")
                client.is_active = False
                for t in streaming_tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*streaming_tasks, return_exceptions=True)
        else:
            await all_streaming

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        exit_code = 1
    finally:
        await client.end_all_sessions()
        _print_stream_summary(client)
        if args.summary_jsonl:
            _write_summary_jsonl(client, args.summary_jsonl)

    if client.abort_reason:
        print("ERROR: Aborting early after a SageMaker response stream failure.")
        print(f"Cause: {client.abort_reason}")
        print(
            "Validate your input parameters and examine CloudWatch Logs on the SageMaker "
            "Endpoint to determine the root cause."
        )
        return 1

    if exit_code:
        return exit_code

    logger.info("Stress test complete")
    return 0 if not any(c.errored for c in client.connections) else 1


async def run_microphone(args) -> int:
    # Check for pyaudio before doing anything else
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        print("ERROR: pyaudio is required for microphone input.")
        print("Install it with: uv add pyaudio")
        print("On macOS you may also need: brew install portaudio")
        return 1

    # Handle --list-devices flag
    if args.list_devices:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("Available audio input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']} ({int(info['defaultSampleRate'])} Hz)")
        pa.terminate()
        return 0

    keyterms = _split_csv(args.keyterms)
    language_hints = _split_csv(args.language_hints)
    try:
        extra = _parse_extra(args.extra)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    if args.profanity_filter is not None:
        extra.setdefault("profanity_filter", args.profanity_filter)

    client = MicFluxClient(
        endpoint_name=args.endpoint_name,
        region=args.region,
        num_connections=args.connections,
        sample_rate=args.sample_rate,
        device_index=args.device,
    )

    # Fail fast on a missing/not-ready endpoint (e.g. wrong --region) instead of
    # wedging in the bidirectional-stream open.
    if not args.skip_verify:
        try:
            client.verify_endpoint()
        except (RuntimeError, ClientError, NoCredentialsError, PartialCredentialsError) as e:
            print(f"ERROR: {e}")
            return 1
    else:
        logger.info(f"Skipping endpoint verification (--skip-verify) for '{args.endpoint_name}'")

    limit_str = f"{args.duration}s" if args.duration else "until Ctrl+C"

    _print_banner(args, [
        f"Input:         microphone",
        f"Sample Rate:   {args.sample_rate} Hz",
        f"Device:        {args.device if args.device is not None else 'default'}",
        f"Limit:         {limit_str}",
    ])

    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)
    exit_code = 0

    try:
        await client.initialize_connections(
            model=args.model,
            encoding=args.encoding,
            eot_threshold=args.eot_threshold,
            eager_eot_threshold=args.eager_eot_threshold,
            eot_timeout_ms=args.eot_timeout_ms,
            keyterms=keyterms,
            language_hints=language_hints,
            extra=extra,
        )

        await client.start_microphone()

        print("\n" + "=" * 60)
        print(f"LIVE TRANSCRIPTION - {args.connections} Flux Connection(s)")
        print("  ✓  = EndOfTurn  |  ~  = EagerEndOfTurn  |  [update] = interim")
        if args.duration:
            print(f"   (Running for {args.duration}s, or press Ctrl+C to stop early)")
        else:
            print("   (Press Ctrl+C to stop)")
        print("=" * 60 + "\n")

        stream_coro = client.stream_microphone_audio()

        if args.duration:
            try:
                await asyncio.wait_for(stream_coro, timeout=args.duration)
            except asyncio.TimeoutError:
                print(f"\nDuration of {args.duration}s reached, stopping...")
                client.is_active = False
        else:
            await stream_coro

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        exit_code = 1
    finally:
        await client.end_all_sessions()
        _print_stream_summary(client)
        if args.summary_jsonl:
            _write_summary_jsonl(client, args.summary_jsonl)

    if client.abort_reason:
        print("ERROR: Aborting early after a SageMaker response stream failure.")
        print(f"Cause: {client.abort_reason}")
        print(
            "Validate your input parameters and examine CloudWatch Logs on the SageMaker "
            "Endpoint to determine the root cause."
        )
        return 1

    if exit_code:
        return exit_code

    logger.info("Stress test complete")
    return 0 if not any(c.errored for c in client.connections) else 1


# ---------------------------------------------------------------------------
# list-endpoints subcommand handler
# ---------------------------------------------------------------------------

# Maps SageMaker endpoint status values to a short display label and colour code.
_STATUS_DISPLAY = {
    "InService":      ("InService",      "\033[32m"),   # green
    "Creating":       ("Creating",       "\033[33m"),   # yellow
    "Updating":       ("Updating",       "\033[33m"),   # yellow
    "RollingBack":    ("RollingBack",    "\033[33m"),   # yellow
    "SystemUpdating": ("SystemUpdating", "\033[33m"),   # yellow
    "Failed":         ("Failed",         "\033[31m"),   # red
    "Deleting":       ("Deleting",       "\033[31m"),   # red
    "OutOfService":   ("OutOfService",   "\033[90m"),   # grey
}
_RESET = "\033[0m"


def _colour_status(status: str, use_colour: bool) -> str:
    label, code = _STATUS_DISPLAY.get(status, (status, "\033[0m"))
    return f"{code}{label}{_RESET}" if use_colour else label


def run_list_endpoints(args) -> int:
    """List SageMaker endpoints, optionally filtered by status."""
    try:
        session = boto3.Session(region_name=args.region)
        sm = session.client("sagemaker")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("ERROR: AWS credentials not found. Configure via one of:")
        print("  1. aws configure")
        print("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
        print("  3. ~/.aws/credentials")
        print("  4. IAM role (when running on AWS infrastructure)")
        return 1

    status_filter = args.status.capitalize() if args.status else None

    try:
        list_kwargs: dict = {}
        if status_filter:
            list_kwargs["StatusEquals"] = status_filter

        endpoints: list[dict] = []
        paginator = sm.get_paginator("list_endpoints")
        for page in paginator.paginate(**list_kwargs):
            endpoints.extend(page.get("Endpoints", []))

    except sm.exceptions.ClientError as e:
        print(f"ERROR: Failed to list endpoints: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error listing endpoints: {e}")
        logger.debug("list_endpoints error", exc_info=True)
        return 1

    use_colour = sys.stdout.isatty()

    if not endpoints:
        qualifier = f" with status '{status_filter}'" if status_filter else ""
        print(f"No SageMaker endpoints found{qualifier} in {args.region}.")
        return 0

    # Column widths
    name_w = max(len(ep["EndpointName"]) for ep in endpoints)
    name_w = max(name_w, len("ENDPOINT NAME"))
    status_w = max(len(ep["EndpointStatus"]) for ep in endpoints)
    status_w = max(status_w, len("STATUS"))

    header = (
        f"{'ENDPOINT NAME':<{name_w}}  "
        f"{'STATUS':<{status_w}}  "
        f"{'CREATED':<20}  "
        f"LAST MODIFIED"
    )
    sep = "-" * len(header)

    print(f"\nSageMaker Endpoints  [{args.region}]")
    print(sep)
    print(header)
    print(sep)

    for ep in sorted(endpoints, key=lambda e: e["EndpointName"]):
        name = ep["EndpointName"]
        status = ep["EndpointStatus"]
        created = ep["CreationTime"].strftime("%Y-%m-%d %H:%M:%S")
        modified = ep["LastModifiedTime"].strftime("%Y-%m-%d %H:%M:%S")
        status_str = _colour_status(status, use_colour)
        # Pad without ANSI codes affecting alignment
        pad = status_w - len(status)
        print(f"{name:<{name_w}}  {status_str}{' ' * pad}  {created:<20}  {modified}")

    print(sep)
    qualifier = f" (filtered: {status_filter})" if status_filter else ""
    print(f"{len(endpoints)} endpoint(s){qualifier}\n")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream audio to multiple simultaneous Deepgram Flux connections "
            "on Amazon SageMaker for stress testing."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # -- list-endpoints subcommand --------------------------------------------
    list_parser = subparsers.add_parser(
        "list-endpoints",
        help="List available SageMaker endpoints",
        description="List SageMaker endpoints in the target region with their status.",
    )
    list_parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    list_parser.add_argument(
        "--status",
        default=None,
        metavar="STATUS",
        choices=[s.lower() for s in _STATUS_DISPLAY],
        help=(
            "Filter by endpoint status. "
            "Choices: " + ", ".join(s.lower() for s in _STATUS_DISPLAY)
        ),
    )
    list_parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: WARNING)",
    )

    # -- file subcommand ------------------------------------------------------
    file_parser = subparsers.add_parser(
        "file",
        help="Stream a WAV audio file at real-time pace",
        description=(
            "Stream a 16-bit PCM WAV file in real-time to multiple simultaneous "
            "Flux connections for repeatable load testing."
        ),
    )
    _add_common_args(file_parser)
    file_parser.add_argument(
        "--file",
        required=True,
        metavar="WAV_FILE",
        help="Path to a 16-bit PCM WAV file to stream",
    )
    file_parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the WAV file continuously until --duration is reached or Ctrl+C",
    )
    file_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help="Open connections in batches of N (ramp-up). Default: all at once.",
    )
    file_parser.add_argument(
        "--batch-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Seconds to wait between opening connection batches (default: 0)",
    )
    # --- Mid-stream control-message hooks (Flux-distinctive protocol features) ---
    file_parser.add_argument(
        "--keepalive-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Send a tiny binary keepalive frame every N seconds (during the "
             "idle hold and while streaming). NOT a JSON KeepAlive — that is "
             "Nova-only; Flux keeps alive via WS ping / continued frames.",
    )
    file_parser.add_argument(
        "--idle-before-audio",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Before streaming audio, hold the connection open for N seconds of "
             "silence, sending only tiny binary keepalive frames every "
             "--keepalive-interval (default %ss). Verifies the connection "
             "survives a long idle gap: the audio that follows must still "
             "transcribe." % int(DEFAULT_IDLE_KEEPALIVE_S),
    )
    file_parser.add_argument(
        "--finalize-at-end",
        action="store_true",
        help="Send a Finalize message after the last audio chunk to flush the final turn",
    )
    file_parser.add_argument(
        "--reconfigure-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Send one mid-stream Configure after N seconds of audio, applying any "
             "--reconfigure-* values below (exercises ConfigureSuccess/ConfigureFailure)",
    )
    file_parser.add_argument(
        "--reconfigure-eot-threshold",
        type=float,
        default=None,
        help="eot_threshold to apply in the mid-stream Configure",
    )
    file_parser.add_argument(
        "--reconfigure-eager-eot-threshold",
        type=float,
        default=None,
        help="eager_eot_threshold to apply in the mid-stream Configure "
             "(set > --reconfigure-eot-threshold to force a ConfigureFailure)",
    )
    file_parser.add_argument(
        "--reconfigure-eot-timeout-ms",
        type=int,
        default=None,
        help="eot_timeout_ms to apply in the mid-stream Configure",
    )
    file_parser.add_argument(
        "--reconfigure-keyterms",
        default="",
        metavar="TERM1,TERM2",
        help="Replacement keyterms list to apply in the mid-stream Configure",
    )
    file_parser.add_argument(
        "--reconfigure-language-hints",
        default="",
        metavar="en,es",
        help="Replacement language_hints to apply in the mid-stream Configure "
             "(flux-general-multi only)",
    )

    # -- microphone subcommand ------------------------------------------------
    mic_parser = subparsers.add_parser(
        "microphone",
        help="Capture live microphone input and stream it in real-time",
        description=(
            "Capture live audio from a microphone and stream it to multiple "
            "simultaneous Flux connections for real-time stress testing."
        ),
    )
    _add_common_args(mic_parser)
    mic_parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_MIC_SAMPLE_RATE,
        help=f"Microphone sample rate in Hz (default: {DEFAULT_MIC_SAMPLE_RATE})",
    )
    mic_parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="INDEX",
        help="PyAudio input device index (default: system default). Use --list-devices to enumerate.",
    )
    mic_parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 0

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    if args.subcommand == "list-endpoints":
        return run_list_endpoints(args)

    err = _validate_common_args(args)
    if err:
        print(f"ERROR: {err}")
        return 1

    if args.subcommand == "file":
        return await run_file(args)
    else:
        return await run_microphone(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

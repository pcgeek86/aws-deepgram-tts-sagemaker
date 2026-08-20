#!/usr/bin/env python3
"""
Deepgram SageMaker WAV File STT Client - Multiple Connections

Streams or batch-transcribes a WAV audio file using a Deepgram model
deployed on Amazon SageMaker.

Sub-commands
------------
stream  – Bidirectional streaming via the /v1/listen WebSocket-style
          HTTP/2 endpoint.  Audio is paced to the WAV file's sample rate
          to simulate live input.  Supports multiple simultaneous connections
          for load testing.

batch   – Synchronous HTTP via the SageMaker InvokeEndpoint API and the
          Deepgram pre-recorded /v1/listen REST endpoint.  The entire WAV
          file is posted in a single request.  Supports configurable
          concurrency and a total request count for throughput testing.
"""

import asyncio
import argparse
import concurrent.futures
import json
import logging
import os
import resource
import signal
import statistics
import sys
import threading
import time
import wave
from collections import Counter, deque
from urllib.parse import quote
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestStreamEventPayloadPart,
    RequestPayloadPart
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
from smithy_http.aio.crt import AWSCRTHTTPClient, _AWSCRTEventLoop
import awscrt.io as crt_io
from botocore.config import Config as BotoConfig

# Configuration constants
DEFAULT_REGION = "us-east-2"
CHUNK_SIZE = 8192  # Bytes per audio chunk (stream mode)
KEEP_ALIVE_INTERVAL_SECONDS = 5.0

# SageMaker InvokeEndpoint body limit (bytes)
INVOKE_ENDPOINT_MAX_BYTES = 26_214_400  # 25 MB

# Deepgram supported redaction entity types
# https://developers.deepgram.com/docs/supported-entity-types
REDACT_ENTITIES = [
    # Broad group redactions
    "pii", "phi", "pci",
    # PII entities
    "account_number", "age", "bank_account", "cardinal", "credit_card",
    "credit_card_expiration", "cvv", "date", "date_interval", "dob",
    "driver_license", "email_address", "event", "filename", "gender_sexuality",
    "healthcare_number", "ip_address", "location", "location_address",
    "location_city", "location_coordinate", "location_country", "location_state",
    "location_zip", "money", "name", "name_given", "name_family",
    "name_medical_professional", "numerical_pii", "occupation", "ordinal",
    "origin", "passport_number", "password", "percent", "phone_number",
    "physical_attribute", "ssn", "time", "url", "username", "vehicle_id",
    # PHI entities
    "condition", "drug", "injury", "blood_type", "medical_process", "statistics",
    # Other entities
    "language", "marital_status", "organization", "political_affiliation",
    "religion", "routing_number", "zodiac_sign",
]


def aws_config(fips: bool = False) -> BotoConfig | None:
    """botocore Config for this run — FIPS endpoints when `fips`, else default.

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
    return BotoConfig(use_fips_endpoint=True) if fips else None


def resolve_bidi_endpoint(region: str, fips: bool = False) -> str:
    """SageMaker bidi-streaming URL for `region` (port 8443, per AWS's
    bidirectional-stream requirement).

    Derived from botocore's own sagemaker-runtime endpoint resolution rather
    than a hardcoded hostname, so `--fips` moves the stream onto
    runtime-fips.sagemaker.<region>.amazonaws.com. The smithy HTTP/2 client
    takes a literal `endpoint_uri` and performs no resolution of its own, so
    without this the stream would silently stay on the NON-FIPS host while
    every boto3 call in the same run honored the FIPS setting — a FIPS run that
    quietly is not one.
    """
    url = (boto3.Session(region_name=region)
           .client("sagemaker-runtime", config=aws_config(fips))
           .meta.endpoint_url)
    return f"{url}:8443"


logger = logging.getLogger(__name__)


class _AtomicStderrHandler(logging.Handler):
    """Logging handler that writes each record as a single atomic os.write().

    The default StreamHandler does ``stream.write(msg); stream.write('\\n')``
    which is two syscalls.  If the terminal is using VT100 scroll regions,
    a stdout write for the status bar can land between those two stderr
    writes, visually corrupting both.  This handler formats the record
    (including the trailing newline) into one buffer and writes it in a
    single ``os.write(2, …)`` call so the kernel delivers it atomically.
    """

    def __init__(self, fmt: logging.Formatter | None = None):
        super().__init__()
        if fmt:
            self.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            os.write(2, msg.encode("utf-8", errors="replace"))
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Streaming classes
# ---------------------------------------------------------------------------

class DeepgramSageMakerConnection:
    """
    Represents a single bidirectional streaming connection to SageMaker.

    Each connection handles its own stream and response processing.
    """

    def __init__(self, connection_id, client, endpoint_name, write_fn=None,
                 use_close_stream=True, raw=False, ring_size: int = 50,
                 control_data_type: str = "UTF8"):
        self.connection_id = connection_id
        self.client = client
        self.endpoint_name = endpoint_name
        self._write_fn = write_fn or print
        self.use_close_stream = use_close_stream
        self.control_data_type = control_data_type
        self.raw = raw
        self.stream = None
        self.output_stream = None
        self.is_active = False
        self.response_task = None
        self.keep_alive_task = None
        self._send_lock = asyncio.Lock()
        self.chunk_count = 0
        self.byte_count = 0
        self.transcript_count = 0
        self.final_transcript_parts: list[str] = []
        self.interim_count = 0
        self.errored = False
        self.error_messages: list[str] = []
        # Red-team / metering-correlation diagnostics
        self.dg_request_id: str | None = None
        self.metadata_msg: dict | None = None
        self.message_type_counts: Counter[str] = Counter()
        self.unknown_message_count = 0
        self.ring_buffer: deque[tuple[float, str]] = deque(maxlen=ring_size)
        self.session_start_at: float | None = None
        self.first_final_at: float | None = None
        self.last_final_at: float | None = None
        self.last_recv_at: float | None = None
        self.close_observed_at: float | None = None
        self.close_reason: str | None = None
        # Guards against closing the input stream / sending CloseStream more
        # than once across the EOF, error, and end_session paths.
        self._input_closed = False
        self._close_stream_sent = False

    async def start_session(self, sample_rate, model="nova-3", language="en", **kwargs):
        """
        Start a bidirectional streaming session with Deepgram on SageMaker.

        Args:
            sample_rate: Sample rate of the audio being sent.
            model: Deepgram model (default: nova-3).
            language: Language code (default: en).
            **kwargs: Additional Deepgram query parameters.
        """
        keywords_list = kwargs.pop('keywords', [])
        keyterms_list = kwargs.pop('keyterms', [])
        redact_list = kwargs.pop('redact_entities', [])

        query_params = {
            "model": model,
            "language": language,
            "encoding": "linear16",
            "sample_rate": sample_rate,
        }
        query_params.update(kwargs)

        # URL-encode values. Critical for params whose values carry reserved
        # query-string characters (`:`, `&`, etc.) — e.g. `replace=src:dst`,
        # which SageMaker's bidi-stream `model_query_string` would otherwise
        # mangle to an empty `replace: {}` on the API side.
        query_string = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in query_params.items())

        if keywords_list:
            keywords_params = "&".join(f"keywords={quote(kw)}" for kw in keywords_list)
            query_string = f"{query_string}&{keywords_params}"

        if keyterms_list:
            keyterms_params = "&".join(f"keyterm={quote(kt)}" for kt in keyterms_list)
            query_string = f"{query_string}&{keyterms_params}"

        if redact_list:
            redact_params = "&".join(f"redact={quote(r)}" for r in redact_list)
            query_string = f"{query_string}&{redact_params}"

        logger.debug(f"[Connection {self.connection_id}] Starting session: {query_string}")

        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path="v1/listen",
            model_query_string=query_string
        )

        try:
            self.stream = await asyncio.wait_for(
                self.client.invoke_endpoint_with_bidirectional_stream(stream_input),
                timeout=30,
            )
            self.is_active = True
            self.session_start_at = time.monotonic()

            output = await asyncio.wait_for(self.stream.await_output(), timeout=30)
            self.output_stream = output[1]

            self.response_task = asyncio.create_task(self._process_responses())
            self.keep_alive_task = asyncio.create_task(self._send_keep_alives())
            logger.info(f"[Connection {self.connection_id}] Session started")
        except Exception:
            self.is_active = False
            if self.stream:
                try:
                    await self.stream.input_stream.close()
                    self._input_closed = True
                except Exception:
                    pass
            raise

    async def _send_payload_bytes(self, payload_bytes: bytes, data_type: str, *,
                                  require_active=True):
        """Send one SageMaker event-stream payload, serialized per connection.

        ``data_type`` picks the WebSocket opcode SageMaker delivers to the
        container (AWS bidi container contract §3.1): ``UTF8`` → Text frame,
        ``BINARY`` (or an absent DataType) → Binary frame. It is required
        rather than defaulted because getting it wrong is silent: a control
        message sent as BINARY is consumed as audio unless the server reframes
        it. Audio is ``BINARY``; control messages are ``UTF8``.
        """
        if require_active and not self.is_active:
            return False
        if self._input_closed:
            return False
        payload = RequestPayloadPart(bytes_=payload_bytes, data_type=data_type)
        event = RequestStreamEventPayloadPart(value=payload)
        async with self._send_lock:
            await self.stream.input_stream.send(event)
        return True

    async def _send_control_message(self, message_type: str, *, require_active=True):
        """Send a Deepgram-compatible JSON control message.

        Uses ``--control-data-type`` (default ``UTF8``, the correct and
        documented setting). ``BINARY`` reaches the server only via the
        container's deprecated binary→text reframing, and is selectable to
        exercise it.
        """
        message = json.dumps({"type": message_type}).encode("utf-8")
        return await self._send_payload_bytes(
            message, self.control_data_type, require_active=require_active
        )

    async def _send_keep_alives(self):
        """Send Deepgram KeepAlive control messages while the stream is open."""
        try:
            while self.is_active:
                await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)
                if not self.is_active:
                    break
                sent = await self._send_control_message("KeepAlive")
                if sent:
                    logger.debug(
                        f"[Connection {self.connection_id}] Sent KeepAlive"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self.is_active:
                return
            msg = str(e)
            logger.error(
                f"[Connection {self.connection_id}] Error sending KeepAlive: {msg}"
            )
            self.errored = True
            self.error_messages.append(msg)
            self.is_active = False

    async def send_audio_chunk(self, audio_bytes):
        """Send a chunk of audio data to the stream."""
        if not self.is_active:
            return
        try:
            await self._send_payload_bytes(audio_bytes, "BINARY")
            self.chunk_count += 1
            self.byte_count += len(audio_bytes)
            logger.debug(
                f"[Connection {self.connection_id}] Sent chunk {self.chunk_count} "
                f"({len(audio_bytes)} bytes)"
            )
        except Exception as e:
            msg = str(e)
            logger.error(f"[Connection {self.connection_id}] Error sending audio chunk: {msg}")
            self.errored = True
            self.error_messages.append(msg)
            self.is_active = False
            # Close the input stream so the server closes the output stream,
            # allowing _process_responses to exit via receive() returning None.
            # Do not cancel response_task directly — awscrt futures raise
            # InvalidStateError when cancelled while a body callback is in flight.
            # No CloseStream here: the stream has already errored, so a bare
            # close is the only meaningful action.
            await self._close_input_stream()

    async def _send_close_stream(self, timeout: float = 5.0):
        """Send a single Deepgram ``{"type":"CloseStream"}`` frame to signal
        end-of-audio and ask the server to finalize
        (https://developers.deepgram.com/docs/close-stream).

        Deliberately does NOT close the input (request) stream. Over the
        SageMaker bidirectional transport, closing the input stream is rendered
        to the container as a WebSocket Close, and doing that immediately after
        CloseStream races the server's CloseStream-triggered finalize — the
        server flushes the trailing segment and then closes the response stream,
        but a near-simultaneous Close makes the API break before that final result
        is delivered, dropping the tail. The well-behaved sequence
        is: send CloseStream, drain the response stream until the server closes
        it, THEN close the input stream as cleanup (see ``end_session``).

        Sent with ``DataType`` from ``--control-data-type`` (default ``UTF8``).
        That field picks the WebSocket opcode SageMaker delivers to the container
        (AWS bidi container contract §3.1): ``UTF8`` → Text frame, absent or
        ``BINARY`` → Binary frame. Deepgram control messages are only parsed in
        from Text frames, so ``UTF8`` is the correct and documented setting —
        ``BINARY`` works only because the container sniffs small JSON-object
        binary frames and reframes them, and is kept selectable to exercise
        that path.

        Idempotent and a no-op when this connection has CloseStream disabled
        (``--no-use-close-stream``) or the input stream is already closed.
        """
        if not self.use_close_stream or self._close_stream_sent or self._input_closed:
            return
        self._close_stream_sent = True
        try:
            await asyncio.wait_for(
                self._send_control_message("CloseStream", require_active=False),
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(
                f"[Connection {self.connection_id}] Could not send CloseStream: {e}"
            )

    async def _close_input_stream(self, timeout: float = 5.0):
        """Close the input (request) stream once.

        On the CloseStream path, call this only AFTER draining the response
        stream (see ``_send_close_stream`` for why). On the error / bare-close
        path (``--no-use-close-stream``), the input-stream close is itself the
        only end-of-audio signal, so it is called directly.

        Idempotent: the EOF, error, and end_session paths may all reach here,
        but the close is issued at most once.
        """
        if self._input_closed:
            return
        self._input_closed = True
        try:
            await asyncio.wait_for(self.stream.input_stream.close(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[Connection {self.connection_id}] Timeout closing input stream")
        except Exception as e:
            logger.error(f"[Connection {self.connection_id}] Error closing input stream: {e}")

    async def _process_responses(self):
        """Process streaming responses from Deepgram."""
        try:
            logger.debug(f"[Connection {self.connection_id}] Response processor started")

            while self.is_active:
                result = await self.output_stream.receive()
                if result is None:
                    self.close_observed_at = time.monotonic()
                    self.close_reason = "stream_end_clean"
                    logger.info(
                        f"[Connection {self.connection_id}] Stream closed cleanly "
                        f"(dg_request_id={self.dg_request_id}, "
                        f"finals={self.transcript_count}, interims={self.interim_count})"
                    )
                    break
                if result.value and result.value.bytes_:
                    self.last_recv_at = time.monotonic()
                    self._handle_response(result.value.bytes_.decode('utf-8'))


        except Exception as e:
            msg = _unwrap_streaming_error(e)
            self.close_observed_at = time.monotonic()
            self.close_reason = f"exception: {type(e).__name__}: {msg}"
            if "Input stream broken" in msg and (self._close_stream_sent or self._input_closed):
                logger.info(
                    f"[Connection {self.connection_id}] Stream closed after client shutdown: {msg}"
                )
                return
            logger.error(
                f"[Connection {self.connection_id}] Error in response processor "
                f"(dg_request_id={self.dg_request_id}, "
                f"types={dict(self.message_type_counts)}, "
                f"last_frames={list(self.ring_buffer)[-5:]!r}): {msg}",
                exc_info=True,
            )
            self.errored = True
            self.is_active = False
            self.error_messages.append(msg)

    def _handle_response(self, raw: str):
        """Parse and print a streaming transcript response."""
        self.ring_buffer.append((time.monotonic(), raw))
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            self.unknown_message_count += 1
            logger.warning(f"[Connection {self.connection_id}] Non-JSON response: {raw!r}")
            return

        msg_type = parsed.get("type") or ("Results" if "channel" in parsed else "Unknown")
        self.message_type_counts[msg_type] += 1

        # Deepgram embeds request_id + model_info inside every Results message's
        # nested "metadata" object (NOT a separate Metadata-typed message). Also
        # accept a top-level request_id for Metadata frames in case future
        # protocol versions split it out.
        if self.dg_request_id is None:
            rid = (parsed.get("metadata") or {}).get("request_id") or parsed.get("request_id")
            if rid:
                self.dg_request_id = rid
                self.metadata_msg = parsed.get("metadata") or parsed
                logger.info(
                    f"[Connection {self.connection_id}] dg_request_id={rid}"
                )

        if self.raw:
            self._write_fn(f"[Conn {self.connection_id}] RAW {raw}")

        if 'channel' not in parsed:
            # Surface endpoint error messages (e.g. {"error": "...", "message": "..."})
            err_msg = parsed.get('error') or parsed.get('message')
            if err_msg and msg_type not in ("Metadata", "UtteranceEnd", "SpeechStarted"):
                self.errored = True
                self.error_messages.append(str(err_msg))
                logger.error(
                    f"[Connection {self.connection_id}] Endpoint error "
                    f"(dg_request_id={self.dg_request_id}, type={msg_type}): {err_msg}"
                )
            return

        alternatives = parsed.get('channel', {}).get('alternatives', [])
        if not alternatives or not alternatives[0].get('transcript'):
            return

        transcript = alternatives[0]['transcript']
        if not transcript.strip():
            return

        confidence = alternatives[0].get('confidence', 0)
        is_final = parsed.get('is_final', False)
        speech_final = parsed.get('speech_final', False)

        if is_final:
            now = time.monotonic()
            if self.first_final_at is None:
                self.first_final_at = now
            self.last_final_at = now
            self.transcript_count += 1
            self.final_transcript_parts.append(transcript)
            self._write_fn(f"[Conn {self.connection_id}] ✓ {transcript} ({confidence:.1%})")
        else:
            self.interim_count += 1
            self._write_fn(f"[Conn {self.connection_id}]   {transcript} [interim]")

    def summary(self) -> dict:
        """Per-connection structured summary for red-team + metering reconciliation."""
        def _delta(a, b):
            return round(b - a, 4) if (a is not None and b is not None) else None
        return {
            "connection_id": self.connection_id,
            "dg_request_id": self.dg_request_id,
            "errored": self.errored,
            "error_messages": list(self.error_messages),
            "close_reason": self.close_reason,
            "chunks_sent": self.chunk_count,
            "bytes_sent": self.byte_count,
            "transcripts_final": self.transcript_count,
            "transcripts_interim": self.interim_count,
            "combined_final_text": " ".join(self.final_transcript_parts).strip(),
            "message_type_counts": dict(self.message_type_counts),
            "unknown_message_count": self.unknown_message_count,
            "first_final_latency_s": _delta(self.session_start_at, self.first_final_at),
            "session_duration_s": _delta(self.session_start_at, self.close_observed_at),
            "ring_buffer_tail": [
                {"t_rel": round(t - (self.session_start_at or t), 4), "frame": frame}
                for t, frame in list(self.ring_buffer)[-10:]
            ],
        }

    async def _drain_responses(self, timeout: float = 10.0):
        """Wait for the response processor to read the server's final results
        and observe the server's clean close (receive() -> None). On timeout,
        force the input stream closed to unblock the server, then cancel.

        Uses asyncio.wait (not wait_for) so the task isn't cancelled mid-await —
        awscrt raises InvalidStateError if a future is cancelled while its
        C-level body callback is in flight.
        """
        if not (self.response_task and not self.response_task.done()):
            return
        done, _ = await asyncio.wait({self.response_task}, timeout=timeout)
        if not done:
            logger.warning(
                f"[Connection {self.connection_id}] Timeout waiting for final responses; "
                "forcing task shutdown"
            )
            self.is_active = False
            await self._close_input_stream()
            self.response_task.cancel()
            try:
                await self.response_task
            except (asyncio.CancelledError, Exception):
                pass

    async def end_session(self):
        """Close the streaming session.

        Well-behaved close ordering: send CloseStream, drain the
        response stream until the server finalizes the trailing segment and
        closes the stream on its own, and ONLY THEN close our input stream.
        Closing the input stream is rendered to the SageMaker container as a
        WebSocket Close; doing it immediately after CloseStream races the
        server's finalize and drops the trailing transcript. `is_active` is held
        True across the drain so `_process_responses` keeps reading finals until
        the server's clean close — streaming is already complete by the time
        end_session runs, so no further audio is sent meanwhile.
        """
        logger.debug(f"[Connection {self.connection_id}] Ending session")

        if self.keep_alive_task and not self.keep_alive_task.done():
            self.keep_alive_task.cancel()
            try:
                await self.keep_alive_task
            except asyncio.CancelledError:
                pass

        if self.use_close_stream:
            # Idempotent: the EOF path normally already sent CloseStream; this
            # also covers connections that never reached EOF (--duration cutoff).
            await self._send_close_stream()
            await self._drain_responses(timeout=10.0)
            self.is_active = False
            await self._close_input_stream()
        else:
            # Bare-close mode (--no-use-close-stream): the input-stream close is
            # the only end-of-audio signal, so close first, then drain.
            self.is_active = False
            await self._close_input_stream()
            await self._drain_responses(timeout=10.0)

        logger.info(
            f"[Connection {self.connection_id}] Session ended "
            f"(sent {self.chunk_count} chunks)"
        )


class MultiConnectionWAVClient:
    """
    Streams a WAV file in real-time to multiple simultaneous Deepgram
    connections on AWS SageMaker.

    Audio chunks are paced according to the WAV file's sample rate so that
    the stream arrives at the endpoint at the same rate as live audio would.
    The file loops if --loop is specified, otherwise it plays once.
    """

    def __init__(self, endpoint_name, wav_path, region=DEFAULT_REGION, num_connections=1,
                 use_close_stream=True, raw=False, ring_size: int = 50,
                 control_data_type: str = "UTF8", fips: bool = False):
        self.endpoint_name = endpoint_name
        self.wav_path = wav_path
        self.region = region
        self.num_connections = num_connections
        self.use_close_stream = use_close_stream
        self.control_data_type = control_data_type
        self.raw = raw
        self.ring_size = ring_size
        self.fips = fips
        self.bidi_endpoint = resolve_bidi_endpoint(region, fips)
        self._clients: list[SageMakerRuntimeHTTP2Client] = []
        self._credentials_ready = False
        self.connections = []
        self.is_active = False
        self._init_complete = False

        self.sample_rate = None
        self.channels = None
        self.sample_width = None
        self.duration_seconds = None

    def _open_wav(self):
        """Read WAV header and validate, returning a wave.Wave_read object."""
        try:
            wf = wave.open(self.wav_path, 'rb')
        except FileNotFoundError:
            logger.error(f"WAV file not found: {self.wav_path}")
            raise
        except wave.Error as e:
            logger.error(f"Invalid WAV file '{self.wav_path}': {e}")
            raise

        self.sample_rate = wf.getframerate()
        self.channels = wf.getnchannels()
        self.sample_width = wf.getsampwidth()
        self.duration_seconds = wf.getnframes() / self.sample_rate

        if self.sample_width != 2:
            wf.close()
            raise ValueError(
                f"WAV file must be 16-bit PCM (sample width 2 bytes). "
                f"Got {self.sample_width * 8}-bit audio. "
                "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
            )

        logger.log(
            logging.INFO if not self._init_complete else logging.DEBUG,
            f"WAV: {self.wav_path} | {self.sample_rate} Hz | "
            f"{self.channels}ch | {self.duration_seconds:.2f}s"
        )
        return wf

    def _ensure_credentials(self):
        """Resolve AWS credentials once and export them to the environment."""
        if self._credentials_ready:
            return
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()

            frozen = credentials.get_frozen_credentials()
            os.environ['AWS_ACCESS_KEY_ID'] = frozen.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = frozen.secret_key
            if frozen.token:
                os.environ['AWS_SESSION_TOKEN'] = frozen.token

            caller_identity = session.client('sts').get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found. Configure via one of:")
            logger.error("  1. aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
            logger.error("  3. ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e

        self._credentials_ready = True

    # Each bidirectional stream needs its own TCP connection so that
    # audio data flows immediately without being queued behind other
    # streams' data in the CRT's HTTP/2 multiplexer.  Sharing even
    # 2 streams per connection causes the CRT to starve one of them,
    # triggering Deepgram's 12-second idle timeout (NET-0001).
    _MAX_STREAMS_PER_CONNECTION = 1

    def _create_one_client(self, eventloop: _AWSCRTEventLoop) -> SageMakerRuntimeHTTP2Client:
        """Create a single SageMaker HTTP/2 client with its own CRT transport."""
        transport = AWSCRTHTTPClient(eventloop=eventloop)

        config = Config(
            endpoint_uri=self.bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
            transport=transport,
        )
        return SageMakerRuntimeHTTP2Client(config=config)

    def _initialize_clients(self):
        """Prepare the shared CRT event loop and credential environment.

        Clients are created lazily (one per connection) via
        ``_get_client()`` so that hundreds of CRT transports aren't
        allocated upfront.
        """
        self._ensure_credentials()

        # Single shared event loop (one I/O thread per CPU core).
        self._eventloop = _AWSCRTEventLoop.__new__(_AWSCRTEventLoop)
        elg = crt_io.EventLoopGroup(0)
        host_resolver = crt_io.DefaultHostResolver(elg)
        self._eventloop.bootstrap = crt_io.ClientBootstrap(elg, host_resolver)

        logger.info(
            f"CRT event loop ready "
            f"(clients created lazily, 1 per connection)"
        )

    def _get_client(self, connection_index: int) -> SageMakerRuntimeHTTP2Client:
        """Return a client for the given connection, creating it lazily."""
        # With _MAX_STREAMS_PER_CONNECTION = 1, each connection gets
        # its own client.  Reuse clients when the ratio allows sharing.
        client_index = connection_index // self._MAX_STREAMS_PER_CONNECTION
        while len(self._clients) <= client_index:
            self._clients.append(self._create_one_client(self._eventloop))
        return self._clients[client_index]

    async def initialize_connections(
        self,
        batch_size: int = 0,
        batch_delay: float = 0.0,
        model="nova-3",
        language="en",
        loop: bool = False,
        **kwargs,
    ) -> list[asyncio.Task]:
        """
        Open streaming connections in batches and immediately start streaming
        the WAV file to each connection as soon as its batch is ready.

        Args:
            batch_size: Connections to open per batch (0 = all at once).
            batch_delay: Seconds to wait between batches.
            model: Deepgram model.
            language: Language code.
            loop: Whether each connection should loop the WAV file.
            **kwargs: Additional Deepgram query parameters.

        Returns:
            List of asyncio Tasks, one per connection, each streaming the
            full WAV file independently.
        """
        if not self._clients:
            self._initialize_clients()

        effective_batch = batch_size if batch_size > 0 else self.num_connections
        num_batches = (self.num_connections + effective_batch - 1) // effective_batch
        logger.info(
            f"Initializing {self.num_connections} connection(s) "
            f"in {num_batches} batch(es) of up to {effective_batch} "
            f"(delay {batch_delay}s between batches)..."
        )

        streaming_tasks: list[asyncio.Task] = []
        self.is_active = True
        max_retries = 5
        # Limit concurrent connection setups so the CRT isn't overwhelmed
        # with hundreds of simultaneous TCP handshakes, while still
        # allowing each connection to stream immediately after opening.
        setup_sem = asyncio.Semaphore(effective_batch)

        async def _open_and_stream(conn: DeepgramSageMakerConnection):
            """Open one connection and start streaming immediately.

            The semaphore gates connection setup so only ``batch_size``
            sessions open concurrently, but once a session is open it
            releases the semaphore and starts streaming right away —
            no waiting for sibling connections.
            """
            async with setup_sem:
                for attempt in range(max_retries):
                    try:
                        await conn.start_session(
                            self.sample_rate, model=model,
                            language=language, **kwargs,
                        )
                        break
                    except Exception as exc:
                        is_retryable = (
                            isinstance(exc, asyncio.TimeoutError)
                            or "ThrottlingException" in str(exc)
                        )
                        if is_retryable and attempt < max_retries - 1:
                            wait = 2 ** attempt
                            logger.warning(
                                f"[Connection {conn.connection_id}] "
                                f"Failed ({type(exc).__name__}), "
                                f"retrying in {wait}s "
                                f"(attempt {attempt + 1}/{max_retries})..."
                            )
                            await asyncio.sleep(wait)
                        else:
                            logger.error(
                                f"[Connection {conn.connection_id}] "
                                f"Failed to open: {exc}"
                            )
                            conn.errored = True
                            conn.error_messages.append(str(exc))
                            return
            # Semaphore released — stream immediately.
            try:
                await self._stream_wav_to_connection(conn, loop=loop)
            except Exception as exc:
                logger.error(
                    f"[Connection {conn.connection_id}] "
                    f"Streaming failed: {exc}"
                )
                conn.errored = True
                conn.error_messages.append(str(exc))

        try:
            for batch_start in range(0, self.num_connections, effective_batch):
                if not self.is_active:
                    break
                batch_end = min(batch_start + effective_batch, self.num_connections)
                batch_num = batch_start // effective_batch + 1

                for i in range(batch_start, batch_end):
                    client = self._get_client(i)
                    conn = DeepgramSageMakerConnection(
                        i + 1, client, self.endpoint_name,
                        write_fn=self._safe_print,
                        use_close_stream=self.use_close_stream,
                        raw=self.raw,
                        ring_size=self.ring_size,
                        control_data_type=self.control_data_type,
                    )
                    self.connections.append(conn)

                    task = asyncio.create_task(
                        _open_and_stream(conn),
                        name=f"stream-conn-{conn.connection_id}",
                    )
                    streaming_tasks.append(task)

                logger.info(
                    f"Opening batch {batch_num}/{num_batches}: "
                    f"connection(s) {batch_start + 1}–{batch_end}..."
                )

                if batch_end < self.num_connections and batch_delay > 0:
                    await asyncio.sleep(batch_delay)
        except Exception:
            # Cancel and await any streaming tasks already launched by
            # earlier batches so they don't become orphaned.
            self.is_active = False
            for task in streaming_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*streaming_tasks, return_exceptions=True)
            raise

        self._init_complete = True
        logger.info(f"All {self.num_connections} connection task(s) launched")
        return streaming_tasks

    async def _stream_wav_to_connection(
        self, conn: DeepgramSageMakerConnection, loop: bool = False
    ):
        """
        Stream the full WAV file to a single connection, paced to real-time
        speed.  Loops the file if loop=True; otherwise plays it once.
        """
        bytes_per_frame = self.sample_width * self.channels
        frames_per_chunk = CHUNK_SIZE // bytes_per_frame
        chunk_duration = frames_per_chunk / self.sample_rate

        play_count = 0
        total_chunks = 0
        stream_start = time.monotonic()

        while self.is_active and conn.is_active:
            wf = self._open_wav()
            play_count += 1
            logger.debug(
                f"[Connection {conn.connection_id}] Streaming WAV (pass {play_count})..."
            )

            try:
                while self.is_active and conn.is_active:
                    raw = wf.readframes(frames_per_chunk)
                    if not raw:
                        break

                    await conn.send_audio_chunk(raw)
                    total_chunks += 1

                    elapsed = time.monotonic() - stream_start
                    sleep_time = total_chunks * chunk_duration - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            finally:
                wf.close()

            if not loop:
                break

        logger.info(
            f"[Connection {conn.connection_id}] Finished streaming WAV "
            f"({play_count} pass(es), {total_chunks} chunks)"
        )

        # Signal EOF so the server sends final results.  Without this, idle
        # connections (especially early batches) wait indefinitely for more
        # audio and may be timed out by the server.  When CloseStream is
        # enabled we send {"type":"CloseStream"} but DO NOT close the input
        # stream here — closing it (a WS Close to the container) immediately
        # after CloseStream races the server's finalize and drops the trailing
        # transcript. end_session drains the responses, then closes
        # the input stream. With --no-use-close-stream the bare input-close is
        # the only EOF signal, so do that directly. With --loop this fires once
        # per session (after the loop fully exits), not on every wrap.
        if conn.is_active and conn.stream:
            if conn.use_close_stream:
                await conn._send_close_stream()
            else:
                await conn._close_input_stream()

    def _safe_print(self, text: str):
        """Print a line of text atomically to stdout.

        Uses os.write for a single atomic write syscall so transcript
        output cannot interleave with the status bar's ANSI escape
        sequences or with stderr (logger) output.
        """
        os.write(sys.stdout.fileno(), (text + "\n").encode())

    async def _run_status_dashboard(self, interval: float = 2.0):
        """
        Show live status in the terminal title bar.

        Using the terminal title (OSC escape ``\\033]0;…\\007``) avoids all
        interleaving issues with scroll regions, cursor positioning, and
        concurrent stdout/stderr writes.  The title bar is updated
        independently of the main output area and can never be overwritten
        by transcript lines or log messages.

        Falls back to periodic printed lines when stdout is not a TTY.
        """
        start = time.monotonic()

        # Compute audio duration per chunk from WAV properties
        # (always set by _open_wav before streaming starts).
        assert self.sample_width is not None and self.channels is not None
        assert self.sample_rate is not None
        bytes_per_frame = self.sample_width * self.channels
        frames_per_chunk = CHUNK_SIZE // bytes_per_frame
        chunk_duration = frames_per_chunk / self.sample_rate

        is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

        def _format_status() -> str:
            elapsed = time.monotonic() - start
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            time_str = (
                f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins}:{secs:02d}"
            )

            active = sum(
                1 for c in self.connections if c.is_active and not c.errored
            )
            errored = sum(1 for c in self.connections if c.errored)
            total_transcripts = sum(
                c.transcript_count for c in self.connections
            )

            total_chunks = sum(c.chunk_count for c in self.connections)
            audio_secs = total_chunks * chunk_duration
            if audio_secs >= 3600:
                audio_str = f"{audio_secs / 3600:.1f} hr"
            elif audio_secs >= 60:
                audio_str = f"{audio_secs / 60:.1f} min"
            else:
                audio_str = f"{audio_secs:.0f}s"

            if not self._init_complete:
                phase = f"Opening {active}/{self.num_connections}"
            else:
                phase = f"Streaming {active}/{self.num_connections}"

            err_part = f", {errored} err" if errored else ""

            return (
                f"[{time_str}] {phase}{err_part}"
                f" | {total_transcripts} transcripts"
                f" | {audio_str} audio"
            )

        def _draw(text: str):
            if is_tty:
                # Set terminal title — completely independent of the
                # scrollback area, immune to interleaving.
                os.write(sys.stdout.fileno(), f"\033]0;{text}\007".encode())
            else:
                os.write(sys.stdout.fileno(), f"---{text}---\n".encode())

        try:
            _draw(_format_status())
            while self.is_active:
                await asyncio.sleep(interval)
                if not self.is_active:
                    break
                _draw(_format_status())
        finally:
            if is_tty:
                # Reset title to empty.
                os.write(sys.stdout.fileno(), b"\033]0;\007")

    async def end_all_sessions(self):
        """Close all streaming sessions."""
        if not self.is_active and not self.connections:
            return
        self.is_active = False
        logger.info(f"Closing {len(self.connections)} connection(s)...")
        await asyncio.gather(*[c.end_session() for c in self.connections])
        logger.info("All sessions ended")


# ---------------------------------------------------------------------------
# Batch (pre-recorded) class
# ---------------------------------------------------------------------------

class BatchSTTClient:
    """
    Sends a WAV audio file to a Deepgram STT SageMaker endpoint using the
    synchronous InvokeEndpoint API (pre-recorded / batch mode).

    Deepgram parameters are passed via the
    X-Amzn-SageMaker-Custom-Attributes header, which is forwarded verbatim
    to the container and interpreted as a URL query string.

    Supports concurrent requests with configurable parallelism for throughput
    and latency stress testing.

    Note: SageMaker InvokeEndpoint has a 25 MB request body limit. WAV files
    larger than this cannot be submitted in batch mode; use stream mode or
    split the audio into shorter segments.
    """

    def __init__(self, endpoint_name: str, wav_path: str, region: str = DEFAULT_REGION,
                 fips: bool = False):
        self.endpoint_name = endpoint_name
        self.wav_path = wav_path
        self.region = region
        self.fips = fips
        self._session: boto3.Session | None = None
        self._thread_local = threading.local()

        self.sample_rate: int | None = None
        self.channels: int | None = None
        self.duration_seconds: float | None = None

    def _initialize_client(self):
        """Resolve AWS credentials and verify identity. Stores the session for per-thread client creation."""
        try:
            session = boto3.Session(region_name=self.region)
            credentials = session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()
            credentials.get_frozen_credentials()  # force resolution

            caller_identity = session.client('sts').get_caller_identity()
            logger.debug(f"Authenticated as: {caller_identity.get('Arn', 'Unknown')}")

            self._session = session
            logger.info("AWS credentials resolved")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found. Configure via one of:")
            logger.error("  1. aws configure")
            logger.error("  2. Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
            logger.error("  3. ~/.aws/credentials")
            logger.error("  4. IAM role (when running on AWS infrastructure)")
            raise RuntimeError("AWS credentials not available") from e

    def _get_thread_client(self):
        """Return a per-thread boto3 sagemaker-runtime client, creating one if needed."""
        if not hasattr(self._thread_local, 'sm_client'):
            self._thread_local.sm_client = self._session.client(
                'sagemaker-runtime', config=aws_config(self.fips))
            logger.debug(f"Created sagemaker-runtime client for thread {threading.current_thread().name}")
        return self._thread_local.sm_client

    def load_wav(self) -> bytes:
        """
        Read and validate the WAV file, returning its raw bytes.

        Populates sample_rate, channels, and duration_seconds as a side effect.
        Raises ValueError for non-16-bit PCM files or files exceeding the
        SageMaker InvokeEndpoint 25 MB body limit.
        """
        try:
            with wave.open(self.wav_path, 'rb') as wf:
                sample_width = wf.getsampwidth()
                if sample_width != 2:
                    raise ValueError(
                        f"WAV file must be 16-bit PCM (sample width 2 bytes). "
                        f"Got {sample_width * 8}-bit audio. "
                        "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 output.wav"
                    )
                self.sample_rate = wf.getframerate()
                self.channels = wf.getnchannels()
                self.duration_seconds = wf.getnframes() / self.sample_rate
        except FileNotFoundError:
            logger.error(f"WAV file not found: {self.wav_path}")
            raise
        except wave.Error as e:
            logger.error(f"Invalid WAV file '{self.wav_path}': {e}")
            raise

        with open(self.wav_path, 'rb') as f:
            audio_bytes = f.read()

        if len(audio_bytes) > INVOKE_ENDPOINT_MAX_BYTES:
            mb = len(audio_bytes) / 1_048_576
            raise ValueError(
                f"WAV file is {mb:.1f} MB, which exceeds the SageMaker InvokeEndpoint "
                f"25 MB body limit. Use stream mode for longer audio, or split the file "
                "into shorter segments with: "
                "ffmpeg -i input.wav -f segment -segment_time 60 segment_%03d.wav"
            )

        logger.info(
            f"WAV: {self.wav_path} | {self.sample_rate} Hz | "
            f"{self.channels}ch | {self.duration_seconds:.2f}s | "
            f"{len(audio_bytes) / 1024:.1f} KB"
        )
        return audio_bytes

    def _invoke_once(
        self,
        request_id: int,
        audio_bytes: bytes,
        custom_attributes: str,
        stop_event: threading.Event,
    ) -> tuple[int, float, dict | None, Exception | None]:
        """
        Make a single InvokeEndpoint call on the calling thread.

        Uses a per-thread boto3 client so concurrent calls don't share connections.
        Checks stop_event before starting; sets it on any error so remaining
        pending threads abort without doing work.
        Returns (request_id, elapsed_seconds, parsed_response_or_None, error_or_None).
        """
        if stop_event.is_set():
            return request_id, 0.0, None, RuntimeError("Aborted due to prior error")

        sm_client = self._get_thread_client()
        start = time.monotonic()
        try:
            response = sm_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=audio_bytes,
                ContentType='audio/wav',
                Accept='application/json',
                CustomAttributes=custom_attributes,
            )
            elapsed = time.monotonic() - start
            body_bytes = response['Body'].read()
            result = json.loads(body_bytes)
            return request_id, elapsed, result, None

        except ClientError as e:
            elapsed = time.monotonic() - start
            stop_event.set()
            code = e.response['Error']['Code']
            msg = e.response['Error']['Message']
            if code == 'ModelError':
                # Prefer OriginalMessage (raw container body) over the wrapped Error.Message
                original = e.response.get('OriginalMessage', '')
                if original:
                    try:
                        parsed = json.loads(original)
                        inner = json.dumps(parsed, indent=2)
                    except (json.JSONDecodeError, ValueError):
                        inner = original
                else:
                    inner = _unwrap_sagemaker_model_error(msg)
                err = RuntimeError(
                    f"Model returned an error [{code}]:\n{inner}"
                )
            else:
                err = RuntimeError(
                    f"SageMaker InvokeEndpoint failed [{code}]: {msg}. "
                    "Check the endpoint name, region, and that the endpoint is InService."
                )
            return request_id, elapsed, None, err

        except Exception as e:
            elapsed = time.monotonic() - start
            stop_event.set()
            return request_id, elapsed, None, e

    def run(
        self,
        custom_attributes: str,
        num_requests: int = 1,
        concurrency: int = 1,
    ) -> list[tuple[int, float, dict | None, Exception | None]]:
        """
        Run num_requests InvokeEndpoint calls using a ThreadPoolExecutor with
        up to concurrency threads in parallel.  Each thread maintains its own
        boto3 sagemaker-runtime client to avoid connection contention.

        Returns a list of (request_id, elapsed, result, error) tuples in
        completion order.
        """
        if self._session is None:
            self._initialize_client()

        audio_bytes = self.load_wav()

        stop_event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="batch-invoke",
        )
        try:
            futures = {
                executor.submit(self._invoke_once, i + 1, audio_bytes, custom_attributes, stop_event): i + 1
                for i in range(num_requests)
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    req_id = futures[future]
                    logger.error(f"[Request {req_id}] Unexpected thread error: {e}")
                    stop_event.set()
                    results.append((req_id, 0.0, None, e))
        except (KeyboardInterrupt, SystemExit):
            logger.warning("Batch run interrupted — cancelling pending requests")
            stop_event.set()
            for f in futures:
                f.cancel()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap_streaming_error(exc: BaseException) -> str:
    """
    Extract the most informative message from a streaming exception.

    Walks the exception's attribute set and cause chain looking for richer
    detail than str(exc) alone provides.  Falls back to _unwrap_sagemaker_model_error
    on the string representation when nothing better is found.
    """
    # Attributes that Smithy-generated SDK exceptions and awscrt errors expose
    _DETAIL_ATTRS = ("message", "reason", "error_message", "detail", "body")

    def _candidate(e: BaseException) -> str | None:
        # Prefer dedicated message attributes over str()
        for attr in _DETAIL_ATTRS:
            val = getattr(e, attr, None)
            if val and isinstance(val, str) and val.strip():
                candidate = val.strip()
                # Only use if it carries more info than the string form
                if candidate not in str(e):
                    return candidate
        # awscrt errors expose a numeric code; pair it with the name if present
        code = getattr(e, "code", None)
        name = getattr(e, "name", None)
        if code is not None or name is not None:
            parts = []
            if name:
                parts.append(name)
            if code is not None:
                parts.append(f"code={code}")
            suffix = f" [{', '.join(parts)}]"
            return str(e) + suffix
        return None

    # Walk the cause chain collecting candidates
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidate = _candidate(current)
        if candidate:
            return _unwrap_sagemaker_model_error(candidate)
        cause = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
        current = cause

    return _unwrap_sagemaker_model_error(str(exc))


def _unwrap_sagemaker_model_error(msg: str) -> str:
    """
    Extract the inner model error message from a SageMaker error string.

    Handles two formats:

    Batch (ClientError / ModelError):
        Received client error (NNN) from primary with message "INNER". See ...

    Streaming (exception from awscrt / SageMaker runtime):
        An error occurred while streaming ... primary. See https://... for more information.

    For the batch format the inner message is extracted and pretty-printed if
    it is JSON.  For the streaming format the CloudWatch boilerplate suffix is
    stripped so only the actionable sentence is kept.  Falls back to returning
    the original *msg* unchanged when neither pattern matches.
    """
    import re

    # Batch: extract quoted inner message
    match = re.search(r'with message "(.+?)"\.\s*See ', msg, re.DOTALL)
    if match:
        inner = match.group(1)
        try:
            parsed = json.loads(inner)
            return json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, ValueError):
            return inner

    # Streaming: strip "See https://... for more information." boilerplate
    cleaned = re.sub(r'\s*See https?://\S+ for more information\.?\s*$', '', msg, flags=re.DOTALL).strip()
    if cleaned and cleaned != msg:
        return cleaned

    return msg


def _parse_redact(value: str) -> list[str]:
    """
    Parse a comma-separated --redact string into a validated list of entity types.

    Raises SystemExit with an error message for any unrecognised entity.
    """
    if not value:
        return []
    entities = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if item not in REDACT_ENTITIES:
            print(
                f"ERROR: '{item}' is not a valid redact entity. "
                f"Supported values: {', '.join(REDACT_ENTITIES)}"
            )
            raise SystemExit(1)
        entities.append(item)
    return entities


def _build_batch_query_string(args) -> str:
    """
    Build a Deepgram invocation path + query string for pre-recorded mode.

    The string is passed as X-Amzn-SageMaker-Custom-Attributes and forwarded
    to the Deepgram container.  It must include the API path so the container
    can route the request to the correct Deepgram endpoint, matching how the
    streaming mode passes model_invocation_path="v1/listen".
    """
    params: list[str] = [
        f"model={args.model}",
        f"language={args.language}",
        f"punctuate={args.punctuate}",
        f"diarize={args.diarize}",
    ]
    if args.redact:
        params.extend(f"redact={quote(r)}" for r in args.redact)
    if args.keyterms:
        params.extend(f"keyterm={quote(kt)}" for kt in args.keyterms)
    if args.keywords:
        params.extend(f"keywords={quote(kw)}" for kw in args.keywords)
    return "v1/listen?" + "&".join(params)


def _extract_transcript(result: dict) -> tuple[str, float]:
    """
    Pull the primary transcript and confidence from a Deepgram pre-recorded response.

    Returns ("", 0.0) if the response structure is unexpected.
    """
    try:
        alt = result['results']['channels'][0]['alternatives'][0]
        return alt.get('transcript', ''), alt.get('confidence', 0.0)
    except (KeyError, IndexError):
        return '', 0.0


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by both stream and batch subcommands."""
    parser.add_argument("endpoint_name", help="SageMaker endpoint name")
    parser.add_argument(
        "--file",
        required=True,
        metavar="WAV_FILE",
        help="Path to a 16-bit PCM WAV file",
    )
    parser.add_argument(
        "--model",
        default="nova-3",
        help="Deepgram model (default: nova-3)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--punctuate",
        default="true",
        choices=["true", "false"],
        help="Enable punctuation (default: true)",
    )
    parser.add_argument(
        "--diarize",
        default="false",
        choices=["true", "false"],
        help="Enable speaker diarization (default: false)",
    )
    parser.add_argument(
        "--redact",
        default="",
        metavar="ENTITY,ENTITY,...",
        help=(
            "Comma-separated list of entity types to redact from transcripts. "
            f"Supported values: {', '.join(REDACT_ENTITIES)}"
        ),
    )
    parser.add_argument(
        "--keyterms",
        default="",
        metavar="TERM,TERM,...",
        help=(
            "Comma-separated list of keyterms to boost recognition for "
            "(e.g., 'Deepgram,SageMaker'). Each term is sent as keyterm=<value>."
        ),
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--fips",
        action="store_true",
        help=(
            "Send traffic to the FIPS 140-3 SageMaker runtime endpoint "
            "(runtime-fips.sagemaker.<region>.amazonaws.com) instead of the "
            "standard one. OFF by default. Not available in every region — see "
            "https://docs.aws.amazon.com/general/latest/gr/rande.html#FIPS-endpoints"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

async def run_stream(args) -> int:
    # Ensure we have enough file descriptors for many TCP connections.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired = max(soft, min(hard, 8192))
    if soft < desired:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
        logger.debug(f"Raised fd limit from {soft} to {desired}")

    if getattr(args, "crt_trace", False):
        os.environ["AWS_CRT_LOG_LEVEL"] = "Trace"
        logger.warning("AWS_CRT_LOG_LEVEL=Trace — expect very verbose stderr output")

    keywords_list = []
    if args.keywords:
        for kw in args.keywords.split(','):
            kw = kw.strip()
            if kw:
                keywords_list.append(kw)

    keyterms_list = []
    if args.keyterms:
        for kt in args.keyterms.split(','):
            kt = kt.strip()
            if kt:
                keyterms_list.append(kt)

    redact_list = _parse_redact(args.redact)

    extra_params: dict[str, str] = {}
    for pair in args.extra.split("&"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"ERROR: --extra entry '{pair}' must be in k=v form")
            return 1
        k, v = pair.split("=", 1)
        extra_params[k.strip()] = v.strip()

    if args.connections < 1:
        print("ERROR: --connections must be a positive integer (minimum 1)")
        return 1

    batch_size = args.batch_size if args.batch_size > 0 else args.connections
    if batch_size < 1:
        print("ERROR: --batch-size must be a positive integer (minimum 1)")
        return 1

    client = MultiConnectionWAVClient(
        endpoint_name=args.endpoint_name,
        wav_path=args.file,
        region=args.region,
        num_connections=args.connections,
        use_close_stream=args.use_close_stream,
        raw=args.raw,
        ring_size=args.ring_size,
        control_data_type=args.control_data_type,
        fips=args.fips,
    )

    try:
        wf = client._open_wav()
        wf.close()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    limit_str = (
        f"{args.duration}s" if args.duration
        else ("until file ends (looping)" if args.loop else "until file ends")
    )

    num_batches = (args.connections + batch_size - 1) // batch_size
    batch_str = (
        f"{batch_size} per batch × {num_batches} batch(es)"
        if num_batches > 1
        else f"{args.connections} (single batch)"
    )

    print("=" * 60)
    print("Deepgram SageMaker WAV Streaming Client")
    print("=" * 60)
    print(f"Endpoint:     {args.endpoint_name}")
    print(f"WAV File:     {args.file}")
    print(f"Duration:     {client.duration_seconds:.2f}s")
    print(f"Sample Rate:  {client.sample_rate} Hz")
    print(f"Channels:     {client.channels}")
    print(f"Connections:  {args.connections} ({batch_str})")
    if num_batches > 1:
        print(f"Batch Delay:  {args.batch_delay}s")
    print(f"Model:        {args.model}")
    print(f"Language:     {args.language}")
    print(f"Region:       {args.region}")
    print(f"Bidi URL:     {client.bidi_endpoint}")
    print(f"FIPS:         {'yes' if '-fips.' in client.bidi_endpoint else 'no'}")
    print(f"Loop:         {'yes' if args.loop else 'no'}")
    print(f"Close:        {'CloseStream + WS Close' if args.use_close_stream else 'bare WS Close'}")
    if args.use_close_stream:
        frame = "Text" if args.control_data_type == "UTF8" else "Binary"
        print(f"Control:      DataType={args.control_data_type} (WS {frame} frame)")
    print(f"Limit:        {limit_str}")
    if redact_list:
        print(f"Redact:       {', '.join(redact_list)}")
    if keywords_list:
        print(f"Keywords:     {', '.join(keywords_list)}")
    if keyterms_list:
        print(f"Keyterms:     {', '.join(keyterms_list)}")
    print("=" * 60)

    def signal_handler(sig, frame):
        client._safe_print("\n\nReceived interrupt signal, stopping...")
        client.is_active = False

    signal.signal(signal.SIGINT, signal_handler)

    wall_start = time.monotonic()
    dashboard_task = asyncio.create_task(client._run_status_dashboard())

    try:
        streaming_tasks = await client.initialize_connections(
            batch_size=batch_size,
            batch_delay=args.batch_delay,
            model=args.model,
            language=args.language,
            diarize=args.diarize,
            punctuate=args.punctuate,
            interim_results="true" if args.interim_results else "false",
            keywords=keywords_list,
            keyterms=keyterms_list,
            redact_entities=redact_list,
            loop=args.loop,
            **extra_params,
        )

        client._safe_print("\n" + "=" * 60)
        client._safe_print(f"LIVE TRANSCRIPTION - {args.connections} Connection(s)")
        if args.duration:
            client._safe_print(f"   (Running for {args.duration}s, or press Ctrl+C to stop early)")
        else:
            client._safe_print("   (Press Ctrl+C to stop)")
        client._safe_print("=" * 60 + "\n")

        all_streaming = asyncio.gather(*streaming_tasks, return_exceptions=True)
        if args.duration:
            try:
                results = await asyncio.wait_for(all_streaming, timeout=args.duration)
            except asyncio.TimeoutError:
                client._safe_print(f"\nDuration of {args.duration}s reached, stopping...")
                client.is_active = False
                for task in streaming_tasks:
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(*streaming_tasks, return_exceptions=True)
        else:
            results = await all_streaming

        # Surface any exceptions from streaming tasks that would otherwise
        # be silently swallowed by return_exceptions=True.
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                conn_id = i + 1
                logger.error(
                    f"[Connection {conn_id}] Streaming task failed: {result}",
                    exc_info=result,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        return 1
    finally:
        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass
        await client.end_all_sessions()
        wall_elapsed = time.monotonic() - wall_start

        conns = client.connections
        successful = [c for c in conns if not c.errored]
        errored = [c for c in conns if c.errored]

        # Aggregate error messages across all connections (deduplicated with counts)
        all_errors: dict[str, int] = {}
        for c in errored:
            for msg in c.error_messages:
                all_errors[msg] = all_errors.get(msg, 0) + 1

        print("\n" + "=" * 60)
        print("STREAM SUMMARY")
        print("=" * 60)
        print(f"Total connections:  {len(conns)}")
        print(f"Successful:         {len(successful)}")
        print(f"Errored:            {len(errored)}")
        print(f"Total wall time:    {wall_elapsed:.2f}s")

        with_rid = sum(1 for c in conns if c.dg_request_id)
        total_finals = sum(c.transcript_count for c in conns)
        total_interims = sum(c.interim_count for c in conns)
        total_unknown = sum(c.unknown_message_count for c in conns)
        ffl = [
            c.summary()["first_final_latency_s"] for c in conns
            if c.summary()["first_final_latency_s"] is not None
        ]
        agg_types: Counter[str] = Counter()
        for c in conns:
            agg_types.update(c.message_type_counts)

        print(f"With dg_request_id: {with_rid}/{len(conns)}")
        print(f"Final transcripts:  {total_finals}")
        print(f"Interim results:    {total_interims}")
        print(f"Unknown messages:   {total_unknown}")
        if agg_types:
            print(f"Msg-type histogram: {dict(agg_types)}")
        if ffl:
            ffl_sorted = sorted(ffl)
            p50 = ffl_sorted[len(ffl_sorted) // 2]
            p95 = ffl_sorted[max(0, int(len(ffl_sorted) * 0.95) - 1)]
            print(
                f"First-final latency (s): "
                f"min={min(ffl):.2f} p50={p50:.2f} p95={p95:.2f} max={max(ffl):.2f}"
            )

        if all_errors:
            print("\nErrors:")
            for msg, count in all_errors.items():
                prefix = f"  (x{count}) " if count > 1 else "  "
                print(f"{prefix}{msg}")

        if args.summary_jsonl:
            try:
                with open(args.summary_jsonl, "w") as f:
                    for c in conns:
                        f.write(json.dumps(c.summary()) + "\n")
                print(f"\nPer-connection summary written: {args.summary_jsonl}")
            except OSError as e:
                logger.error(f"Could not write --summary-jsonl {args.summary_jsonl}: {e}")

        print("=" * 60)

    logger.info("Transcription complete")
    return 0 if not any(c.errored for c in client.connections) else 1


async def run_batch(args) -> int:
    redact_list = _parse_redact(args.redact)

    if args.concurrency < 1:
        print("ERROR: --concurrency must be a positive integer (minimum 1)")
        return 1
    if args.requests is None:
        args.requests = args.concurrency
    if args.requests < 1:
        print("ERROR: --requests must be a positive integer (minimum 1)")
        return 1
    if args.concurrency > args.requests:
        print("WARNING: --concurrency exceeds --requests; clamping to request count")
        args.concurrency = args.requests

    client = BatchSTTClient(
        endpoint_name=args.endpoint_name,
        wav_path=args.file,
        region=args.region,
        fips=args.fips,
    )

    # Load WAV early for validation and metadata
    try:
        client._initialize_client()
        audio_bytes = client.load_wav()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    args.redact = redact_list
    args.keyterms = [kt.strip() for kt in args.keyterms.split(',') if kt.strip()]
    args.keywords = [kw.strip() for kw in args.keywords.split(',') if kw.strip()]
    custom_attributes = _build_batch_query_string(args)

    print("=" * 60)
    print("Deepgram SageMaker WAV Batch Client (Pre-Recorded)")
    print("=" * 60)
    print(f"Endpoint:     {args.endpoint_name}")
    print(f"WAV File:     {args.file}")
    print(f"File Size:    {len(audio_bytes) / 1024:.1f} KB")
    print(f"Duration:     {client.duration_seconds:.2f}s")
    print(f"Sample Rate:  {client.sample_rate} Hz")
    print(f"Channels:     {client.channels}")
    print(f"Model:        {args.model}")
    print(f"Language:     {args.language}")
    print(f"Region:       {args.region}")
    _rt_url = client._get_thread_client().meta.endpoint_url
    print(f"Runtime URL:  {_rt_url}")
    print(f"FIPS:         {'yes' if '-fips.' in _rt_url else 'no'}")
    print(f"Requests:     {args.requests}")
    print(f"Concurrency:  {args.concurrency}")
    if redact_list:
        print(f"Redact:       {', '.join(redact_list)}")
    if args.keywords:
        print(f"Keywords:     {', '.join(args.keywords)}")
    if args.keyterms:
        print(f"Keyterms:     {', '.join(args.keyterms)}")
    print(f"Path+Params:  {custom_attributes}")
    print("=" * 60)

    print(f"\nRunning {args.requests} request(s) "
          f"({args.concurrency} concurrent)...\n")

    wall_start = time.monotonic()

    results = await asyncio.to_thread(
        client.run,
        custom_attributes,
        args.requests,
        args.concurrency,
    )

    wall_elapsed = time.monotonic() - wall_start

    # Sort by request_id for consistent output
    results.sort(key=lambda r: r[0])

    successes = 0
    failures = 0
    latencies: list[float] = []

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for req_id, elapsed, result, error in results:
        if error:
            failures += 1
            print(f"[Request {req_id:>3}] ERROR ({elapsed:.2f}s): {error}")
        else:
            successes += 1
            latencies.append(elapsed)
            transcript, confidence = _extract_transcript(result)
            if transcript.strip():
                print(
                    f"[Request {req_id:>3}] ✓ ({elapsed:.2f}s) "
                    f"({confidence:.1%}) {transcript}"
                )
            else:
                print(f"[Request {req_id:>3}] ✓ ({elapsed:.2f}s) (no transcript)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total requests:  {args.requests}")
    print(f"Successful:      {successes}")
    print(f"Failed:          {failures}")

    if latencies:
        latencies_sorted = sorted(latencies)
        p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
        print(f"Min latency:     {min(latencies):.2f}s")
        print(f"Avg latency:     {statistics.mean(latencies):.2f}s")
        print(f"P95 latency:     {latencies_sorted[p95_idx]:.2f}s")
        print(f"Max latency:     {max(latencies):.2f}s")
        if len(latencies) > 1:
            print(f"Std dev:         {statistics.stdev(latencies):.2f}s")

    print(f"Total wall time: {wall_elapsed:.2f}s")
    if args.requests > 0 and wall_elapsed > 0:
        print(f"Throughput:      {args.requests / wall_elapsed:.2f} req/s")
    print("=" * 60)

    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream or batch-transcribe a WAV file using a Deepgram model "
            "deployed on Amazon SageMaker."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # -- stream subcommand ----------------------------------------------------
    stream_parser = subparsers.add_parser(
        "stream",
        help="Bidirectional streaming transcription (real-time, /v1/listen WebSocket)",
        description=(
            "Stream a WAV file in real-time to one or more simultaneous Deepgram "
            "connections on SageMaker using bidirectional HTTP/2 streaming."
        ),
    )
    _add_common_args(stream_parser)
    stream_parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="Total number of simultaneous streaming connections (default: 1)",
    )
    stream_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of new connections to open per batch (default: open all at once). "
            "Streaming begins immediately when each batch opens. "
            "Use with --batch-delay to ramp up load gradually."
        ),
    )
    stream_parser.add_argument(
        "--batch-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Seconds to wait between opening connection batches (default: 0)",
    )
    stream_parser.add_argument(
        "--keywords",
        default="",
        metavar="KEYWORD[:INTENSITY],KEYWORD[:INTENSITY],...",
        help=(
            "Comma-separated list of keywords to boost recognition for nova-2 and earlier models "
            "(e.g., 'Deepgram:5,SageMaker'). Append :INTENSITY to boost (positive) or suppress "
            "(negative) a keyword. Each keyword is sent as keywords=<value>. "
            "For nova-3, use --keyterms instead."
        ),
    )
    stream_parser.add_argument(
        "--interim-results",
        action="store_true",
        default=False,
        help="Enable interim (partial) results (default: disabled)",
    )
    stream_parser.add_argument(
        "--extra",
        default="",
        metavar="k=v&k2=v2",
        help=(
            "Extra Deepgram query parameters appended verbatim to the request "
            "(e.g. 'sentiment=true&topics=true&detect_language=true'). Use to "
            "exercise features without a dedicated flag."
        ),
    )
    stream_parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Print every response frame's raw JSON (for inspecting feature metadata).",
    )
    stream_parser.add_argument(
        "--use-close-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Send a Deepgram CloseStream message before closing the WebSocket so "
            "the server flushes the final transcript tail "
            "(https://developers.deepgram.com/docs/close-stream). "
            "Pass --no-use-close-stream to exercise the bare WebSocket Close path "
            "instead (default: enabled)."
        ),
    )
    stream_parser.add_argument(
        "--control-data-type",
        choices=["UTF8", "BINARY"],
        default="UTF8",
        help=(
            "PayloadPart.DataType used for Deepgram control messages such as "
            "CloseStream. SageMaker maps this to the WebSocket opcode delivered "
            "to the container: UTF8 -> Text frame, BINARY -> Binary frame. "
            "Control messages are only parsed from Text frames, so UTF8 is "
            "correct and is what the Deepgram SageMaker docs recommend. BINARY "
            "reaches the API only via the container's binary-control reframing "
            "fallback — select it to exercise that path (default: UTF8). "
            "Audio chunks are always BINARY."
        ),
    )
    stream_parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the WAV file continuously until --duration is reached or Ctrl+C",
    )
    stream_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop automatically after this many seconds (default: run until file ends or Ctrl+C)",
    )
    stream_parser.add_argument(
        "--ring-size",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Per-connection ring buffer of last-N raw response frames "
            "(tail dumped on error + in --summary-jsonl). Default: 50."
        ),
    )
    stream_parser.add_argument(
        "--summary-jsonl",
        default=None,
        metavar="PATH",
        help=(
            "Write one JSONL line per connection at end of run with "
            "dg_request_id, per-message-type counts, transcript counts, "
            "first-final latency, close reason, error messages, and ring-buffer "
            "tail. Use to correlate client-visible behavior with container CloudWatch "
            "logs by dg_request_id."
        ),
    )
    stream_parser.add_argument(
        "--crt-trace",
        action="store_true",
        default=False,
        help=(
            "Enable awscrt trace logging (sets AWS_CRT_LOG_LEVEL=Trace). Surfaces "
            "HTTP/2 PING/GOAWAY/RST_STREAM frames the smithy bidi-stream API "
            "otherwise hides. Extremely verbose — use only for narrow repro."
        ),
    )

    # -- batch subcommand -----------------------------------------------------
    batch_parser = subparsers.add_parser(
        "batch",
        help="Synchronous pre-recorded transcription (HTTP POST, SageMaker InvokeEndpoint)",
        description=(
            "Submit a WAV file as a single HTTP POST using the SageMaker InvokeEndpoint "
            "API and the Deepgram pre-recorded /v1/listen REST endpoint. "
            f"File size is limited to {INVOKE_ENDPOINT_MAX_BYTES // 1_048_576} MB by "
            "SageMaker. Use --requests and --concurrency to stress-test throughput."
        ),
    )
    _add_common_args(batch_parser)
    batch_parser.add_argument(
        "--keywords",
        default="",
        metavar="KEYWORD[:INTENSITY],KEYWORD[:INTENSITY],...",
        help=(
            "Comma-separated list of keywords to boost recognition for nova-2 and earlier models "
            "(e.g., 'Deepgram:5,SageMaker'). Append :INTENSITY to boost (positive) or suppress "
            "(negative) a keyword. Each keyword is sent as keywords=<value>. "
            "For nova-3, use --keyterms instead."
        ),
    )
    batch_parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Total number of InvokeEndpoint requests to send (default: same as --concurrency)",
    )
    batch_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of requests to run in parallel (default: 1)",
    )

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 0

    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = _AtomicStderrHandler(fmt=log_fmt)
    handler.setLevel(getattr(logging, args.log_level))
    logging.root.handlers = [handler]
    logging.root.setLevel(getattr(logging, args.log_level))

    if args.subcommand == "stream":
        return await run_stream(args)
    else:
        return await run_batch(args)


if __name__ == "__main__":
    # Suppress noisy CRT InvalidStateError when _on_complete fires on
    # an already-cancelled future during connection timeout/retry.
    _default_unraisablehook = sys.unraisablehook

    def _quiet_unraisablehook(args):
        if args.exc_type is not None and issubclass(args.exc_type, InvalidStateError):
            return
        _default_unraisablehook(args)

    from concurrent.futures._base import InvalidStateError
    sys.unraisablehook = _quiet_unraisablehook

    sys.exit(asyncio.run(main()))

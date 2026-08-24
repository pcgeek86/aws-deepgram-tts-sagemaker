"""SageMaker client for Deepgram Flux TTS on `/v2/speak`.

Two surfaces, both served by the SAME `streaming` container image:

  FluxTtsStream  — WebSocket `/v2/speak` over SageMaker bidirectional streaming
                   (`invoke_endpoint_with_bidirectional_stream`).
  invoke_batch() — `POST /invocations`, routed to `POST /v2/speak` by the
                   `x-amzn-sagemaker-custom-attributes` header.

Differences from the Aura-2 client (`python-tts/tts_stress.py`) that are easy to
get wrong:

  * `model_invocation_path` is `v2/speak`, not `v1/speak`.
  * The turn model is explicit. Text sent via `Speak` accumulates into the
    ACTIVE TURN; `Flush` ends it. `SpeechMetadata` (the per-turn billing record)
    is emitted only at a manual `Flush`, never at the server's internal
    auto-flush boundaries — so a client that never flushes never sees one.
  * `/v2/speak` rejects unknown query params, so a stray or misspelled param is
    a hard 400 rather than being ignored. A param the deployment carries no rate
    for is likewise refused before synthesis, as a 400 `unsupported_parameter`.
  * Server frames are a tagged union on `type`; audio arrives as binary frames
    with no JSON wrapper.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
from smithy_aws_core.identity import EnvironmentCredentialsResolver

from e2e.e2e_test_common import ENCODING, SAMPLE_RATE

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "flux-alexis-en"

#: SageMaker's bidirectional-streaming port. NOT 443 — see `make_client`.
BIDI_PORT = 8443


def bidi_endpoint_uri(region: str, fips: bool = False) -> str:
    """The SageMaker runtime URI for bidirectional streaming (port 8443).

    `fips` selects the FIPS 140-3 runtime host. Port 8443 IS served there —
    verified 2026-08-20 in us-west-2 — which was the open question, since
    nothing in the AWS docs states that bidi streaming is available on the FIPS
    endpoint at all. This selects the AWS endpoint only and says nothing about
    the container's own crypto.
    """
    host = "runtime-fips" if fips else "runtime"
    return f"https://{host}.sagemaker.{region}.amazonaws.com:{BIDI_PORT}"


def make_client(region: str, fips: bool = False) -> SageMakerRuntimeHTTP2Client:
    """Build the HTTP/2 SageMaker runtime client used for bidi streaming.

    Credentials come from the ENVIRONMENT, not the shared config file — the
    smithy client's resolver is `EnvironmentCredentialsResolver`. Callers using
    `AWS_PROFILE` must therefore materialise that profile into
    AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN first; `ensure_env_credentials()`
    below does it.
    """
    # `endpoint_uri` with **PORT 8443** is MANDATORY for bidirectional streaming.
    # SageMaker serves `invoke_endpoint_with_bidirectional_stream` on 8443, not on
    # the default 443 that the SDK's endpoint resolver produces. Omit it and the
    # call reaches `runtime.sagemaker.<region>.amazonaws.com:443`, which accepts
    # the connection and then never answers: the client blocks forever in
    # `await_output()`, the container sees nothing, and the endpoint's
    # `Invocations` metric stays 0 with no 4XX/5XX. There is no error to read —
    # it looks exactly like a broken endpoint or a service outage. (Diagnosed
    # 2026-08-10 after chasing it through two accounts' worth of red herrings;
    # `python-stt`, `python-flux` and `python-tts` all set this.)
    #
    # NB: the kwarg is `auth_scheme_resolver`, NOT `http_auth_scheme_resolver`
    # — the latter is from an older smithy-core and raises TypeError on the
    # pinned version. Matches `python-flux/flux_stress.py`.
    return SageMakerRuntimeHTTP2Client(
        config=Config(
            endpoint_uri=bidi_endpoint_uri(region, fips),
            region=region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
        )
    )


def ensure_env_credentials(region: str) -> None:
    """Materialise the active boto3 credentials into the process environment.

    The bidi smithy client only reads env credentials, so an `AWS_PROFILE`-based
    invocation would otherwise fail to sign. Idempotent.
    """
    import os

    import boto3

    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return
    creds = boto3.Session(region_name=region).get_credentials()
    if creds is None:
        raise SystemExit(
            "no AWS credentials found — set AWS_PROFILE or the AWS_* env vars"
        )
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    os.environ.setdefault("AWS_DEFAULT_REGION", region)


class FluxTtsStream:
    """One bidirectional `/v2/speak` session against a SageMaker endpoint."""

    def __init__(self, client, endpoint_name: str, connection_id: int = 0):
        self.client = client
        self.endpoint_name = endpoint_name
        self.connection_id = connection_id

        self.stream = None
        self.output_stream = None
        self.reader_task: asyncio.Task | None = None
        self.is_active = False
        self.close_sent = False

        # Telemetry
        self.audio_bytes = bytearray()
        self.audio_frames = 0
        self.first_audio_at: float | None = None
        self.session_start_at: float | None = None
        self.connected: dict | None = None
        self.speech_started: list[str] = []
        self.flushed: list[str] = []
        self.turns: list[dict] = []          # SpeechMetadata payloads
        self.interrupted: list[dict] = []    # SpeechInterrupted payloads
        self.session_metadata: dict | None = None
        self.configure_ok: list[dict] = []
        self.configure_failed: list[dict] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self._flushed_event = asyncio.Event()
        self._turn_event = asyncio.Event()

    # -- lifecycle ---------------------------------------------------------

    async def start(self, voice: str = DEFAULT_VOICE, **params) -> None:
        """Open the session.

        Only params that are BOTH valid `/v2/speak` query params AND carried in
        the deployment's rate card will get through; anything else is a 400.
        """
        query = {"model": voice, "encoding": ENCODING, "sample_rate": SAMPLE_RATE}
        query.update(params)
        query_string = "&".join(f"{k}={v}" for k, v in query.items())

        self.session_start_at = time.monotonic()
        logger.debug("[%s] open v2/speak?%s", self.connection_id, query_string)

        self.stream = await self.client.invoke_endpoint_with_bidirectional_stream(
            InvokeEndpointWithBidirectionalStreamInput(
                endpoint_name=self.endpoint_name,
                model_invocation_path="v2/speak",
                model_query_string=query_string,
            )
        )
        self.is_active = True
        _, self.output_stream = await self.stream.await_output()
        self.reader_task = asyncio.create_task(self._read_loop())
        await asyncio.sleep(0.1)  # let the reader attach before the first send

    async def _send(self, message: dict) -> bool:
        if not self.is_active:
            return False
        try:
            await self.stream.input_stream.send(
                RequestStreamEventPayloadPart(
                    value=RequestPayloadPart(
                        bytes_=json.dumps(message).encode("utf-8"), data_type="UTF8"
                    )
                )
            )
            return True
        except Exception as e:  # noqa: BLE001 — transport errors are all fatal here
            logger.error("[%s] send %s failed: %s", self.connection_id, message.get("type"), e)
            self.is_active = False
            return False

    # -- client messages ---------------------------------------------------

    async def speak(self, text: str) -> bool:
        """Append text to the active turn. The server inserts NO whitespace
        between successive `Speak` messages — the caller owns spacing."""
        return await self._send({"type": "Speak", "text": text})

    async def flush(self) -> bool:
        """End the active turn. Triggers `Flushed` then, once the turn's audio is
        fully sent, exactly one `SpeechMetadata`."""
        self._flushed_event.clear()
        self._turn_event.clear()
        return await self._send({"type": "Flush"})

    async def configure(self, *, speed: float | None = None) -> bool:
        cfg: dict = {}
        if speed is not None:
            cfg["speed"] = speed
        return await self._send({"type": "Configure", **cfg})

    async def interrupt(self, *, playback_offset_ms: int | None = None) -> bool:
        """Cancel the active turn.

        `playback_offset` is the ONLY field `Interrupt` accepts, and the message
        rejects unknown fields — sending anything else (e.g. a `speech_id`, which
        is not part of this message) makes the whole frame unparseable and comes
        back as `[MESSAGE-0000]`. There is no need to name the turn: Interrupt
        always applies to the active one.

        The offset is measured in milliseconds from the start of the SESSION's
        audio, not the current turn, and must advance past the position the
        previous interrupt established. Omitting it means the server will not
        report `text_spoken`/`text_remaining`.
        """
        msg: dict = {"type": "Interrupt"}
        if playback_offset_ms is not None:
            msg["playback_offset"] = {"type": "time_ms", "value": playback_offset_ms}
        return await self._send(msg)

    async def close(self) -> bool:
        ok = await self._send({"type": "Close"})
        self.close_sent = True
        return ok

    # -- waiting -----------------------------------------------------------

    async def wait_for_flushed(self, timeout: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._flushed_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_turn(self, timeout: float = 120.0) -> bool:
        """Wait for the turn's `SpeechMetadata` — i.e. all of the turn's audio
        has been sent. This, not `Flushed`, is the end-of-turn signal."""
        try:
            await asyncio.wait_for(self._turn_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # -- server frames -----------------------------------------------------

    def _handle_frame(self, data: bytes) -> None:
        try:
            msg = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            msg = None

        # Not JSON → binary audio. Do this by exclusion rather than by sniffing
        # magic bytes: raw linear16 PCM has no header to sniff.
        if not isinstance(msg, dict) or "type" not in msg:
            if self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
            self.audio_bytes.extend(data)
            self.audio_frames += 1
            return

        t = msg["type"]
        if t == "Connected":
            self.connected = msg
        elif t == "SpeechStarted":
            self.speech_started.append(msg.get("speech_id", ""))
        elif t == "Flushed":
            self.flushed.append(msg.get("speech_id", ""))
            self._flushed_event.set()
        elif t == "SpeechMetadata":
            self.turns.append(msg)
            self._turn_event.set()
        elif t == "SpeechInterrupted":
            self.interrupted.append(msg)
            # Carries the same per-turn billing block, nested — count it as the
            # turn's record so character totals reconcile.
            if isinstance(msg.get("metadata"), dict):
                self.turns.append(msg["metadata"])
            self._turn_event.set()
        elif t == "SessionMetadata":
            self.session_metadata = msg
        elif t == "ConfigureSuccess":
            self.configure_ok.append(msg.get("applied", {}))
        elif t == "ConfigureFailure":
            self.configure_failed.append(msg)
        elif t == "Warning":
            self.warnings.append(
                f"[{msg.get('warn_code') or msg.get('code', '')}] "
                f"{msg.get('warn_msg') or msg.get('description', '')}".strip()
            )
        elif t == "Error":
            # Always fatal per the protocol: followed by a WS close.
            self.errors.append(
                f"[{msg.get('err_code') or msg.get('code', '')}] "
                f"{msg.get('err_msg') or msg.get('description', '')}".strip()
            )
            self._flushed_event.set()
            self._turn_event.set()
        else:
            logger.debug("[%s] unhandled frame type %r", self.connection_id, t)

    async def _read_loop(self) -> None:
        try:
            while self.is_active:
                result = await self.output_stream.receive()
                if result is None:
                    break
                if result.value and result.value.bytes_:
                    self._handle_frame(result.value.bytes_)
            # Drain whatever is still buffered after the server closed. Bounded
            # so a wedged stream can't hang the run.
            for _ in range(50):
                try:
                    result = await asyncio.wait_for(self.output_stream.receive(), 0.5)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    break
                if result is None:
                    break
                if result.value and result.value.bytes_:
                    self._handle_frame(result.value.bytes_)
        except Exception as e:  # noqa: BLE001
            # The bidi transport raises on essentially every teardown; that is
            # not a test failure on its own (see README).
            logger.debug("[%s] read loop ended: %s", self.connection_id, e)

    async def finish(self, timeout: float = 30.0) -> None:
        """Close the input side and let the reader drain."""
        try:
            if not self.close_sent:
                await self.close()
            if self.stream is not None:
                try:
                    await self.stream.input_stream.close()
                except Exception:  # noqa: BLE001, S110
                    pass
            if self.reader_task is not None:
                try:
                    await asyncio.wait_for(self.reader_task, timeout)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    self.reader_task.cancel()
        finally:
            self.is_active = False

    # -- convenience -------------------------------------------------------

    @property
    def ttfa_s(self) -> float | None:
        """Time from session open to first audio byte."""
        if self.session_start_at is None or self.first_audio_at is None:
            return None
        return self.first_audio_at - self.session_start_at

    @property
    def billable_chars(self) -> int:
        if self.session_metadata:
            return int(self.session_metadata.get("total_billable_character_count", 0))
        return sum(int(t.get("billable_character_count", 0)) for t in self.turns)


# ---------------------------------------------------------------------------
# Batch surface
# ---------------------------------------------------------------------------

def invoke_batch(
    runtime_client,
    endpoint_name: str,
    text: str,
    voice: str = DEFAULT_VOICE,
    **params,
) -> tuple[bytes, dict]:
    """`POST /invocations` → `POST /v2/speak`. Returns `(audio, meta)`.

    The target path is chosen by `CustomAttributes`, NOT by the URL — the
    `/invocations` handler reads
    `x-amzn-sagemaker-custom-attributes: <path>?<query>` and proxies there
    (defaulting to `v1/listen` when absent, which would 400 for TTS).
    `Content-Type: application/json` matters too: the server needs it to read the
    `{"text": ...}` body, and the container forwards `content-type` through to it.
    """
    query = {"model": voice, "encoding": ENCODING, "sample_rate": SAMPLE_RATE}
    query.update(params)
    query_string = "&".join(f"{k}={v}" for k, v in query.items())

    started = time.monotonic()
    resp = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="*/*",
        CustomAttributes=f"v2/speak?{query_string}",
        Body=json.dumps({"text": text}).encode("utf-8"),
    )
    audio = resp["Body"].read()
    return audio, {
        "elapsed_s": time.monotonic() - started,
        "content_type": resp.get("ContentType", ""),
        "request_id": resp.get("ResponseMetadata", {})
        .get("HTTPHeaders", {})
        .get("x-amzn-sagemaker-request-id", ""),
    }

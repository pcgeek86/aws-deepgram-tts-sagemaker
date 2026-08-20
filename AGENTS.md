# AGENTS.md

Notes for AI agents working in this repo.

## This is an external, customer-facing repository — no internal codenames

Customers read this code. **Never use Deepgram-internal component codenames**
anywhere they can be seen — comments, docstrings, `--help` text, log messages,
scenario names/notes, commit messages, READMEs. Use the external name a
customer would recognise from the public docs:

| internal codename | use instead |
|---|---|
| stem | the API / the API server |
| shim | the container / the Deepgram container |
| impeller | the engine |
| Wilhelmina | the model catalog |

Also **do not cite internal source**: no `listen.rs:2804-2822` line references,
no `inference-shim/src/...` paths, no internal symbol names, no "CLAUDE.md
says …". Keep the substance (what the behavior is, when it was confirmed) and
drop the pointer.

Exception: strings that are real identifiers rather than descriptions — model
and component names a customer actually passes or stages (`nova-3`,
`flux-general-multi`, `semantic_tagger`, `g2p`, and their UUIDs) — stay as they
are. The test is whether renaming it would break something; if it is prose,
rename it.

Check before pushing:

```bash
grep -rniw "stem\|shim\|impeller\|wilhelmina" \
  --include='*.py' --include='*.ts' --include='*.java' --include='*.md' .
```

## Keep the top-level README in sync

[`README.md`](README.md) at the repo root is the index of every script, grouped
by service (STT / TTS / Flux) and language (JavaScript / Python / Java). When
you add, rename, remove, or change the purpose of anything user-facing in a
subfolder, update that index in the **same change** — do not leave it for later.
This includes:

- a new script or CLI subcommand (e.g. a new `*_stress.py` mode),
- a new `e2e/` driver or scenario surface,
- a new component/language subfolder (add a section and a pointer to its README),
- a renamed or deleted file that the README links to.

How to keep it consistent:

- Match the existing entry style: a markdown link to the file, then an em-dash
  one-line description at index altitude. Leave the deep detail (flags,
  pass/fail, parameter matrices) in the subfolder's own README and link to it.
- Every path the README links to must exist — verify after editing.
- Mirror the change in the relevant subfolder README too, and if it's an e2e
  driver with a non-trivial runtime, add a row to the wall-clock table below.

The same rule applies to this file: when an e2e driver's runtime, pass/fail
format, or invocation contract changes, update the relevant section here.

## E2E wall-clock expectations

Pick the right runner. The streaming STT e2e includes a real-time
sustained-load scenario whose floor is the duration of the audio file. Long
scenarios will exceed agent harness time limits in many platforms — run the
streaming STT e2e as a detached background process with an explicit long
timeout; the others fit easily in a single agent turn.

| Script | Total wall-clock | Longest scenario | Notes |
|---|---|---|---|
| `python-stt/e2e/e2e_test_streaming.py` | **~17–18 min** | `concurrent_10x_15min` ≈ **915 s** | Plays a ~15-min file at real-time across 10 concurrent WS connections. Do not run inside a short-lived agent subprocess — run as a backgrounded shell command with a 40+ min timeout and poll for the `PASSED:.*FAILED:` line in the tail. |
| `python-stt/e2e/e2e_test_batch.py` | ~1 min | summarize/topics scenarios ~4 s | 22 scenarios, mostly 1–2 s each. Safe to run inline. |
| `python-tts/e2e/e2e_test_batch.py` | ~1.5 min | speed_duration ~12 s | 20 scenarios. Safe inline. |
| `python-tts/e2e/e2e_test_streaming.py` | ~1.5 min | multi_phrase_flush ~17 s | 8 scenarios. Safe inline. |
| `python-flux-tts/e2e/e2e_test_batch.py` | ~2 min | `concurrent_5`, `speed` | 7 scenarios. Safe inline. |
| `python-flux-tts/e2e/e2e_test_streaming.py` | ~3 min | `speed`, `long_turn` | 10 scenarios. `speed` renders the same text twice (default + 1.3) to compare durations, so it costs two turns. Safe inline. |

Wall-clocks measured against single-GPU (STT, Flux TTS) and multi-GPU (Aura-2
TTS) SageMaker endpoints. Network and instance class shift the numbers, but the
streaming-STT floor is set by real-time playback of the 15-min sample file and
won't change.

## Invocation defaults

All scripts take the endpoint name as the first positional arg and default
`--region us-east-2`. They use boto3 and respect any standard AWS credential
chain (`AWS_PROFILE`, env vars, instance role). Typical invocation:

```bash
cd python-stt
uv run e2e/e2e_test_streaming.py <endpoint-name>
```

`uv sync` is not required up front — `uv run` resolves the project venv on
first call.

### `--fips` routes traffic to the FIPS 140-3 endpoints (OFF by default)

`python-stt`'s `e2e_test_streaming.py`, `e2e_test_batch.py` and
`stt_wav_stress.py` (both `stream` and `batch`) take `--fips`, which moves every
AWS call in the run onto the FIPS endpoints — `runtime-fips.sagemaker.<region>
.amazonaws.com`, `api-fips.…`, `s3-fips.…`. Omit it and nothing changes. Not
every region has them: https://docs.aws.amazon.com/general/latest/gr/rande.html#FIPS-endpoints

Each run prints the resolved URL plus `FIPS: yes|no` in its header, so a log
proves which endpoints were used rather than asserting it.

Two non-obvious things:

- **The bidi stream needs this explicitly — it cannot inherit FIPS from AWS
  config.** The smithy HTTP/2 client takes a literal `endpoint_uri` and performs
  no endpoint resolution, so the streaming URL used to be a hardcoded
  `runtime.sagemaker.<region>.amazonaws.com:8443`. Without the flag the stream
  would silently stay on the non-FIPS host while every boto3 call in the same run
  honored the FIPS setting. `resolve_bidi_endpoint()` now derives it from
  botocore's own `sagemaker-runtime` resolution and appends `:8443`. Verified
  2026-08-20: bidi streaming **does** work on port 8443 of the FIPS host
  (nova-3 monolingual, us-west-2).
- **Do NOT set `AWS_USE_FIPS_ENDPOINT=true` process-wide with an SSO profile.**
  That variable also redirects the IAM Identity Center portal, and
  `portal.sso-fips.<region>.amazonaws.com` does not exist, so the run dies during
  credential resolution with `EndpointConnectionError` naming an SSO URL — before
  it reaches SageMaker at all. `--fips` applies FIPS **per client** instead, which
  leaves the credential chain on its normal endpoints. A warm credential cache
  masks the failure, so it presents as intermittent.

Coverage is `python-stt` only. `python-flux`, `python-tts`, `python-flux-tts`,
the JS clients and the Java load test still hardcode the non-FIPS bidi host; each
needs the same one-line change to gain FIPS support.

### Multilingual endpoints take `--language multi`, not a specific code

A multilingual STT listing (e.g. the **Nova-3 Multilingual STT Streaming**
marketplace product) loads ONE model registered under `language=multi`, not a
per-language model. Invoke it with `--language multi` — the engine then runs
language detection across the bundled languages. Passing a specific
`--language en` (which is the **default**) resolves to `model=general
language=en tier=nova-3`, which a multilingual bundle has no match for: the engine
logs `Could not find a model that matched … language=en`, the API returns 400, and
**every** WS connection closes in ~1 s with `finals=0` / WER 100 %. That looks
like a dead endpoint but is just the wrong param — re-run with `--language
multi` before concluding the listing is broken. (Monolingual listings use the
specific code; Flux multilingual is selected by model name
`--model flux-general-multi`, not a language param.)

## Pass/fail parsing

The final block is always:

```
=====...
PASSED: N  FAILED: N  TOTAL: N
```

Grep for `^PASSED:` to get the counts; nothing else in the output uses that
prefix. The scenario table immediately above is the per-scenario record.

**Judge by the `^PASSED:` line, NEVER by the process exit code.** The runner does
exit non-zero on failure, but if you wrap the run with a trailing command — e.g.
`… 2>&1; echo "EXIT=$?"` — the *wrapper's* (echo's) exit code is what a detached
background process / task notification reports, masking a real failure as
"exit 0". A run can show `PASSED: 0  FAILED: 16` yet notify "exit code 0". If you
must log the code, capture and re-raise it (`rc=$?; echo "EXIT=$rc"; exit $rc`);
otherwise drop the trailing echo and read the tail for `^PASSED:` / `FAILED:`.

## Endpoint deletion ordering

If you orchestrate deploy → e2e → cleanup, the cleanup step must wait for the
e2e process to fully exit before calling `delete-endpoint`. The streaming e2e
keeps making invocations until the very end (`concurrent_10x_15min`,
`adversarial_bare_close`). Killing the endpoint mid-run causes the remaining
scenarios to all 5xx and produces a misleading "PASSED: 0" result.

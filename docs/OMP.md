# Traceability with OMP

The wiring comes from [`@oh-my-pi/pi-ai`](https://www.npmjs.com/package/@oh-my-pi/pi-ai)
(`can1357/oh-my-pi`). This package is a Python port of what OMP does on the wire, and OMP
is the source of truth: when a provider changes, the fix appears there first.

It is split across **three** packages, and anchors may point at any of them:

| Package | What it holds |
|---|---|
| `@oh-my-pi/pi-ai` | the logic — `providers/`, `utils/`, `stream.ts` |
| `@oh-my-pi/pi-catalog` | the wire constants — header values, pinned client versions |
| `@oh-my-pi/pi-utils` | what is shared across every provider — `USER_AGENT`, `VERSION`, paths |

The last two are easy to forget, and forgetting them cost twice:

- `providers/openai-codex-responses.ts` imports `OPENAI_HEADERS`, `OPENAI_HEADER_VALUES`
  and `CODEX_CLIENT_VERSION` from `pi-catalog`. Without it, `originator` ended up as `"pi"`
  instead of `"omp"` and the `version` header did not exist at all — and the audit marked
  them "unverifiable" instead of going to fetch them.
- The same file imports `USER_AGENT` from `pi-utils`. Without it, the value was **invented**
  by analogy (`codex/0.153.0 (external, cli)`) when the real one is `omp/18.2.6`.

The pattern is the same in both cases: search the packages at hand, fail to find, and write
a plausible value instead of searching the one that is missing. A missing package does not
produce a failed anchor — it produces an anchor that is **never written at all**, and so
nothing flags it. `check_omp_drift.py` downloads all three.

This is only useful if, on seeing a change in OMP, you know in ten seconds what to update
here. Hence a single convention.

## The convention

One line per ported function, immediately before the `def`:

```python
# omp: providers/anthropic.ts :: ensureMaxTokensForThinking
def apply_thinking_params(kwargs, model): ...
```

Rules:

- path relative to `src/` in the npm tarball;
- `::` separates file and symbol;
- **no line number** — it changes every release and the symbol does not;
- **one line, one symbol**; several symbols, several lines;
- no prose. The *why* belongs in the docstring.

The version lives in one place only, `OMP_VERSION` in `tools/check_omp_drift.py`.

## Checking

```bash
python tools/check_omp_drift.py
```

Downloads the pinned version and checks two things for every anchor:

- **the name still exists** in the file it points at — a rename goes red;
- **the declaration has not changed.** Each anchored declaration (function, const, object,
  array) is extracted from the TypeScript source, normalized (comments and whitespace
  removed) and hashed; the hash is compared with `tools/omp_anchors.lock`. A different hash
  is reported as `BODY DRIFT`, with every place in this package that ports it.

The name check alone missed three real changes between 18.3.2 and 18.4.1:
`VOLATILE_SYSTEM_SEGMENT_MARKERS` gained `<project-context>`, `applyHeadCaching` started
relocating the system breakpoint, and the 64k OAuth output clamp was removed — each one
kept its name, and each one mattered on the wire.

It also compares against npm's `latest` and warns when a newer version exists. It runs in
CI.

### Raising the OMP version

1. Raise `OMP_VERSION` and run the script. `BODY DRIFT` lists what changed upstream in the
   code we port; a missing symbol lists what moved.
2. For each item, read the upstream diff and port it (or record a deliberate divergence
   below). Don't regenerate the lock first: the drift list is the work list.
3. Only then run `python tools/check_omp_drift.py --update`, which rewrites the lock for the
   pinned version, and commit it with the ported changes.

Adding an anchor also needs `--update`, so its declaration enters the lock.

## Deliberate divergences

Where we do not follow OMP, and why. Each one was measured against the real service.

| Where | OMP | Here | Why |
|---|---|---|---|
| Anthropic betas | includes `redact-thinking-2026-02-12` (`usage/claude.ts`) | omitted | With it, Anthropic returns signed but empty thinking blocks: measured on sonnet-4-6, 74 chars without the beta, 0 with it. |
| Anthropic betas | includes `context-1m-2025-08-07` | omitted | Returns a credit 429 on subscription tokens. |
| Thinking budget | up to 32768 | ceiling 8192 | Short TPM window on the Max subscription; 32768 gives a 429. |
| Reasoning loop | *retryable* error, the retry layer asks again | raises | There is no replay-safe window here: `reasoning_content` has already been flushed to the client before detection, and retrying duplicated it in the same stream. |
| `-thinking` variants | strippable | only `gemini-2.5-flash-thinking` | `gemini-3.7/3.8-flash-thinking` do not exist upstream; stripping them silently served `-low` for an invented name. |
| Unserved name | falls back to a nearby model | raises | Answering with a different model makes billing and comparisons lie, and the client never knows. |
| Background renewal | one broker refreshes the store; each sweep renewal is forced and waits for the refresh lease (`auth-broker/refresher.ts`) | every worker sweeps; freshness is re-checked inside the lock and a busy lock is skipped | LiteLLM runs `--num_workers` processes, each with its own sweep: forcing would rotate the token once per worker per cycle. Measured with four workers (DECISIONS D8): one lock file, one new token, no `invalid_grant`. |

## Divergences corrected by comparing against the source

These were not decisions: they were errors from having been ported from an intermediate
copy instead of the source. They are recorded because the failure mode is instructive.

| Where | Was | Corrected to | How it was noticed |
|---|---|---|---|
| `google_finish_reason` | enumerated the **error** reasons | enumerates the **normal** ones (`STOP`, `MAX_TOKENS`) and treats the rest as an error, like `mapStopReasonString` | Five reasons (`FINISH_REASON_UNSPECIFIED`, `LANGUAGE`, `IMAGE_OTHER`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION`) passed as `stop`: a server-blocked response reached the client as if it were complete. |
| `thinking_loop` | **character** trigrams, cluster of 2, no warm-up | **word** trigrams, `SEGMENT_MIN_CLUSTER=4`, `SEGMENT_MIN_COUNT=8`, two exact-cycle regimes, canonicalized anchors | Character trigrams give high similarity to unrelated texts; firing at 2 segments killed legitimate reasoning that OMP lets through. |

**Method lesson:** port from the source and verify against the intermediate — never the
reverse. A second-hand copy inherits the first one's errors without flagging them.

A divergence without a measurement is not a divergence: it is an unfixed bug.

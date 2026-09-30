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
| Codex whitespace loop, streamed | replays the turn (up to 2 times) while nothing visible was delivered | a streamed turn raises 502; a non-streamed one is replayed as in omp | omp drops the half-built call from its own event stream before replaying; a chat, Messages or Responses client has already received the call's opening chunk when the brake trips, and a replay would open a second call it cannot tell from the first. |
| Loop guard on visible text | `guardThinkingLoopStream` also feeds the answer text to an exact-cycle detector (`checkAssistantContent`, on by default; the gateway never turns it off) | only reasoning is judged | Measured live on the 0.1.17 candidate, "Write the line 'hello world' 60 times, one per line, nothing else.", non-streamed chat: gpt-5.5 and gemini-3-flash both failed 502 "repeated an exact 12-character cycle 21x/32x back-to-back" after three attempts; 0.1.16 answered 200 with the 60 lines. Lists, tables, CSV and code repeat by design. |
| Cloud Code turn cut by `MAX_TOKENS` with no text | fails "returned a thought-only response without final output" | `finish_reason: length` with what arrived | Measured live on gemini-3-flash, "Write a 600-word essay about bridges." with `max_tokens=64`: the candidate answered 502, 0.1.16 answered 200 `length`. The client asked for the cut; a STOP with nothing in it still fails as omp's does. |
| `-thinking` variants | strippable | only `gemini-2.5-flash-thinking` | `gemini-3.7/3.8-flash-thinking` do not exist upstream; stripping them silently served `-low` for an invented name. |
| Unserved name | falls back to a nearby model | raises | Answering with a different model makes billing and comparisons lie, and the client never knows. |
| Codex tool without parameters | `{}` normalized to `parameters: true` (`sanitizeSchemaForOpenAIResponses`) | `{"type": "object", "properties": {}}` | Measured on the live Codex backend on 2026-09-30: `true` returns 400 `invalid_type` ("Invalid type for 'tools[0].parameters': expected an object, but got a boolean instead"); the empty object schema returns 200. |
| Antigravity 429 | transient: moves to the other host, which retries it (`streamGoogleGeminiCli`) | stays on the host that answered, which gets the last host's in-place retries | Both hosts front the same account and quota: measured, the rotation turned an ~11 s failure into ~22 s with the same verdict. |
| Antigravity output ceiling | the wire profile's fixed `maxOutputTokens` overwrites the caller's (`preserves-max-output-tokens #false`); elsewhere the thinking budget is added on top of `max_tokens` (`maxTokensWithThinkingBudget`) | the caller's `max_tokens` is the total, lowered only to the catalog's declared ceiling | The budget bounds thinking, not the answer: live 2026-09-30, claude-sonnet-4-6 asked for 64 tokens with 1024 on top went out as 1088 and answered 623 words (844 tokens) `STOP`. With 64 as the total, 9 models x 3 efforts all ended `MAX_TOKENS` within it. |
| Antigravity budget under a small ceiling | `ceiling - 1024`, else thinking off (budget 0) | Claude: the same (its minimum is 1024); everything else keeps thinking at the catalog's `minThinkingBudget`, or its own budget without one | Live 2026-09-30, `maxOutputTokens: 64`: budget 0 is refused by gemini-3.1-pro-low and gemini-pro-agent ("Budget 0 is invalid. This model only works in thinking mode."), gemini-2.5-flash, gemini-2.5-flash-lite, gemini-3.5-flash-lite and gpt-oss-120b-medium ("invalid argument"); a budget above the ceiling is accepted by all of them (the ceiling bounds thoughts and answer together); on the flash ids 32 left ~50 words of answer where 4000 left none. |
| Antigravity `reasoning_effort: "none"` | thinking off: `{budget: 0}` or `{level: MINIMAL}` where `thinking-suppress-when-off`, else raised to the lowest effort on thinking-mandatory models | budget 0 only where it is accepted (Claude, catalog `minThinkingBudget` <= 32); elsewhere thinking at the catalog minimum (or its own budget without one) | Live 2026-09-30: budget 0 is refused by gemini-3.1-pro-low, gemini-pro-agent, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-3.5-flash-lite (minimum 128) and gpt-oss-120b-medium (no minimum); each answered 200 at 128 (gpt-oss at 8192). omp would send budget 0 to 3.1-pro (`thinking-suppress-when-off`). |
| Antigravity `thinkingConfig` | omitted when no reasoning is asked for (unless `thinking-suppress-when-off`) | always sent | Omitted, the CCA reapplies the per-id server default and bills thinking tokens without returning the text. |
| Antigravity tool-result images | a following user turn (chat gateway; Gemini < 3) | inside `functionResponse.parts` for every model | Measured: every generation the account serves (gemini-3.8-flash, 3.1-pro, 3.1-flash-lite, 2.5-flash, 2.5-flash-lite, pro-agent) described a blue screenshot returned that way. |
| Antigravity media by URL | `[image: <url>]` placeholder text (chat gateway, no fetcher) | fetched and inlined | `fileData` with a web URL answers 404 "Requested entity was not found"; a placeholder means the model never sees the image. |
| Antigravity sampling | sends `temperature`, `topP`, `topK`, `presencePenalty` as given | drops `presencePenalty` on Gemini, and `topP < 0.95` on a thinking Claude | Measured 2026-09-30, one field at a time: every Gemini (3-flash, 3.1-pro-low, 3.8-flash-medium, 2.5-flash) answered a penalty with 400 "Penalty is not enabled for this model"; claude-sonnet-4-6 thinking answered `topP 0.9` with 400 (`top_p` must be >= 0.95 or unset); everything else, and all five fields on gpt-oss-120b-medium, answered 200. |
| Antigravity Claude tool without parameters | `parameters: {}` (`buildTools`) | `{"type": "object", "properties": {}}` | Measured 2026-09-30 on claude-sonnet-4-6: `{}` answered 400 "tools.0.custom.input_schema.type: Field required", the object schema 200. Gemini takes `{}` and keeps it. |
| Background renewal | one broker refreshes the store; each sweep renewal is forced and waits for the refresh lease (`auth-broker/refresher.ts`) | every worker sweeps; freshness is re-checked inside the lock and a busy lock is skipped | LiteLLM runs `--num_workers` processes, each with its own sweep: forcing would rotate the token once per worker per cycle. Measured with four workers (DECISIONS D8): one lock file, one new token, no `invalid_grant`. |

## Divergences corrected by comparing against the source

These were not decisions: they were errors from having been ported from an intermediate
copy instead of the source. They are recorded because the failure mode is instructive.

| Where | Was | Corrected to | How it was noticed |
|---|---|---|---|
| `google_finish_reason` (now `_AntigravityReader._finish`) | enumerated the **error** reasons | enumerates the **normal** ones (`STOP`, `MAX_TOKENS`) and treats the rest as an error, like `mapStopReasonString` | Five reasons (`FINISH_REASON_UNSPECIFIED`, `LANGUAGE`, `IMAGE_OTHER`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION`) passed as `stop`: a server-blocked response reached the client as if it were complete. |
| `thinking_loop` | **character** trigrams, cluster of 2, no warm-up | **word** trigrams, `SEGMENT_MIN_CLUSTER=4`, `SEGMENT_MIN_COUNT=8`, two exact-cycle regimes, canonicalized anchors | Character trigrams give high similarity to unrelated texts; firing at 2 segments killed legitimate reasoning that OMP lets through. |
| Reasoning loop | raised on every path, called a deliberate divergence | re-samples a non-streamed turn up to 3 attempts, fails a streamed one | omp's gateway answers a non-streamed request through `completeSimple`, which re-samples (`resolveWithThinkingLoopRetries`); only its streamed path fails, since the reasoning already left. The divergence was real for the stream only. |

**Method lesson:** port from the source and verify against the intermediate — never the
reverse. A second-hand copy inherits the first one's errors without flagging them.

A divergence without a measurement is not a divergence: it is an unfixed bug.

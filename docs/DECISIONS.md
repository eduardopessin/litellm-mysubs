# Architecture decisions

Each entry records what was decided, what was measured, and what would reopen the question.
Without the measurement it is not a decision — it is a preference.

---

## D1 — Streaming stays in a monkey-patch, not in `CustomLLM`

**Date:** 2026-09-18 · **Against:** `litellm[proxy]` 1.101.0 · **Status:** decided

### The question

LiteLLM has an official path for adding a provider — `custom_provider_map` with a
`CustomLLM` class (`litellm/utils.py :: custom_llm_setup`). Using it would remove the
dependency on internal symbols (`Router.acompletion`,
`route_llm_request.route_request`), which can change between releases without notice.

### What was measured

Minimal handler, no upstream, returning `prompt_tokens=100, completion_tokens=5,
cached_tokens=80`:

| How the final chunk is emitted | Usage delivered |
|---|---|
| `GenericStreamingChunk` (the type the signature declares) | has no reasoning field at all |
| `setattr(chunk, "usage", u)` | **8 / 2** |
| `usage=` as a model field | **8 / 2** |
| final chunk with `choices=[]` | **8 / 2** |
| raw dict | `MidStreamFallbackError` |

In all of them: `cached_tokens` lost.

Instrumenting `CustomStreamWrapper.chunk_creator`::

    [chunk_creator] in.usage=100 -> out.usage=None
    chunks accumulated in the wrapper: 1
    delivered to the client: 8

The branch that would preserve the value exists in
`litellm_core_utils/streaming_handler.py`::

    if hasattr(chunk, "usage") and chunk.usage is not None:
        model_response.usage = chunk.usage

…but it is not reached from a custom handler. The `calculate_total_usage` that decides the
final value reads `if "usage" in chunk` — access **by key**, over an object that has
already lost the field. Also verified that `_hidden_params["usage"]` is `None` on this
path, contrary to what issue [#12233] suggests.

Related and still open upstream: [#12970], "Streaming response reports incorrect
prompt_tokens usage — significantly inflated".

### Why this is blocking

Without the real usage, LiteLLM estimates with `token_counter` and **every cache hit
becomes invisible** in `/spend/logs`. On a subscription account the cache is the difference
between 8697 and 2876 prompt tokens on the same request: losing that accounting is losing
the only way to know why the quota ran out.

### Decision

The streaming generators stay in a monkey-patch. It is the only path that controls the
usage that reaches spend logging.

### What the spike gave in return

- **`/v1/responses` reaches the handler without a patch** in `route_request`. One of the
  three current patches may disappear in a future migration.
- **Any name under the prefix reaches the handler**, which decides — confirms the design of
  the name guard already implemented.
- **`reasoning_content` survives** when a raw `ModelResponseStream` is emitted.

### Consequence

The LiteLLM version matrix in CI and `tests/test_litellm_contract.py` stop being a
precaution and become mandatory: the dependency on internals is permanent, and that test is
what turns an incompatible upgrade into a red CI run instead of a production outage.

### What would reopen this

LiteLLM preserving the usage reported by a `CustomLLM` — either by closing [#12970] or by
exposing a dedicated field for it. Re-check with `python tools/spike_custom_llm.py` when
the pinned version moves up.

[#12233]: https://github.com/BerriAI/litellm/issues/12233
[#12970]: https://github.com/BerriAI/litellm/issues/12970

---

## D2 — Models declared in `config.yaml`, wildcard as the safety net

**Date:** 2026-09-18 · **Status:** decided

### The question

If the `claude-*` wildcard serves the whole family, why keep explicit entries?

### What was measured

- The aliases are not derivable: `claude-opus` → `claude-opus-4-8`,
  `claude-3-5-sonnet` → `claude-sonnet-4-6`. The wildcard would send `anthropic/claude-opus`
  upstream, which answers 404.
- `model_info` disappears on wildcard-materialized entries (`id`, `max_input_tokens`,
  `supports_vision`) — that was the difference that made shadowing in `simple-shuffle`
  dangerous.
- The wildcard does not enumerate: with no declared entries, `/v1/models` shows only the
  literal patterns, and clients that discover models there get nothing.

### Decision

Config for stable names, aliases and metadata. Wildcard as a safety net for whatever has
not been declared yet — that is what makes it possible to serve a new model of the family
on day one.

---

## D3 — Injection into the Router, not `POST /model/new`

**Date:** 2026-09-18 · **Status:** decided

### What was measured

    POST /model/new {"model_name": "zz-probe", ...}   -> 200, returns model_id
    /model/info                                        -> does not appear
    /v1/models                                         -> does not appear
    POST /v1/chat/completions model=zz-probe           -> 400 no healthy deployments

The model is written to Postgres and never reaches the Router when
`general_settings.supported_db_objects` does not include `"models"` — and not including it
is the correct configuration on installations that serve A2A agents, where the wrong list
makes `_should_load_db_object` return `False` for everything.

### Decision

The plugin injects via `Router.set_model_list()` and persists on its own. An "apply" that
returns 200 and does nothing is the worst possible failure mode.

## D4 — Intentional divergences from `sitecustomize.py`

**Date:** 2026-09-18 · **Status:** decided

`tools/check_equivalence.py` was born to prove that the extraction changed nothing of what
goes on the wire. It delivered: while it was refactor only, the 20 samples matched byte for
byte.

That stopped being true when the port started correcting the original against the OMP
source. A checker that fails on 19 out of 20 cases cannot tell a regression from a fix, and
in that state it is noise with authority.

The checkers were removed before publication: they need the original `sitecustomize.py`,
which lives in another repository and is not a dependency of this one, and by the end they
reported whole families of functions as "missing from the original" — the port had moved
far enough that there was nothing left to compare. The divergences below are the record
they leave behind; any behaviour outside this list is a regression.

### The system prompt (19 cases)

The original sent `You are a Claude agent, built on Anthropic's Claude Agent SDK.`; the
port sends `You are Claude Code, Anthropic's official CLI for Claude.` — the one the real
CLI sends, which is what the `User-Agent` and the beta list claim to be. Two identities in
the same request is what makes Anthropic treat it as non-CLI traffic.

The cache anchor in the system block comes from the same place: OMP pins the
`tools`+`system` prefix, which does not change between turns, instead of rewriting the tail
anchor on every request.

### `max_tokens` (6 cases)

The original always pinned 16384. The port derives `budget + OUTPUT_FALLBACK_BUFFER`,
capped by `MAX_OUTPUT_TOKENS`, which is the source's rule
(`providers/anthropic.ts:3868`). Pinning the value gave 16384 to a `minimal` request — 12
thousand output tokens reserved and unused — and the same 16384 to an `xhigh` one, which
is left without room to answer after thinking.

### `budget_tokens` (2 cases)

OMP's scale (`low` = 4096) instead of the original's table (`low` = 2048), with the
`THINKING_CEILING` cap applied **after** the step, to preserve the relative order.

### `output_config.effort` (1 case)

With `tool_choice` forced on an adaptive model, the port pins `effort: "low"`. Omitting
`thinking` does not turn reasoning off on these models — the API turns it back on by
default — and pinning the lowest step is the only way to reduce it without the
`tool_choice` + `thinking` 400.

### `thinking.display`

Restored after the checker caught it missing. The gate is generational — opus ≥ 4.7,
sonnet/fable/mythos ≥ 5 (`compat/resolve.ts :: defaultSupportsDisplay`) — and does not
coincide with `is_adaptive`: opus-4-6 and sonnet-4-6 are adaptive and reject the field with
a 400.

## D5 — How the "MySubs" button gets into the LiteLLM UI

**Date:** 2026-09-18 · **Status:** decided

### What was measured

The proxy UI is **pre-compiled** Next.js, served from
`litellm/proxy/_experimental/out/`. The side menu is React code inside a minified chunk:

    out/_next/static/chunks/0c63y7umyjwi-.js
    …{key:"experimental",page:"experimental",label:"Experimental",
       icon:(0,a.jsx)($.FlaskConical,{…}),children:[{key:"prompts",…}]}

Two facts that decide it:

1. The `experimental` item has **`children`** — adding an entry means adding an element to
   that list.
2. Other entries use **`external_url`** (`learning-resources` points at
   `models.litellm.ai/cookbook`). There is precedent for a menu item that leaves the SPA.

The chunk name is a build hash: it changes with every LiteLLM version.

### Decision

**A FastAPI sub-app mounted at `/mysubs` with `app.mount()`**, serving its own UI. Access is
by direct URL and by an injected menu item.

**The menu item injection is optional and best-effort.** A string patch into a minified
bundle whose name is a build hash breaks silently on the next LiteLLM version — and a
button that disappears without warning is worse than a button that never existed.
Therefore:

- `/mysubs` always works, by URL, without depending on any patch.
- The injection looks for the pattern; **if it does not find it, it does not fail** — it
  records that this version's UI was not recognized and tells the user the direct URL.
- The file in `site-packages` is never rewritten: a modified copy is served from memory, so
  that a `pip install --force-reinstall` does not leave inconsistent state.

### Rejected alternative

Recompiling the LiteLLM UI with the entry included. It gives a native button, but forces
tracking every upstream release with a frontend fork — a permanent cost for a cosmetic
gain.

---

## D6 — The token interceptor runs on the user's machine, not on the proxy

**Date:** 2026-09-19 · **Against:** `@oh-my-pi/pi-ai@18.2.6` · **Status:** decided

### The question

Paste works, but it asks the user to copy a code from one page to another. Quota Desktop
asks for none of that: it opens the loopback port, the browser is redirected there, and the
connection is made. The question was whether the plugin could not do the same — open the
port itself and make the flow transparent.

### What decides it

It can, but **not from the LiteLLM process**. The registered redirect is
`http://localhost:54545/callback`, and `localhost` resolves in the user's browser. A port
opened inside the pod listens on a different machine: the browser hits the workstation's
loopback and finds nothing. It is exactly why Quota Desktop exists as a native application
instead of a dashboard page.

They only coincide when the proxy runs on the same machine as the browser — `pip install
litellm` on a laptop. On a container or cluster installation, never.

### Decision

A separate command, `mysubs-login`, installed by the same wheel and run by the user on
their own machine. It opens the port, catches the code, exchanges it for a credential and
deposits it on the proxy — local or remote. The page issues a pairing code and shows the
command already assembled with the right URL.

**Paste remains the path that survives everything** and was not touched: port taken, SSH
with no tunnel, machine with no browser. The interceptor is the transparent path for whoever
has a browser on the same machine, not a replacement for the other one.

### Constraints measured in the OMP source

From `src/registry/oauth/callback-server.ts` and `src/compat/rules/auth/*.kdl`:

| Provider | Port | Path | Port fallback |
|---|---|---|---|
| `anthropic` | 54545 | `/callback` | allowed |
| `openai-codex` | 1455 | `/auth/callback` | **`port-fallback=#false`** |
| `google-antigravity` | 51121 | `/oauth-callback` (host `127.0.0.1`) | allowed |

- **Dual-stack bind is mandatory.** `localhost` resolves to `127.0.0.1` *and* `::1`, and
  clients — Windows in particular — try `::1` first. Binding only the IPv4 literal hands
  the authorization code to whoever holds the IPv6 loopback on the same port. With
  `IPV6_V6ONLY=1`, otherwise the IPv6 socket claims the IPv4 mapping and the second bind
  fails against ourselves.
- **A missing IPv6 is tolerated**, as long as one family remains: a kernel with
  `ipv6.disable=1` must not be able to kill the flow (OMP issue #8814). `EADDRINUSE` on any
  family, on the other hand, aborts everything — half the loopback handed to someone else
  is the same risk.
- **Exact port, never a fallback.** The provider validates the registered redirect URI; a
  random port produces an opaque refusal after the user has already logged in.

### Why the pairing code, and not the admin key

The deposit is the only route outside the `proxy_admin` guard. The alternative was for the
command to carry the key that administers the whole proxy on a command line — to save one
route. The code lives ten minutes, is used once, and is valid for a single provider; it is
the code that decides which provider the credential belongs to, not the request body.
Letting the client name it would turn a code issued to connect Codex into an overwrite of
the Anthropic credential.

The code never travels in the query string: the route returns the page instead of
redirecting, because a code in a URL ends up in the browser history, in the logs of any
proxy in front, and in the `Referer` of the next request.

### What would reopen this

A provider starting to accept a redirect URI that is not loopback — then the callback could
go straight back to the proxy and the command would no longer be needed.

---

## D7 — The deposit rate limiter is per worker

**Date:** 2026-09-19 · **Against:** `litellm[proxy]` 1.101.0, `--num_workers 4` · **Status:** decided

### The question

`POST /mysubs/api/deposit` is the only route outside the `proxy_admin` guard — the pairing
code is the authorization, because the alternative was putting the proxy key on a command
line. It lacked an attempt limit and a record of failures.

### What was measured

With the in-memory limiter and `--num_workers 4`:

    configured: 10 failures per origin
    measured:   22 failures before the first 429

gunicorn distributes the requests and each process counts on its own. The effective ceiling
is `max_failures × workers`.

### Decision

Accepted. Shared state in a file — like the refresher's `flock` — would add I/O on every
request path and a new failure mode, for a gain the arithmetic denies:

| workers | attempts per TTL | probability of a hit |
|---|---|---|
| 1 | 100 | 1 in 11 500 trillion |
| 4 | 400 | 1 in 2 880 trillion |
| 16 | 1600 | 1 in 720 trillion |

A 2^60 space (12 symbols from a 32-symbol alphabet), a 10-minute TTL, single use.

**Entropy is the defense; the limiter is there to stop hammering and leave a trail.**
Without it there was no signal of an attack in the logs at all — that was the real gap.

### What would reopen this

Lowering the code's entropy, widening the TTL, or allowing several valid codes at once for
the same provider. Any of those changes the table above.

---

## D8 — Multi-worker: the refresher lock is enough

**Date:** 2026-09-19 · **Against:** `litellm[proxy]` 1.101.0, `--num_workers 4` · **Status:** verified

### What was measured

Four real workers (confirmed in `ss -tlnp`: four PIDs on the same port), Antigravity token
forced to expire in 60s — inside the 5 min skew that triggers the refresh:

    t+45s  locks=0   0.3min ...ytNQ0211
    t+50s  locks=1  54.9min ...NjaQ0211    <- refreshed
    t+90s  locks=1  54.2min ...NjaQ0211    <- stable

A single lock file, a single new token, refresh token rotated, and an inference request
right after answered. **No `invalid_grant`** — which is the failure mode that `flock` and
the single-owner rule exist to prevent.

### Why this matters

Single-use rotating tokens do not tolerate two refreshers: one's copy invalidates the
other's and the account enters an `invalid_grant` cycle until someone logs in by hand. The
design was unit-tested; what was missing was seeing it with real processes.

---

## D9 — The whole repository is in English (reverses the earlier split)

**Date:** 2026-09-19 · **Against:** `litellm[proxy]` 1.101.0 · **Status:** decided (reversal)

### The question

The package goes to PyPI. Translate everything to English, nothing, or part of it?

### What was measured

The LiteLLM UI is monolingual: `<html lang="en">`, zero translation files, zero language
selector. The `"pt"`/`"fr"` occurrences in the bundle are CSS units (`pt`, `pc`) and
Tailwind classes (`pt-4`, `pr-2`), not languages — verified with a context `grep`.

There is therefore no i18n to plug into. Building a translation system to sit next to a UI
that has none would be infrastructure without a counterpart.

Volume, counted by AST over every string constant in `src/`:

| | Volume |
|---|---|
| Visible strings (UI, CLI, errors, logs) | ~172 |
| Comments and docstrings | ~2094 lines, 15% of the code |

### What was decided first, and why

The first decision split the repository: visible strings in English, comments and
docstrings in Portuguese. The reasoning was the risk in the second column of that table.
These are not ordinary comments — each one carries a measurement (*"Measured with
`--num_workers 4`: 22 failures before the first 429"*). Translating 2094 lines of them is
an operation in which every error erases a piece of evidence and no test catches it. It is
the repo's most fragile asset, and leaving it in the language it was measured in was the
cheapest way to not damage it.

### What changed

The decision to publish as a public, international repository. That moves the cost to the
other side: for anyone evaluating the project, reading the code, or sending a patch,
Portuguese comments are not a preserved asset — they are a wall in front of the only thing
that explains why the code is the way it is. A measurement nobody can read defends nothing.

The translation risk did not disappear; it was paid, with the rule that the measurement
comes first: number, endpoint, status code, observed behavior copied exactly, and anything
not understood reported instead of guessed.

### The decision

**Everything in English** — code, comments, docstrings, documentation, test names,
commit messages. No Portuguese remains in what is published.

### How regression is prevented

`tests/test_ui_interceptor.py::TestThePageIsInEnglish` serves four page states (connect,
pair, connected with discovery, refusal), strips `<script>` and `<style>`, and looks for
Portuguese markers on word boundaries. Proven to fail: putting the button back as
`Ligar a subscrição` brought down 3 of the 4 cases.

That test guards the visible strings, which is where a regression is invisible to the
author and immediate for the user. The comments have no equivalent guard: what protects
them is review.

### What would reopen this

LiteLLM gaining real i18n. Then the question changes: it stops being "which language" and
becomes "how does the plugin declare its translation keys".

---

## D10 — Anchors now pin the value, not just the name

**Date:** 2026-09-19 · **Against:** `@oh-my-pi/pi-ai` 18.2.6 · **Status:** decided

### The question

The wiring is OMP's. The `# omp: file.ts :: symbol` anchors and
`tools/check_omp_drift.py` exist so that a rename on their side does not go undiscovered.
The question: is that enough, when upstream touches the grammar?

### What was measured

Two things, both against the real package.

**1. One anchor was already broken and nobody had noticed.**
`usage.py:490` pointed at `fetchAvailableModels fallback` — two words, a description, not a
symbol. The checker reported it, CI runs it, and the job was red without that having
reached anyone. The real symbol is `FETCH_AVAILABLE_MODELS_PATH`. Fixed: 177 green anchors.

**2. The failure mode that matters was not covered at all.**
Simulation over the real tarball: changing `claudeCodeSdkVersion` from `0.112.1` to
`0.999.0` leaves the 177 anchors **green**. The name survives; the value is what goes on the
wire.

This is not hypothetical — it is exactly how `wham/usage` came to be written as
`codex/wham/usage`: the symbol existed in both versions, and only the real call told them
apart, with a 403.

Of the 177 anchors, 21 pointed at constants whose value is a wire literal: endpoint paths,
client ids, pinned client versions, token URLs.

### The decision

A second anchor form, `# omp=`, which also pins the literal:

```python
# omp: usage/openai-codex.ts :: CODEX_USAGE_PATH
# omp= CODEX_USAGE_PATH = "wham/usage"
```

The file is inherited from the name anchor above, and the inheritance **stops** at the first
line that is not an anchor — without that, a stray `# omp=` in the middle would inherit the
path of another context and check the wrong constant.

Nine values pinned: `claudeCodeSdkVersion`, `FREE_TIER_ID`, the Codex `CLIENT_ID`,
`OAUTH_TOKEN_URL`, Anthropic's `DEFAULT_ENDPOINT`, `CODEX_USAGE_PATH`, and the three
Antigravity `v1internal` paths.

A composed value (`` `${BASE}/v1:load` ``) returns empty instead of faking a comparison: the
value anchor does not serve those, and saying so is better than an invented green.

### The proof

Simulation of three changes that used to go unnoticed, all caught now:

| upstream change | verdict |
|---|---|
| `0.112.1` → `0.999.0` | `oauth.py:83 value changed` |
| `wham/usage` → `wham/v2/usage` | `usage_probe.py:40 value changed` |
| `app_EMoam…` → `app_NOVO` | `oauth.py:142 value changed` |

`tests/test_omp_drift.py` — 20 tests, no network. Proven by mutation: restoring the old
checker (ignoring the value) brings down 6; never cutting the inheritance brings down 1;
letting a prefix match as a symbol brings down 1.

### What is left uncovered, and it is honest to say so

The anchor proves the symbol exists and that the literal is the expected one. It does **not**
prove that the logic around it is still the same. If OMP keeps
`CODEX_USAGE_PATH = "wham/usage"` but changes the shape of the response body, nothing here
gives a signal — only a real request does, and that is what `usage_probe` is for.

### What would reopen this

OMP starting to publish the wire values in a data file (JSON, KDL) instead of TypeScript
constants. Then the check stops being textual and becomes a structure comparison, which is
stronger.

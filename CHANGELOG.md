# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2026-09-21

### Added

- **`/v1/responses` now serves streaming for Codex models.** 0.1.3 and 0.1.4 fixed the
  non-streaming path; a streamed request still fell through to LiteLLM's native OpenAI
  client and came back `Incorrect API key provided: None`. Agent clients stream by
  default, so the route stayed unusable for them.

  The event sequence was **captured from this proxy's own native `/v1/responses` path**
  rather than assumed — a model served by vLLM, `stream: true`, recorded off the wire:

  ```
  response.created -> response.in_progress
  -> output_item.added -> content_part.added
  -> output_text.delta* -> output_text.done -> content_part.done
  -> output_item.done
  -> response.completed
  ```

  Reasoning and tool calls are emitted as completed items once the turn closes, because
  they are only known then; text streams as it arrives.

  Two defects found by testing against a real proxy before release, not after:

  - **Events must be LiteLLM's typed models, not dicts.** The proxy serialises a chunk
    with `_serialize_streaming_chunk`, which calls `.model_dump_json()`. A plain dict
    falls through to `str()` and reaches the client as a Python repr with single quotes,
    which no JSON parser accepts.
  - **`ContentPartDoneEvent` requires `logprobs` on the part.** Omitting it raised mid
    stream and truncated the response after the deltas, with no terminal event.

### Fixed

- **Every subscription model showed the generic icon in the UI.** Deployments did not
  declare `custom_llm_provider`, so the UI fell back to inferring the provider from
  `model_name` — and `get_llm_provider` **raises** on the public names this package
  builds. Measured: `BadRequestError` for `mysubs/codex/gpt-5.5`,
  `mysubs/claudecode/claude-opus-5` and `mysubs/antigravity/gemini-3-flash`. With no
  provider resolved, OpenAI and Google rows were indistinguishable in the Logs tab.

  The provider is now declared, using the same family prefix that already goes on the
  wire:

  | public name | wire | provider |
  |---|---|---|
  | `mysubs/codex/gpt-5.5` | `openai/gpt-5.5` | `openai` |
  | `mysubs/claudecode/claude-opus-5` | `anthropic/claude-opus-5` | `anthropic` |
  | `mysubs/antigravity/gemini-3-flash` | `gemini/gemini-3-flash` | `gemini` |

  Tying it to the wire prefix is deliberate: that prefix is the one with a price table
  behind it, so the icon and the cost cannot disagree. A test asserts they stay equal.

### Known issues

- **Antigravity models return `429 RESOURCE_EXHAUSTED`.** Pre-existing, on the chat path,
  and untouched by this release. Measured with the same payload replayed through `curl`,
  so it is not client-specific: deterministic on system-message content, not on size
  (122 KB of filler passes; 1 KB of a real agent prompt fails) and not rate limiting
  (81 characters passes five times in a row, 82 fails three times with pauses between).
  One byte flips it at identical length. The upstream returns a quota error for something
  that is not a quota; this package relays it unchanged.

## [0.1.4] - 2026-09-21

### Fixed

- **`/v1/responses` answered with an empty `output` for Codex models.** 0.1.3 routed the
  request correctly but returned the terminal event's `response` object unchanged, on the
  assumption that it carried the output items. It does not: the items arrive in the
  `response.output_item.done` events during the stream, and `response.completed` closes the
  turn without repeating them.

  The result was a `status: completed` turn with billed output tokens and nothing in it.
  Measured on a live gateway, same prompt, same model:

  | route | answer | `output_tokens` |
  |---|---|---|
  | `/v1/chat/completions` | `4` | 17 |
  | `/v1/responses` (0.1.3) | *(empty)* | 17 |

  Worse than the 401 it replaced: a client cannot tell that from a model that chose to say
  nothing.

  The items are now rebuilt from what the stream reader accumulated — the same source the
  chat path uses, so the two routes cannot disagree — as `reasoning`, `message` and
  `function_call` items in that order, with the composite call id preserved so a follow-up
  turn matches its output to the right call. An upstream that does send `output` keeps it;
  rebuilding is the fallback, not the rule.

  The 0.1.3 test suite missed this because its fixture invented an `output` the real
  endpoint never sends. The fixture now matches the wire, and defaults to omitting it.

## [0.1.3] - 2026-09-21

### Fixed

- **`/v1/responses` returned `Incorrect API key provided: None` for Codex models.** The
  plugin patched `Router.acompletion`, which is the path `/v1/chat/completions` takes. The
  Responses route goes through `Router.aresponses`, which was never intercepted, so the
  request reached LiteLLM's native OpenAI client with no `api_key` — the credential is
  OAuth and lives in the store, not in `config.yaml` — and OpenAI refused it.

  Any client that speaks the Responses API was affected, which includes agent clients that
  route `gpt-*` models there by default. Reproduced network-free against litellm 1.101.0:

  | route | plugin consulted | result |
  |---|---|---|
  | `Router.acompletion` | 1x | served |
  | `Router.aresponses` | **0x** | `AuthenticationError: api_key: None` |

  Two things made this harder to fix than to find. `Router.aresponses` is **not** a class
  method: `Router.__init__` builds it per instance with
  `self.aresponses = self.factory_function(litellm.aresponses, ...)`, capturing the module
  function by value. Patching the class is a no-op, and patching `litellm.aresponses`
  afterwards is too late — the factory already holds the old reference. Measured, both
  orderings:

  | patch timing | effect |
  |---|---|
  | before `Router()` | applies |
  | after `Router()` | **no effect** |

  And after is always: the proxy builds its Router before constructing the `CustomLogger`
  that loads this package. So the bound attribute is replaced on the live instance instead,
  by `bind_responses_route`, at the same moment the UI already waits for `llm_router`.

  Codex is served from its own Responses payload rather than a converted chat completion:
  its endpoint **is** a Responses API, so the terminal event already carries the object the
  route has to return, and rebuilding it would lose `output` item structure, reasoning
  items and call ids. Only `id`, `model` and `usage` are normalised.

  Not served here, deliberately: Anthropic, which LiteLLM's native path already answers
  correctly on this route; Antigravity, which is Gemini-shaped and would need item
  structure invented for it; and streaming, whose Responses SSE protocol is its own event
  sequence rather than the chat chunks this plugin emits. All three fall through to the
  original.

  On the Router the deployment mark decides, and the name heuristic is not consulted:
  `codex.is_codex_model` matches any name containing `gpt-`, so an operator's own `gpt-4o`
  deployment would otherwise have been answered from this subscription. Caught by a test,
  not in production.

  Verified end to end over HTTP against a real proxy with the real route, plus 12 new
  regression tests.

## [0.1.2] - 2026-09-20

### Fixed

- **Streamed calls were logged with zero cost.** `_wrap_stream` handed
  `CustomStreamWrapper` the public model name and `custom_llm_provider="custom_openai"`.
  Neither has a price table, so `response_cost_calculator` returned `0.0` for every
  streamed request, while the non-streaming path — which gets the wire pair from the
  Router — priced the same usage correctly.

  Measured on a live proxy, controlled pair, same prompt, same cache, same key, same
  model, only `stream: true` differing:

  | | prompt | cache_read | spend |
  |---|---|---|---|
  | non-streaming | 40049 | 40035 | `0.0202475` |
  | streaming | 40054 | 40035 | `0` |

  On the installation where this was found, **74 of 89 billable calls (83%)** were
  recorded as free, including 3.8M prompt tokens of `claude-opus-5`. Since agent clients
  stream by default, this was most of the traffic.

  The wire name is read off the deployment in `_wrapped_router_acompletion`, where the
  Router still has it, and travels to the streaming path under a private kwarg popped
  before any delegation upstream.

  Two things this does not fix, both upstream rather than dispatch. Antigravity models
  absent from LiteLLM's price map still cost zero (`gemini-3.x-flash-*` report
  `input=0`/`output=0`), and Anthropic is not covered at all — see below.

### Known issues

- **Claude streaming spend is wrong, and the cause is upstream.** The fix above covers
  Codex and Antigravity, which this plugin serves through its own streaming path.
  Anthropic is served by LiteLLM's native path instead, so the wrapper built here is never
  involved and there is nothing in this package to correct.

  The proxy restamps each outgoing chunk with the public model name. Anthropic carries
  usage on the first chunk (`message_start`) and on `message_delta`; usage-bearing chunks
  are stored as pre-restamp copies while ordinary ones are stored by reference, so what
  reaches `stream_chunk_builder` is `[wire, public, public, wire]` and the assembled model
  is the public name. Measured on 1.101.0, driving a real `CustomStreamWrapper`:

  | | logged model | cost |
  |---|---|---|
  | restamp off | `anthropic/claude-opus-4-20250514` | `0.609735` |
  | restamp on | `mysubs/claudecode/claude-opus-5` | `BadRequestError` |

  The failure mode differs from the one fixed above: the public name is absent from the
  price map entirely, so `completion_cost` raises rather than quietly returning `0.0`.
  Anthropic calls are therefore missing from spend tracking rather than present at zero,
  which is why those rows look different from the Codex rows in `/ui/logs`.

  Tracked upstream as [BerriAI/litellm#42161](https://github.com/BerriAI/litellm/issues/42161),
  with [PR #42176](https://github.com/BerriAI/litellm/pull/42176) open against it. Not
  worked around in-tree: the mutation happens in the proxy after these chunks have left,
  and pricing from a name this plugin never put on the wire would bill against another
  model's rate.

### Documented

- **Known incompatibility with `store_model_in_db: true`.** The proxy schedules an
  `add_deployment` reconcile every 30s whose cleanup step deletes every Router entry that
  is in neither the database nor `config.yaml` — unconditionally, and without logging
  anything. The deployments this plugin injects live in memory, so they are evicted within
  seconds of being applied. Measured on 1.101.0: 37 models at t+5s, 0 at t+15s.

  The README now carries the diagnosis, the workaround (`store_model_in_db: false`, plus
  the `STORE_MODEL_IN_DB` environment variable that overrides the YAML), and the
  three-line upstream change — honouring `model_info.managed_by` in the cleanup loop —
  that would let a database catalog and a plugin coexist. See `docs/DECISIONS.md` (D11)
  for why it is not worked around in-tree.

## [0.1.1] - 2026-09-19

Two installation bugs found by installing from a clean clone into an empty `HOME` —
neither was covered by the test suite, and both now are.

### Fixed

- `mysubs-setup` produced invalid YAML when the configuration already had
  `litellm_settings: {}`. Appending an indented `callbacks:` under an inline empty mapping
  does not parse, so the run refused to write and left a parser error and no installation.
- `mysubs-setup` reported `Already connected. Nothing to do.` on a system with no
  credential at all. The line described the callback being present in `config.yaml`, but
  it reads as a subscription being connected, which sends a fresh install looking for a
  problem that is not there.

### Changed

- The Anthropic requests identify themselves as `litellm-mysubs/…` instead of claiming to
  be `claude-cli`. Measured against the real endpoint, three runs each: the `User-Agent`
  changes no outcome, so the claim bought nothing. The identity block in `system` stays —
  without it the request returns 429 regardless of what the `User-Agent` says.
- `README` now states which modules are a port of
  [oh-my-pi](https://github.com/can1357/oh-my-pi) and which are not, and `LICENSE` carries
  the upstream copyright notice that its MIT terms require.
- `README` opens with what the tool is and where the providers' terms stand, before the
  install instructions.

## [0.1.0] - 2026-09-19

First release.

### Added

- LiteLLM proxy plugin loaded as a single `callbacks` entry in `config.yaml`
  (`litellm_mysubs.proxy_handler_instance`). It does not modify `model_list`,
  `router_settings` or `general_settings`.
- `mysubs-setup` command: locates the LiteLLM configuration used by the environment,
  adds the callback line, and reports the current state with `--status`.
- `mysubs-login <provider> --url <proxy> --code <pairing>` command: runs on the user's
  machine, opens the local callback port and completes the OAuth return without exposing
  the proxy to the browser. Pasting the return URL into the page is supported as a
  fallback.
- MySubs page mounted on the proxy and reachable from the LiteLLM UI under
  **Experimental → MySubs**, gated by LiteLLM administrator authentication.
- OAuth connection flows for three subscription providers: Anthropic (Claude Max),
  OpenAI (ChatGPT Plus / Codex) and Google (Antigravity).
- Model discovery: each connected subscription is probed to determine which models it
  actually serves; the selected ones are injected into the LiteLLM Router as deployments
  prefixed with `mysubs/<subscription>/`.
- Protocol bridges that translate OpenAI-compatible requests into each provider's own
  wire format (Anthropic Messages, OpenAI Responses, Antigravity Cloud Code), including
  streaming, reasoning content, vision input and usage accounting.
- Background token refresh with cross-process `flock` coordination, safe under a
  multi-worker proxy.
- Three credential stores: local file (mode `0600`), the LiteLLM Secret Manager, and
  environment variables.
- Per-subscription usage reporting surfaced on the MySubs page.
- Request throttling per worker, with retry policy derived from provider responses.
- `docs/DECISIONS.md`: ten architecture decisions, each recorded with the measurement
  that supports it and the conditions that would reopen it.
- CI covering Python 3.11-3.13, a LiteLLM version matrix (pinned `1.101.0` and `latest`),
  a contract test against the LiteLLM internal symbols the plugin depends on, and a
  drift check over the source anchors.

[Unreleased]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/eduardopessin/litellm-mysubs/releases/tag/v0.1.0

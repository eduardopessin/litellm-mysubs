# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.9] - 2026-09-22

### Fixed

- **0.1.8 did not actually fix the streamed spend.** The rollout went out, the row still
  read `0.00000000`, and the mistake was in where the correction was applied rather than
  in the diagnosis.

  0.1.8 re-wrapped the stream in an async generator and rewrote the chunks it yielded.
  Two things wrong with that. The cost is computed **inside** `CustomStreamWrapper`, from
  `self._provider_response_model`, which `chunk_creator` reads off the raw chunk before
  anything downstream sees it — so rewriting what comes out changes nothing. And the
  proxy needs the wrapper object itself, because it reads the finished turn off it; a
  generator in its place loses that interface.

  The wrapper is now corrected in place and handed back as itself, with `chunk_creator`
  wrapped — the one place every chunk passes through. Verified against a real
  `CustomStreamWrapper` rather than a stand-in, which is what the new tests do: with
  0.1.8's shape restored, five of them fail.

## [0.1.8] - 2026-09-22

### Fixed

- **Streamed Claude turns are priced again.** The spend row for a streamed call read
  `0.00000000` while the same prompt, unstreamed, was costed normally. Measured on the
  live gateway, three pairs a second apart:

  | streamed | prompt | completion | spend |
  |---|---|---|---|
  | no | 38 | 21 | 0.00071500 |
  | yes | 38 | 24 | **0.00000000** |
  | no | 38 | 24 | 0.00079000 |
  | yes | 38 | 24 | **0.00000000** |

  86 of the last 100 rows carrying tokens were at zero, including turns of 198k prompt
  tokens.

  The cause is upstream and needs no subscription to see
  ([BerriAI/litellm#42161](https://github.com/BerriAI/litellm/issues/42161)). Anthropic
  names the dated build in `message_start`, so a streamed turn arrives with
  `provider_response_model = claude-opus-5-20250930`. Since LiteLLM `134a4cd9fd` the cost
  calculation prefers that field over `response.model` — and the price map carries
  `claude-opus-5`, not the dated build. Same usage, one field apart, on 1.101.0::

      provider_response_model=claude-opus-5-20250930  ->  0.0
      provider_response_model=claude-opus-5           ->  0.00079
      (field absent)                                  ->  0.00079

  Unstreamed turns carry no such field, fall through to `response.model` and price
  correctly, which is why only streaming lost the money.

  Anthropic has no branch in `dispatch` — it is served by LiteLLM's native client — so
  the response on its way back through the wrapper is the only place this package can
  correct it. The field is rewritten to the name that has a rate rather than dropped: it
  is the provider's own answer about which build served the turn, and that belongs in the
  log. A dated name that has a rate of its own (`claude-haiku-4-5-20251001`) is left
  alone, and only a trailing 8-digit date is trimmed, so a build number or a size suffix
  is never mistaken for one.

  This is a workaround for a defect that is not ours, and it should come out when the
  upstream fix lands. A deployment with an explicit `input_cost_per_token` in
  `model_info` is unaffected either way: that sets `custom_pricing` and short-circuits
  the preference.

## [0.1.7] - 2026-09-22

### Fixed

- **A Claude subscription no longer reports "out of extra usage" with credit on the
  account.** Three tool names declared together — `skill_manage`, `skill_view` and
  `skills_list` — are read by Anthropic as a third-party agent, and the turn is refused::

      400 You're out of extra usage. Add more at claude.ai/settings/usage and keep going.

  The account is fine, and that is what made it hard to see: the operator tops up, the
  error persists, and a plain `curl` to the same model answers 200. What differs is the
  tool list. Measured on the live gateway against `claude-opus-5`, everything else held
  equal:

  | tools in the request | status |
  |---|---|
  | 25 client tools, names untouched | 400 |
  | the same 25 minus the trio | 200 |
  | the trio alone | 400 |
  | any two of the three | 200 |
  | 25 with the trio under `mcp__` | 200 |

  Not a size limit: padding the set back to the same byte count without the trio still
  passes (45152 bytes) while the trio fails at 45300. Not a credit limit either — the
  same credential serves the 200s in the table.

  The trio now travels under `mcp__`, the namespace Claude Code itself uses for
  MCP-provided tools, and the response is mapped back so the client only ever sees the
  names it declared. Renaming happens only when all three are present, because any two
  pass and renaming them would buy nothing.

  Two details that a first attempt got wrong, both found by testing rather than by
  reading:

  - `tool_choice` names a tool. Renaming the tools and leaving the choice behind points
    at a tool the request no longer declares — `400 Tool 'skills_list' not found in
    provided tools`. Both spellings are followed, the OpenAI `{"function": {"name": …}}`
    and the Anthropic `{"type": "tool", "name": …}`.
  - The response is not a dict. LiteLLM's native client answers with pydantic models, so
    a mapping-only walk left the alias in place on precisely the route that serves
    Anthropic — `dispatch` has no branch for it. Both shapes are walked now, along with
    the `tool_use` blocks `/v1/messages` emits and the `delta` a streaming chunk carries.

  A name already taken in the same request is skipped: two identical tool names is a hard
  400, strictly worse than the fingerprint being avoided.

## [0.1.6] - 2026-09-21

### Added

- **`/v1/messages` serves every subscription.** Anthropic-native clients speak Messages,
  and answering only chat-completions and Responses made each of them adapt — which is
  what a proxy exists to avoid. Measured on the live gateway, two of the three columns
  were 401s:

  | | chat | responses | messages |
  |---|---|---|---|
  | `mysubs/claudecode/*` | 200 | 200 | **401** |
  | `mysubs/codex/*` | 200 | 200 | **401** |
  | `mysubs/antigravity/*` | 200 | 200 | 200 (LiteLLM's own adapter, not this plugin) |

  The 401 is the failure mode the Responses route had before 0.1.3, for the same reason:
  with no interception the request reaches LiteLLM's native client, and the credential is
  OAuth — it lives in the store, not in `config.yaml`.

  Only the **envelope** is translated. The turns converge on the canonical list `dispatch`
  already consumes and each provider keeps its own wire, so Codex still goes out as
  Responses and Antigravity as Cloud Code, both inheriting every fix below. A defect fixed
  on one route cannot leave another behind.

  Claude Max is the deliberate gap: Messages *is* its wire, so the translation is declined
  and the native path answers it. Two things that declining does **not** excuse, both found
  by deploying and both now covered by tests: the OAuth token still has to be injected
  (`401 Missing Anthropic API Key`), and the Claude Code identity belongs in the top-level
  `system` rather than in `messages[0]` (`400 messages.0: use the top-level 'system'
  parameter`). Placement is a parameter on `build_request` now, because both callers are
  legitimate and only the route knows which is which.

  Streaming replays the finished turn as the Anthropic event sequence. The upstreams do
  not speak it, so translating mid-flight would mean inventing block indices; producing
  the turn first keeps them contiguous, which is what a client tracking them needs.

### Fixed

- **Claude served by Antigravity lost its tool call ids.** `supports_function_ids` gated
  the id on the model name starting with `gemini-3`. Antigravity also serves Anthropic,
  and those run on Vertex, where `tool_use.id` is required — so every Claude served there
  sent `functionCall` with no id. The first turn passes, and the one carrying the result
  back is refused:

  ```
  HTTP 400 messages.1.content.0.tool_use.id: Field required
  ```

  Measured over one omp run: 76 failures on `claude-opus-4-6-thinking` and 70 on
  `claude-sonnet-4-6`, every one on the second turn, while the same models served natively
  by Anthropic were fine. Any agent that uses tools was broken on those two.

- **The thinking budget ignored the caller's output ceiling.** `maxOutputTokens` and
  `thinkingBudget` travelled independently — the ceiling is the client's, the budget comes
  from the catalog — and on the Anthropic backend they are not independent. Two bounds
  apply at once and they close on each other:

  ```
  max_tokens     > budget_tokens    (the ceiling has to leave room)
  budget_tokens >= 1024             (Anthropic's own minimum)
  ```

  So a ceiling of 1024 or less admits no valid budget at all; capping at three quarters —
  the first attempt — merely swapped one rejection for the other. Such a turn is now served
  with thinking off rather than failing: the caller asked for a ceiling, not for reasoning.
  Above that the budget is capped at `ceiling - 1`, never below the floor.

- **A quota refusal reached the client as HTTP 500.** `UpstreamError` carries the real 429,
  but the proxy has no class for it, so an unrecognised exception is reported as
  `internal_server_error` and the status survives only as text inside the message. A client
  cannot back off on a 500, and backing off is the one correct response here. A 429 is now
  raised as `litellm.exceptions.RateLimitError` on both the streaming and non-streaming
  paths; nothing else is remapped, and `RemapRequired`/`RedeemRequired` stay as they are —
  they are signals to `plugin.py`, not answers to the client.

- **A 429 tried the second Antigravity host for nothing.** Failover exists for endpoint
  faults, and a quota refusal is not one: both hosts front the same account and the same
  quota, so the second request repeats a refusal already known. It turned an ~11 s failure
  into ~22 s and changed nothing else. 404 and 503 still fail over.

- **`/v1/responses` turns produced no spend row.** There are three paths, not two, and the
  Responses route had neither of the other two's logging. It is the path that matters most
  in practice: a client that discovers models through LiteLLM routes every OpenAI-backed
  model here, because the plugin declares `providers: ['openai']` on those deployments.

  Measured on the live gateway: an omp run exercising all five Codex models end to end left
  no Codex row of any kind, while Anthropic and Gemini — which go through chat — logged 33
  and 50 over the same minutes. Same accounting hole 0.1.5 closed for chat:
  `x-litellm-key-spend` undercounts and per-key budgets never see these calls.

- **Claude Max on `/v1/messages` answered 401, then 400.** Declining the translation
  routes the turn to LiteLLM's native path, which does not excuse the two things the
  plugin still owes it: the OAuth token has to be injected (`401 Missing Anthropic API
  Key`), and the Claude Code identity belongs in the top-level `system` rather than in
  `messages[0]` (`400 messages.0: use the top-level 'system' parameter`). Placement is a
  parameter on `build_request` now, because both callers are legitimate and only the route
  knows which is which.

- **Spend rows named the model the client asked for, and no provider.** The row reads its
  identity off the logging object, per request — and a request served by this plugin never
  reaches the provider client that would fill those in. Measured on the live gateway: rows
  served natively read `anthropic/claude-opus-5` + `anthropic`, rows served here read
  `mysubs/antigravity/...` with an empty provider. An empty provider is also what leaves
  the Logs tab without an icon, and a name with no rate is what left the cost at zero.

- **Rows carried the right name and still billed nothing.** `completion_cost` was asked
  under the public name, which has no entry in the price map. Measured on 1.101.0 with
  identical usage: `("mysubs/codex/gpt-5.5", "custom_openai")` → `0.0`, and
  `("openai/gpt-5.5", "openai")` → `0.0202325`. 49 of 50 rows were at zero, including
  `claude-opus-5` turns of 123k tokens whose rate is in the map. A model with no rate now
  leaves the field unset rather than asserting the turn was free.

- **Every row read `duration = 0` and had no TTFT.** One instant was captured after the
  await and passed as both `start_time` and `end_time`, so the duration was never measured
  rather than merely small — natively-served rows alongside read 3.9 s to 14 s. TTFT needed
  a second fix and then a third: writing `model_call_details["completion_start_time"]` is a
  no-op, because `_success_handler_helper_fn` tests the **attribute** and overwrites it
  with `end_time`; and timing the first event of any kind measures `response.created`,
  which this plugin emits before the upstream has answered. Measured: `ttft=1ms` against a
  114-second turn. It is taken at the first event carrying output now, through
  `_update_completion_start_time`, which is what LiteLLM's own `_process_chunk` calls.

- **A streamed `/v1/responses` turn left no row at all.** Four defects stacked, each
  hiding the next, and the client got a correct answer every time:

  | # | cause |
  |---|---|
  | 1 | logging sat after the `async for`; the proxy closes the generator at `response.completed`, so `GeneratorExit` discarded it |
  | 2 | the object had no `completed_response`, which is where the proxy reads the finished turn |
  | 3 | it was not a `BaseResponsesAPIStreamingIterator`, so the Router's `isinstance` gate handed it back raw |
  | 4 | the success handler was given `result.response` instead of the terminal `ResponseCompletedEvent`, and `_get_assembled_streaming_response` returns `None` for anything else — silently |

  Subclassing the base means inheriting its methods, so every attribute its constructor
  sets is mirrored by hand; `_stream_created_time` is read on every `__anext__` and
  `_hidden_params` is what the proxy reads to build `x-litellm-model-id` and friends.

- **Gemini turns were costed zero because the effort is part of the model id.**
  `gemini/gemini-3.6-flash-low` has no rate; `gemini/gemini-3.6-flash` does. Effort changes
  the thinking budget, not the per-token rate, so the base name is what the turn is priced
  against — and only when that name actually has a rate, which keeps `gemini-3-flash-agent`
  from being repriced against a sibling it merely shares a prefix with. Recovers 21 of the
  32 served models; the other 11 have no rate under any name and stay unpriced.

- **Gemini on `/v1/responses` was delegated, and delegating does not stop the turn from
  spending the subscription.** The native path prices from the response object, which
  carries the public name and `cost: None` — the mechanism of
  [BerriAI/litellm#42161](https://github.com/BerriAI/litellm/issues/42161), on this route.
  Stamping the identity before the hand-off fixed the provider and not the name. The turn
  is served here now.

  The first version of that serving was a shortcut: it awaited the whole chat turn and
  replayed it as two events. It answered and it priced, and it was not a stream. Measured
  against Codex on the same route and prompt:

  ```
  codex   events=7117  deltas=7107  ttft=    30ms
  gemini  events=   2  deltas=   0  ttft= 20911ms
  ```

  A client reading `text_deltas` got nothing for 21 seconds — same route, same client, two
  contracts. Nothing about the incremental form was Codex-specific: `_ResponsesStreamState`
  already owned every id, index and sequence number. The driver is parameterised by spec,
  reader and usage mapper now, and both subscriptions emit the same sequence: 61 events and
  29 ms on the wire, against Codex's 24 ms.

- **A quota refusal on `/v1/responses` reached the client as HTTP 500.** `dispatch` has had
  the `UpstreamError` translation since the chat route existed; `dispatch_responses` never
  did. It went unnoticed while Gemini was delegated, because LiteLLM's native path
  normalised the error on the way out. A client cannot back off on a 500.

- **A logging failure could truncate a paid-for stream.** `_emit` set its guard before
  unprotected work: both stamps begin by reading `logging_obj.model_call_details`, outside
  the `try`. Measured against a logging object whose attribute raises: 0 of 3 events
  delivered, with `_emitted` already set so `aclose()` would not retry and the `except`
  never ran — the silent-failure mode the surrounding commits existed to remove.

- **The plugin's own diagnostics went nowhere.** `verbose_proxy_logger` is the only logger
  with a handler inside the proxy — the root has none — and it sits at WARNING by default.
  A `getLogger(__name__)` wrote to nothing and an `.info()` was dropped by level, which is
  why a diagnostic build looked like code that never ran. Three `contextlib.suppress`
  blocks that hid real failures (route binding, model reapply, the stream's own emit) now
  report, and the gateway sets `LITELLM_LOG=INFO`.

- **`/v1/messages` replayed the finished turn instead of streaming it.** The same defect
  as on `/v1/responses`, on the last route that still had it, and defended by the same
  argument: that a translated stream would have to invent block indices mid-flight. The
  indices are ours either way — neither upstream sends them — so numbering blocks as they
  open is no more invented than numbering them at the end, and it is what lets text leave
  as it arrives.

  Measured on the live gateway, same prompt and ceiling, `events / deltas / time to first
  event`, before and after:

  | | before | after |
  |---|---|---|
  | Codex | 9 / — / 30714 ms | 1692 / 1687 / 63 ms |
  | Antigravity | 6 / — / 6916 ms | 63 / 58 / 38 ms |

  `message_start` is emitted immediately rather than on the first token: it carries no
  content, and holding it back made the route look slower than it was. Reasoning and tool
  calls are still emitted as whole blocks once the turn closes, which is the division the
  Responses route makes and for the same reason — neither upstream streams them in a form
  that can be replayed without a second interpretation of the same events.

  The spend row comes with it. This route never reaches `CustomStreamWrapper`, because it
  emits Anthropic events rather than chat chunks, so it had the accounting hole
  `/v1/responses` had: `_logged_messages` dispatches from the usage on `message_delta`,
  in a `finally`, so a consumer that stops reading early still bills the turn its
  subscription has been charged for. `UpstreamError` is translated here too, which this
  route also lacked — a quota refusal reached the client as 500 rather than 429.

### Changed

- **`plugin.py` split by responsibility.** It had grown to 1807 lines covering six
  unrelated jobs, and the Messages route was about to add a seventh. Only one of those is
  "plugin" in the sense of coupling to LiteLLM; the rest is logic that does not need to sit
  next to a monkey-patch.

  | module | lines | job |
  |---|---|---|
  | `plugin.py` | 1807 → 459 | install/uninstall, `bind_*`, Router wrappers |
  | `routes.py` | 631 | the three `dispatch*` and the turn builders |
  | `turns.py` | 392 | event readers, chunks, `_model_response` |
  | `specs.py` | 299 | process state, credential, per-wire specs |
  | `observability.py` | 254 | logging, cost identity, error mapping |

  No behaviour change: the suite passes untouched except for pointing `plugin._x` at the
  module that now owns it, which is the point. Two seams needed care — `turns.py` writes
  thought signatures through an injected sink rather than importing the state back, which
  would close a cycle, and `_WIRE_MODEL_KEY` moved to `observability.py`, which is what
  reads it.

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

[Unreleased]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.9...HEAD
[0.1.9]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/eduardopessin/litellm-mysubs/releases/tag/v0.1.0

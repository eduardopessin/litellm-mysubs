# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

  Still zero, and upstream data rather than dispatch: Antigravity models absent from
  LiteLLM's price map (`gemini-3.x-flash-*` report `input=0`/`output=0`).

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

[Unreleased]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/eduardopessin/litellm-mysubs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/eduardopessin/litellm-mysubs/releases/tag/v0.1.0

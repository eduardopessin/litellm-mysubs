# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

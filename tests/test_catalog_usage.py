"""Estado de uso lido dos cabeçalhos das respostas.

Os valores destes testes foram medidos contra o proxy real, não inventados.
"""

from __future__ import annotations

from litellm_mysubs.catalog.usage import from_anthropic_headers, from_codex_headers, from_headers

#: Resposta real de `gpt-5.5` medida em 2026-09-18.
CODEX_HEADERS = {
    "x-codex-active-limit": "premium",
    "x-codex-credits-balance": "0",
    "x-codex-plan-type": "plus",
    "x-codex-primary-reset-at": "1789790331",
    "x-codex-primary-used-percent": "0",
    "x-codex-primary-window-minutes": "300",
    "x-codex-secondary-reset-at": "1789999782",
    "x-codex-secondary-used-percent": "19",
    "x-codex-secondary-window-minutes": "10080",
}

#: Resposta real de `claude-opus-5`, como o proxy a reexpõe.
ANTHROPIC_HEADERS = {
    "llm_provider-anthropic-ratelimit-unified-5h-reset": "1789789200",
    "llm_provider-anthropic-ratelimit-unified-5h-utilization": "0.03",
    "llm_provider-anthropic-ratelimit-unified-7d-reset": "1790204400",
    "llm_provider-anthropic-ratelimit-unified-7d-utilization": "0.24",
}


class TestScales:
    def test_anthropic_fraction_becomes_a_percentage(self) -> None:
        """A Anthropic dá 0.24 para 24%; o Codex dá 19 para 19%.

        Tratar as duas escalas como iguais mostrava 0.24% onde são 24% — um card que diz
        "quase sem uso" numa conta a um quarto do limite semanal.
        """
        snapshot = from_anthropic_headers(ANTHROPIC_HEADERS)
        assert [round(w.used_percent) for w in snapshot.windows] == [3, 24]

    def test_codex_percentage_is_taken_as_is(self) -> None:
        snapshot = from_codex_headers(CODEX_HEADERS)
        assert [round(w.used_percent) for w in snapshot.windows] == [0, 19]


class TestWindows:
    def test_window_minutes_become_readable_labels(self) -> None:
        """300 minutos é "5h" e 10080 é "7d" — é assim que o utilizador pensa no limite."""
        snapshot = from_codex_headers(CODEX_HEADERS)
        assert [w.label for w in snapshot.windows] == ["5h", "7d"]

    def test_reset_is_a_countdown_not_a_timestamp(self) -> None:
        snapshot = from_codex_headers(CODEX_HEADERS, now=1789790331 - 600)
        assert snapshot.windows[0].resets_in_s(now=1789790331 - 600) == 600

    def test_a_reset_in_the_past_never_goes_negative(self) -> None:
        """Um contador negativo no ecrã lê-se como erro; zero lê-se como "já repôs"."""
        snapshot = from_codex_headers(CODEX_HEADERS)
        assert snapshot.windows[0].resets_in_s(now=1789790331 + 5000) == 0


class TestUnknown:
    def test_a_provider_without_quota_headers_is_reported_as_unknown(self) -> None:
        """Medido: o Antigravity não devolve cabeçalhos de quota. Uma barra a zero seria
        lida como "por usar" — o oposto do que se sabe, que é nada."""
        snapshot = from_headers("google-antigravity", {"content-type": "application/json"})
        assert not snapshot.known
        assert snapshot.windows == ()

    def test_garbage_values_do_not_become_zero(self) -> None:
        """`used_percent=0` por causa de um valor ilegível seria inventar um facto."""
        snapshot = from_codex_headers({"x-codex-primary-used-percent": "muito"})
        assert snapshot.windows == ()

    def test_an_empty_snapshot_carries_no_timestamp(self) -> None:
        """Sem dados não há instantâneo, e sem instantâneo não há idade a mostrar."""
        assert from_headers("anthropic", {}).taken_at == 0.0


class TestProxyPrefix:
    def test_headers_are_read_with_or_without_the_proxy_prefix(self) -> None:
        """O mesmo cabeçalho chega com nomes diferentes conforme se fale directamente com o
        upstream ou através do proxy."""
        direto = {k.replace("llm_provider-", ""): v for k, v in ANTHROPIC_HEADERS.items()}
        assert (
            from_anthropic_headers(direto).windows
            == from_anthropic_headers(ANTHROPIC_HEADERS).windows
        )

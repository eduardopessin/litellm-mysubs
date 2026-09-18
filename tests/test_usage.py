"""Contabilidade de tokens e desfecho de turno.

O que não for reportado aqui perde-se: o LiteLLM cai para estimativas e os cache hits
desaparecem do `/spend/logs`. Numa conta de subscrição, cache invisível é a diferença
entre saber e não saber porque a quota acabou.
"""

from __future__ import annotations

import pytest

from litellm_mysubs.wire.usage import (
    codex_finish_reason,
    codex_usage,
    google_finish_reason,
    google_usage,
    make_usage,
)


class TestGoogleUsage:
    def test_cached_tokens_are_not_double_counted(self) -> None:
        """promptTokenCount inclui os cached; somá-los outra vez inflaciona a factura."""
        usage = google_usage(
            {
                "promptTokenCount": 1000,
                "cachedContentTokenCount": 400,
                "candidatesTokenCount": 50,
                "totalTokenCount": 1050,
            }
        )
        assert usage.prompt_tokens == 600
        assert usage.cached_tokens == 400
        assert usage.total_tokens == 1050

    def test_thoughts_count_as_output(self) -> None:
        usage = google_usage({"candidatesTokenCount": 50, "thoughtsTokenCount": 120})
        assert usage.completion_tokens == 170
        assert usage.reasoning_tokens == 120

    def test_empty_metadata_is_zeroed(self) -> None:
        usage = google_usage({})
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (0, 0, 0)


class TestCodexUsage:
    def test_input_tokens_are_not_reduced_by_cache(self) -> None:
        """Ao contrário do Google: subtrair aqui subcontaria o prompt."""
        usage = codex_usage({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400}})
        assert usage.prompt_tokens == 1000
        assert usage.cached_tokens == 400

    def test_legacy_cache_field_is_accepted(self) -> None:
        usage = codex_usage({"input_tokens": 10, "prompt_cache_hit_tokens": 7})
        assert usage.cached_tokens == 7

    def test_reasoning_tokens_extracted(self) -> None:
        usage = codex_usage(
            {"output_tokens": 80, "output_tokens_details": {"reasoning_tokens": 60}}
        )
        assert usage.reasoning_tokens == 60


class TestMakeUsage:
    def test_total_defaults_to_the_sum(self) -> None:
        assert make_usage(prompt_tokens=10, completion_tokens=5).total_tokens == 15

    def test_explicit_total_wins(self) -> None:
        """O upstream sabe melhor: pode incluir tokens que não discriminámos."""
        assert make_usage(prompt_tokens=10, completion_tokens=5, total_tokens=99).total_tokens == 99

    @pytest.mark.parametrize("value", [None, "", "nao-e-numero", -5])
    def test_junk_becomes_zero(self, value: object) -> None:
        """Um contador partido não pode rebentar a contabilidade do turno."""
        assert make_usage(prompt_tokens=value).prompt_tokens == 0


class TestGoogleFinishReason:
    def test_tool_calls_win_over_stop(self) -> None:
        assert google_finish_reason("STOP", has_tool_calls=True) == "tool_calls"

    def test_max_tokens_is_truncation(self) -> None:
        """Sem isto, um corte por limite chegava como resposta normal e curta."""
        assert google_finish_reason("MAX_TOKENS", has_tool_calls=False) == "length"

    def test_max_tokens_with_pending_tool_call(self) -> None:
        """A chamada ainda é o desfecho do turno, mesmo com o limite atingido."""
        assert google_finish_reason("MAX_TOKENS", has_tool_calls=True) == "tool_calls"

    @pytest.mark.parametrize(
        "reason",
        [
            "SAFETY",
            "RECITATION",
            "PROHIBITED_CONTENT",
            "MALFORMED_FUNCTION_CALL",
            # As cinco que faltavam quando se enumerava o erro em vez do sucesso.
            "FINISH_REASON_UNSPECIFIED",
            "LANGUAGE",
            "IMAGE_OTHER",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
        ],
    )
    def test_server_side_blocks_surface_as_content_filter(self, reason: str) -> None:
        """O único valor OpenAI que não mente sobre um corte imposto pelo servidor."""
        assert google_finish_reason(reason, has_tool_calls=False) == "content_filter"

    def test_unknown_reason_is_an_error_not_a_stop(self) -> None:
        """O OMP enumera o que é normal e trata o resto como erro.

        Uma razão nova do upstream tratada como `stop` entrega uma resposta cortada como
        se estivesse completa. Tratada como erro, no pior caso é ruidosa de mais.
        """
        assert google_finish_reason("RAZAO_QUE_AINDA_NAO_EXISTE", has_tool_calls=False) == (
            "content_filter"
        )

    def test_blocked_reason_is_not_masked_by_tool_calls(self) -> None:
        """Um bloqueio de segurança não pode passar por tool_calls."""
        assert google_finish_reason("SAFETY", has_tool_calls=True) == "content_filter"

    @pytest.mark.parametrize("reason", [None, "", "STOP"])
    def test_normal_completion(self, reason: object) -> None:
        assert google_finish_reason(reason, has_tool_calls=False) == "stop"

    def test_case_and_whitespace_tolerated(self) -> None:
        assert google_finish_reason("  max_tokens  ", has_tool_calls=False) == "length"


class TestCodexFinishReason:
    def test_incomplete_is_truncation(self) -> None:
        assert codex_finish_reason("incomplete", has_tool_calls=False) == "length"

    def test_tool_calls_win(self) -> None:
        assert codex_finish_reason("incomplete", has_tool_calls=True) == "tool_calls"

    def test_default_is_stop(self) -> None:
        assert codex_finish_reason(None, has_tool_calls=False) == "stop"
        assert codex_finish_reason("completed", has_tool_calls=False) == "stop"

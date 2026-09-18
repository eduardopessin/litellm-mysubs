"""Política de retry e rotação de endpoint.

No original estas decisões viviam dentro dos laços de ``httpx``, duplicadas entre a versão
síncrona e a assíncrona — e as duas tinham derivado formas diferentes da mesma regra.
Testá-las isoladas é o que impede essa divergência de voltar.
"""

from __future__ import annotations

import pytest

from litellm_mysubs.transport.hosts import (
    HOSTS,
    MAX_EMPTY_RETRIES,
    STREAM_PATH,
    HostRotation,
    empty_retry_delay,
)
from litellm_mysubs.transport.retry import (
    Action,
    decide_antigravity,
    decide_codex,
    is_unsupported_model,
)


class TestCodexDecisions:
    def test_success_returns(self) -> None:
        assert decide_codex(200).action is Action.RETURN

    def test_401_refreshes_the_token(self) -> None:
        decision = decide_codex(401)
        assert decision.action is Action.REFRESH_TOKEN
        assert decision.should_retry is True

    def test_unsupported_alias_is_remapped(self) -> None:
        """Só nomes de família: resolvê-los para a versão servida é honesto."""
        decision = decide_codex(
            400, "The 'codex' model is not supported when using Codex", can_remap=True
        )
        assert decision.action is Action.REMAP_MODEL

    def test_unsupported_arbitrary_name_fails(self) -> None:
        """Substituir um nome arbitrário devolvia 200 com o campo `model` a ecoar o
        pedido, e a facturação passava a mentir."""
        decision = decide_codex(
            400, "The 'gpt-4.1' model is not supported when using Codex", can_remap=False
        )
        assert decision.action is Action.FAIL
        assert decision.should_retry is False

    def test_other_400_is_not_a_model_problem(self) -> None:
        """Um payload inválido não se resolve trocando de modelo."""
        assert decide_codex(400, "Invalid value at 'input'").action is Action.FAIL

    def test_429_redeems_when_credit_exists(self) -> None:
        assert decide_codex(429, can_redeem=True).action is Action.REDEEM_CREDIT

    def test_429_without_credit_fails(self) -> None:
        """Sem crédito, retentar dava o mesmo erro três vezes e triplicava a latência."""
        assert decide_codex(429, can_redeem=False).action is Action.FAIL

    @pytest.mark.parametrize("status", [403, 404, 500, 502, 503])
    def test_other_statuses_propagate(self, status: int) -> None:
        decision = decide_codex(status)
        assert decision.action is Action.FAIL
        assert str(status) in decision.reason

    def test_marker_detection(self) -> None:
        assert is_unsupported_model("The 'x' model is not supported when using Codex") is True
        assert is_unsupported_model("rate limit exceeded") is False


class TestAntigravityDecisions:
    def test_success_returns(self) -> None:
        assert decide_antigravity(200).action is Action.RETURN

    def test_401_refreshes(self) -> None:
        assert decide_antigravity(401).action is Action.REFRESH_TOKEN

    @pytest.mark.parametrize("status", [404, 503])
    def test_no_model_degradation(self, status: int) -> None:
        """Um 404 é "a conta não serve isto" e um 503 é capacidade; nenhum autoriza
        responder com outro modelo."""
        decision = decide_antigravity(status)
        assert decision.action is Action.FAIL
        assert decision.should_retry is False


class TestHostRotation:
    def test_starts_on_the_primary(self) -> None:
        rotation = HostRotation()
        assert rotation.urls()[0].startswith(HOSTS[0])

    def test_always_offers_every_host(self) -> None:
        """Ambos são sempre tentados: nenhum fica excluído por uma falha anterior."""
        assert len(HostRotation().urls()) == len(HOSTS)

    def test_commit_remembers_the_last_good_host(self) -> None:
        rotation = HostRotation()
        rotation.commit(HOSTS[1] + STREAM_PATH)
        assert rotation.urls()[0].startswith(HOSTS[1])
        assert rotation.current == HOSTS[1]

    def test_fallback_host_still_offered_after_switching(self) -> None:
        """O primário não é abandonado: pode voltar a responder."""
        rotation = HostRotation()
        rotation.commit(HOSTS[1] + STREAM_PATH)
        assert any(url.startswith(HOSTS[0]) for url in rotation.urls())

    def test_unknown_url_does_not_move_the_pointer(self) -> None:
        rotation = HostRotation()
        rotation.commit("https://exemplo.invalido/x")
        assert rotation.current == HOSTS[0]

    def test_failover_allowed_before_anything_is_emitted(self) -> None:
        assert HostRotation().can_failover(is_last=False) is True

    def test_no_failover_after_the_first_event(self) -> None:
        """O cliente já viu parte da resposta: recomeçar noutro host duplicava-a."""
        rotation = HostRotation()
        rotation.mark_started()
        assert rotation.can_failover(is_last=False) is False

    def test_no_failover_on_the_last_endpoint(self) -> None:
        assert HostRotation().can_failover(is_last=True) is False

    def test_path_is_configurable(self) -> None:
        rotation = HostRotation()
        urls = rotation.urls("/v1internal:fetchAvailableModels")
        assert all(url.endswith("/v1internal:fetchAvailableModels") for url in urls)

    def test_instances_do_not_share_memory(self) -> None:
        """Dois clientes no mesmo processo podem estar em hosts diferentes."""
        first, second = HostRotation(), HostRotation()
        first.commit(HOSTS[1] + STREAM_PATH)
        assert second.current == HOSTS[0]


class TestEmptyStreamRetry:
    def test_backoff_doubles(self) -> None:
        """500 ms, 1 s — como o OMP."""
        assert empty_retry_delay(1) == 0.5
        assert empty_retry_delay(2) == 1.0

    def test_retry_budget_matches_the_source(self) -> None:
        assert MAX_EMPTY_RETRIES == 2

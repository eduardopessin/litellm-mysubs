"""Failover de endpoint do Antigravity.

Dois hosts aceitam o mesmo envelope. Memoriza-se o último que respondeu **por inteiro**,
para não pagar a falha do primeiro em cada pedido de uma sessão.

Duas invariantes que vêm da fonte e que são fáceis de perder:

1. O último bom só é comprometido depois de um stream **completo** — conteúdo e razão de
   fim. Marcá-lo à primeira resposta OK fixa a rotação num host que aceitou a ligação e
   depois cortou o stream a meio.
2. O failover só é legal enquanto **nada** foi emitido. Depois do primeiro evento o
   cliente já viu parte da resposta, e recomeçar noutro host duplicava-a.

Nota apurada em produção: quando ambos falham, a mensagem de erro nomeia o **último** host
tentado — o que faz o `sandbox` aparecer nos erros sem que ele esteja "preso".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

HOSTS: Final[tuple[str, ...]] = (
    "https://daily-cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
)

STREAM_PATH: Final = "/v1internal:streamGenerateContent?alt=sse"
MODELS_PATH: Final = "/v1internal:fetchAvailableModels"

# omp: providers/google-shared.ts :: MAX_EMPTY_STREAM_RETRIES
#: Tentativas **adicionais** quando o stream fecha sem conteúdo nenhum.
MAX_EMPTY_RETRIES: Final = 2
# omp: providers/google-shared.ts :: EMPTY_STREAM_BASE_DELAY_MS
#: Base do backoff exponencial: 500 ms, 1 s.
EMPTY_RETRY_BASE_S: Final = 0.5


def empty_retry_delay(attempt: int) -> float:
    """Atraso antes da tentativa ``attempt`` (1-indexada): ``base * 2^(n-1)``."""
    return float(EMPTY_RETRY_BASE_S * (2 ** max(0, attempt - 1)))


# omp: providers/google-gemini-cli.ts :: lastGoodEndpoint
@dataclass(slots=True)
class HostRotation:
    """Ordem de tentativa dos endpoints, com memória do último bom.

    Instância em vez de global: dois clientes no mesmo processo não devem partilhar a
    memória de qual host respondeu.
    """

    hosts: tuple[str, ...] = HOSTS
    index: int = 0
    #: Passa a ``True`` no primeiro evento emitido ao cliente.
    started: bool = False

    def urls(self, path: str = STREAM_PATH) -> list[str]:
        """Todos os endpoints, começando no último bom."""
        count = len(self.hosts)
        return [self.hosts[(self.index + offset) % count] + path for offset in range(count)]

    def can_failover(self, *, is_last: bool) -> bool:
        """Se é legal tentar o endpoint seguinte.

        Depois de o cliente ter visto o primeiro evento, mudar de host duplicaria a parte
        já entregue.
        """
        return not is_last and not self.started

    def mark_started(self) -> None:
        """Regista que já foi emitido conteúdo: o endpoint fica comprometido."""
        self.started = True

    def commit(self, url: str) -> None:
        """Memoriza o host **depois** de um stream completo.

        Só deve ser chamado com conteúdo e razão de fim recebidos — não à primeira
        resposta com estado 200.
        """
        for position, host in enumerate(self.hosts):
            if url.startswith(host):
                self.index = position
                return

    @property
    def current(self) -> str:
        return self.hosts[self.index]

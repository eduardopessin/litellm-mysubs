"""Failover de endpoint do Antigravity.

Dois hosts aceitam o mesmo envelope. Memoriza-se o último que respondeu, para não pagar a
falha do primeiro em cada pedido de uma sessão inteira.

Nota apurada em produção: quando ambos falham, a mensagem de erro nomeia o **último**
host tentado — o que faz o `sandbox` aparecer nos erros sem que ele esteja "preso". A
ordem de tentativa começa sempre no último bom, e ambos são sempre tentados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

HOSTS: Final[tuple[str, ...]] = (
    "https://daily-cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
)

STREAM_PATH: Final = "/v1internal:streamGenerateContent?alt=sse"
MODELS_PATH: Final = "/v1internal:fetchAvailableModels"

#: Tentativas quando o stream fecha sem conteúdo nenhum.
MAX_EMPTY_RETRIES: Final = 3
EMPTY_RETRY_BASE_S: Final = 1.0


# omp: providers/google-gemini-cli.ts :: lastGoodEndpoint
@dataclass(slots=True)
class HostRotation:
    """Ordem de tentativa dos endpoints, com memória do último bom.

    Instância em vez de global: dois clientes no mesmo processo não devem partilhar a
    memória de qual host respondeu.
    """

    hosts: tuple[str, ...] = HOSTS
    index: int = 0
    _seen: set[str] = field(default_factory=set, repr=False)

    def urls(self, path: str = STREAM_PATH) -> list[str]:
        """Todos os endpoints, começando no último bom."""
        count = len(self.hosts)
        return [self.hosts[(self.index + offset) % count] + path for offset in range(count)]

    def mark_good(self, url: str) -> None:
        """Memoriza o host que respondeu, para ser o primeiro da próxima vez."""
        for position, host in enumerate(self.hosts):
            if url.startswith(host):
                self.index = position
                self._seen.add(host)
                return

    @property
    def current(self) -> str:
        return self.hosts[self.index]

"""Estado de uso da subscrição, lido dos cabeçalhos das respostas.

Uma subscrição não tem endpoint de quota: o que existe são cabeçalhos que o upstream junta
a cada resposta. Sem os capturar, só se sabe que a conta esgotou quando chega o primeiro
429 — que é tarde para quem estava a contar com ela.

Os nomes e as formas foram **medidos** contra o proxy real, não lidos de documentação:

    x-codex-primary-used-percent: 0        x-codex-primary-window-minutes: 300
    x-codex-secondary-used-percent: 19     x-codex-secondary-window-minutes: 10080
    anthropic-ratelimit-unified-5h-utilization: 0.03   (fracção, não percentagem)
    anthropic-ratelimit-unified-7d-utilization: 0.24

As duas escalas diferem — o Codex dá inteiros de 0 a 100, a Anthropic uma fracção de 0 a 1
— e tratá-las como iguais mostrava 0.24% onde são 24%.

O Google Antigravity **não devolve nada disto** nos cabeçalhos: medido, zero. Mas tem um
endpoint próprio — `:retrieveUserQuotaSummary`, o mesmo que a UI dele usa — que devolve as
janelas de 5 horas e semanal. Ver `from_antigravity_summary`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

#: O proxy reexpõe os cabeçalhos do upstream com este prefixo.
_PROXY_PREFIX: Final = "llm_provider-"


@dataclass(frozen=True, slots=True)
class Window:
    """Uma janela de limite: quanto foi usado e quando repõe."""

    label: str
    used_percent: float
    resets_at: float = 0.0

    def resets_in_s(self, *, now: float | None = None) -> float | None:
        if self.resets_at <= 0:
            return None
        return max(0.0, self.resets_at - (time.time() if now is None else now))


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """O que se sabe do uso de uma subscrição, e quando se soube.

    `taken_at` não é enfeite: um instantâneo sem idade é lido como estado actual, e um valor
    de há três horas apresentado como agora é a forma mais barata de mentir. A UI mostra a
    idade sempre.
    """

    windows: tuple[Window, ...] = ()
    plan: str = ""
    credits_balance: str = ""
    taken_at: float = 0.0

    @property
    def known(self) -> bool:
        return bool(self.windows) or bool(self.plan)

    def age_s(self, *, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.taken_at


def _clean(headers: Mapping[str, Any]) -> dict[str, str]:
    """Cabeçalhos em minúsculas e sem o prefixo do proxy.

    O mesmo cabeçalho chega com nomes diferentes conforme se fale com o upstream
    directamente ou através do proxy; normalizar aqui evita dois caminhos de leitura.
    """
    out: dict[str, str] = {}
    for key, value in headers.items():
        lowered = str(key).lower()
        if lowered.startswith(_PROXY_PREFIX):
            lowered = lowered[len(_PROXY_PREFIX) :]
        out[lowered] = str(value)
    return out


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _window_label(minutes: float | None, fallback: str) -> str:
    if minutes is None or minutes <= 0:
        return fallback
    if minutes < 60:
        return f"{int(minutes)} min"
    if minutes < 1440:
        return f"{int(minutes // 60)}h"
    return f"{int(minutes // 1440)}d"


def from_codex_headers(headers: Mapping[str, Any], *, now: float | None = None) -> UsageSnapshot:
    """Uso do ChatGPT Plus a partir dos `x-codex-*`.

    As percentagens já vêm em 0-100.
    """
    h = _clean(headers)
    windows: list[Window] = []
    for prefix, fallback in (("primary", "5h"), ("secondary", "7d")):
        used = _number(h.get(f"x-codex-{prefix}-used-percent"))
        if used is None:
            continue
        windows.append(
            Window(
                label=_window_label(_number(h.get(f"x-codex-{prefix}-window-minutes")), fallback),
                used_percent=used,
                resets_at=_number(h.get(f"x-codex-{prefix}-reset-at")) or 0.0,
            )
        )
    plan = h.get("x-codex-plan-type", "")
    if not windows and not plan:
        return UsageSnapshot()
    return UsageSnapshot(
        windows=tuple(windows),
        plan=plan,
        credits_balance=h.get("x-codex-credits-balance", ""),
        taken_at=time.time() if now is None else now,
    )


def from_anthropic_headers(
    headers: Mapping[str, Any], *, now: float | None = None
) -> UsageSnapshot:
    """Uso do Claude Max a partir dos `anthropic-ratelimit-unified-*`.

    A utilização vem em fracção (0.24 = 24%), ao contrário do Codex. Converter aqui é o que
    permite à UI ter uma escala só.
    """
    h = _clean(headers)
    windows: list[Window] = []
    for key, label in (("5h", "5h"), ("7d", "7d")):
        used = _number(h.get(f"anthropic-ratelimit-unified-{key}-utilization"))
        if used is None:
            continue
        windows.append(
            Window(
                label=label,
                used_percent=used * 100.0,
                resets_at=_number(h.get(f"anthropic-ratelimit-unified-{key}-reset")) or 0.0,
            )
        )
    if not windows:
        return UsageSnapshot()
    return UsageSnapshot(windows=tuple(windows), taken_at=time.time() if now is None else now)


def from_headers(
    provider: str, headers: Mapping[str, Any], *, now: float | None = None
) -> UsageSnapshot:
    """O instantâneo do provedor, ou um vazio quando ele não publica nada.

    O Google Antigravity cai aqui: medido contra o proxy, não devolve cabeçalhos de quota.
    Um `UsageSnapshot()` vazio é a resposta honesta — `known` a `False` diz à UI para
    escrever "sem dados", em vez de desenhar uma barra a zero.
    """
    if provider == "openai-codex":
        return from_codex_headers(headers, now=now)
    if provider == "anthropic":
        return from_anthropic_headers(headers, now=now)
    return UsageSnapshot()


# omp: usage/google-antigravity.ts :: RETRIEVE_USER_QUOTA_SUMMARY_PATH
#: O endpoint que a própria UI do Antigravity usa. Ao contrário do catálogo de modelos,
#: reporta as duas janelas mesmo quando nenhuma está esgotada.
QUOTA_SUMMARY_PATH: Final = "/v1internal:retrieveUserQuotaSummary"

#: Grupo cujo uso interessa. A conta reporta vários — "Gemini Models" e
#: "Claude and GPT models" — e somá-los daria um número que não corresponde a limite nenhum.
_PREFERRED_GROUP: Final = "gemini"


# omp: usage/google-antigravity.ts :: classifyWindow
def _classify_window(raw: str) -> str:
    """Nome legível da janela, a partir do `window` ou do `bucketId`."""
    lowered = raw.lower()
    if "week" in lowered or "7d" in lowered:
        return "7d"
    if "5h" in lowered or "five" in lowered:
        return "5h"
    if "day" in lowered or "24h" in lowered:
        return "24h"
    return raw or "?"


def _reset_epoch(raw: object) -> float:
    """`2026-09-23T02:30:06Z` em epoch, ou `0.0` se não vier."""
    if not isinstance(raw, str) or not raw:
        return 0.0
    from datetime import datetime

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


# omp: usage/google-antigravity.ts :: buildQuotaSummaryReport
def from_antigravity_summary(
    payload: Mapping[str, Any], *, now: float | None = None
) -> UsageSnapshot:
    """Uso do Antigravity a partir de `:retrieveUserQuotaSummary`.

    O campo é `remainingFraction` — **o inverso** do que a UI mostra. Um `0.7833` são 21.7%
    usados, e tratá-lo como "usado" mostrava uma conta quase esgotada onde ela está a um
    quinto do limite.

    Escolhe-se o grupo do Gemini: a conta reporta vários ("Gemini Models", "Claude and GPT
    models") e somá-los daria um número que não corresponde a limite nenhum.
    """
    groups = payload.get("groups") or payload.get("quotaGroups") or []
    if not isinstance(groups, list) or not groups:
        return UsageSnapshot()

    chosen: Mapping[str, Any] | None = None
    for group in groups:
        if not isinstance(group, dict):
            continue
        name = str(group.get("displayName") or "").lower()
        if _PREFERRED_GROUP in name:
            chosen = group
            break
        chosen = chosen or group
    if chosen is None:
        return UsageSnapshot()

    windows: list[Window] = []
    for bucket in chosen.get("buckets") or []:
        if not isinstance(bucket, dict):
            continue
        remaining = bucket.get("remainingFraction")
        if not isinstance(remaining, int | float):
            continue
        windows.append(
            Window(
                label=_classify_window(str(bucket.get("window") or bucket.get("bucketId") or "")),
                used_percent=max(0.0, min(100.0, (1.0 - float(remaining)) * 100.0)),
                resets_at=_reset_epoch(bucket.get("resetTime")),
            )
        )
    if not windows:
        return UsageSnapshot()
    return UsageSnapshot(
        windows=tuple(windows),
        plan=str(chosen.get("displayName") or ""),
        taken_at=time.time() if now is None else now,
    )

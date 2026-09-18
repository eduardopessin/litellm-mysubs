"""Liga subscrições (Claude Max, ChatGPT Plus, Google Antigravity) ao LiteLLM.

Instalação numa linha do `config.yaml`::

    litellm_settings:
      callbacks: ["litellm_mysubs.proxy_handler_instance"]

Ou, interactivamente, com `mysubs-setup`.

`proxy_handler_instance` é uma **instância**, não a classe: o proxy recusa uma classe com
`ValueError` no arranque — verificado — porque um callback que não é despachável carregaria
sem queixa e seria ignorado em cada pedido.
"""

from __future__ import annotations

from .callback import MySubs

#: O nome que o `config.yaml` refere. A convenção é do LiteLLM, que a sugere na própria
#: mensagem de erro quando se lhe aponta uma classe.
proxy_handler_instance = MySubs()

__all__ = ["MySubs", "proxy_handler_instance"]

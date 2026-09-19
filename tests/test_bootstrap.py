"""Arranque do plugin: não mexer no que já lá estava.

O risco desta camada não é falhar — é funcionar demais. Um plugin que se instala no caminho
dos pedidos de uma instalação que já serve modelos tem de ser invisível para eles.
"""

from __future__ import annotations

from typing import Any

from litellm_mysubs.bootstrap import Bootstrap, disabled
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.registry import ModelRegistry

CONFIG_DEPLOYMENTS = [
    {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"}},
    {
        "model_name": "meu-llama",
        "litellm_params": {"model": "openai/llama-3", "api_base": "http://gpu.lan:8000"},
    },
]


class EmptyStore:
    owns_refresh = True

    def get(self, provider: str) -> Credential | None:
        return None

    def set(self, provider: str, credential: Credential) -> None: ...

    def delete(self, provider: str) -> None: ...

    def reload(self) -> bool:
        return False


class ConnectedStore(EmptyStore):
    def get(self, provider: str) -> Credential | None:
        if provider == "anthropic":
            return Credential(provider="anthropic", access_token="AT")
        return None


class UnreadableStore(EmptyStore):
    def get(self, provider: str) -> Credential | None:
        raise PermissionError("0644, devia ser 0600")


class FakeRouter:
    def __init__(self) -> None:
        self.model_list: list[dict[str, Any]] = [dict(d) for d in CONFIG_DEPLOYMENTS]

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list


class FakeApp:
    def __init__(self, existing: list[str] | None = None) -> None:
        self.routes = [type("R", (), {"path": p})() for p in (existing or [])]
        self.mounted: list[str] = []

    def mount(self, path: str, app: Any) -> None:
        self.mounted.append(path)
        self.routes.append(type("R", (), {"path": path})())


class TestInertWithoutSubscriptions:
    def test_no_credentials_means_no_patch(self) -> None:
        """Um plugin instalado e sem credenciais tem de ser indistinguível de um plugin
        ausente. Sem subscrição ligada o patch só acrescentaria um wrapper que delega
        sempre no original — custo sem benefício, no caminho de todos os pedidos."""
        boot = Bootstrap()
        assert boot.should_patch(EmptyStore()) is False
        assert boot.patch(EmptyStore()) is False
        assert boot.patched is False

    def test_one_connected_subscription_is_enough(self) -> None:
        assert Bootstrap().should_patch(ConnectedStore()) is True

    def test_an_unreadable_store_is_not_proof_of_credentials(self) -> None:
        """Um ficheiro com permissões largas — que o store recusa, e bem — não pode ser
        lido como "há subscrições ligadas"."""
        assert Bootstrap().should_patch(UnreadableStore()) is False

    def test_the_kill_switch_works_without_editing_yaml(self) -> None:
        """Quando o plugin é o suspeito de um problema, uma variável de ambiente é mais
        rápida e reversível do que desinstalar."""
        assert disabled({"MYSUBS_DISABLE": "1"}) is True
        assert disabled({"MYSUBS_DISABLE": "true"}) is True
        assert disabled({}) is False
        assert disabled({"MYSUBS_DISABLE": "0"}) is False


class TestExistingRoutingSurvives:
    def test_applying_subscription_models_keeps_the_config_ones(self) -> None:
        """`apply` substitui as entradas do plugin. Preservar as do `config.yaml` é o que
        impede um "aplicar" de apagar o roteamento de quem nos instalou."""
        router = FakeRouter()
        ModelRegistry(router=router).apply(
            [
                {
                    "model_name": "claude-opus-5",
                    "litellm_params": {"model": "anthropic/claude-opus-5"},
                }
            ]
        )
        names = [d["model_name"] for d in router.model_list]
        assert "gpt-4o" in names
        assert "meu-llama" in names
        assert "claude-opus-5" in names

    def test_config_deployments_are_returned_untouched(self) -> None:
        """Não basta sobreviverem: têm de sobreviver **iguais**. Uma chave acrescentada a um
        deployment alheio pode mudar o modo de autenticação dele."""
        router = FakeRouter()
        ModelRegistry(router=router).apply([{"model_name": "x", "litellm_params": {"model": "y"}}])
        kept = [d for d in router.model_list if d["model_name"] in ("gpt-4o", "meu-llama")]
        assert kept == CONFIG_DEPLOYMENTS

    def test_only_our_entries_are_replaced_on_reapply(self) -> None:
        router = FakeRouter()
        registry = ModelRegistry(router=router)
        registry.apply([{"model_name": "a", "litellm_params": {"model": "anthropic/a"}}])
        registry.apply([{"model_name": "b", "litellm_params": {"model": "anthropic/b"}}])
        names = [d["model_name"] for d in router.model_list]
        assert names.count("gpt-4o") == 1
        assert "a" not in names
        assert "b" in names


class TestMounting:
    def test_mounting_adds_a_prefix_and_touches_nothing_else(self) -> None:
        app = FakeApp(["/ui", "/swagger", "/v1/models"])
        Bootstrap().mount(app, EmptyStore())
        assert app.mounted == ["/mysubs"]
        assert [r.path for r in app.routes][:3] == ["/ui", "/swagger", "/v1/models"]

    def test_an_occupied_prefix_is_not_mounted_over(self) -> None:
        """Duas sub-apps no mesmo prefixo: a segunda apanha os pedidos e a primeira fica
        inalcançável, sem erro nenhum a dizê-lo."""
        app = FakeApp(["/mysubs"])
        boot = Bootstrap()
        assert boot.mount(app, EmptyStore()) is False
        assert app.mounted == []

    def test_mounting_twice_is_a_noop(self) -> None:
        app = FakeApp()
        boot = Bootstrap()
        boot.mount(app, EmptyStore())
        boot.mount(app, EmptyStore())
        assert app.mounted == ["/mysubs"]


class TestProxyContract:
    """O que o proxy exige de um callback. Verificado contra o LiteLLM instalado."""

    def test_the_exported_handler_is_an_instance_not_a_class(self) -> None:
        """O proxy recusa uma classe com `ValueError` e **não arranca**.

        Medido: `callbacks: ["litellm_mysubs.MySubs"]` fazia `load_config` levantar e o
        processo sair com código 3. A mensagem do LiteLLM nomeia a correcção — exportar uma
        instância — porque um callback não despachável carregaria sem queixa e seria
        ignorado em cada pedido.
        """
        import litellm_mysubs

        assert not isinstance(litellm_mysubs.proxy_handler_instance, type)

    def test_the_handler_is_a_custom_logger(self) -> None:
        """`isinstance(loaded, CustomLogger)` é a condição de aceitação do proxy. Um objecto
        com os hooks certos mas sem a herança é recusado no arranque — a herança não é
        estilo, é contrato."""
        from litellm.integrations.custom_logger import CustomLogger

        import litellm_mysubs

        assert isinstance(litellm_mysubs.proxy_handler_instance, CustomLogger)

    def test_the_installer_writes_the_instance_path(self) -> None:
        """O caminho no `config.yaml` tem de ser o que o proxy aceita, senão o instalador
        deixa a instalação num estado que não arranca."""
        from litellm_mysubs.setup_cli import CALLBACK_PATH

        assert CALLBACK_PATH.endswith("proxy_handler_instance")


class TestStoreIsWired:
    """O patch sem store é um patch inútil — e o sintoma aparece longe da causa."""

    def test_patching_configures_the_plugin_store(self) -> None:
        """Medido no proxy real: `install()` sem `configure()` deixa `_access_token` a
        devolver "", o pedido sai com `Authorization: Bearer ` e o httpx recusa-o com
        `Illegal header value b'Bearer '`. O erro chega do cliente do provedor, não do
        plugin, e não diz nada sobre a credencial em falta.
        """
        from litellm_mysubs import plugin

        store = ConnectedStore()
        boot = Bootstrap()
        try:
            assert boot.patch(store) is True
            assert plugin._state.store is store
        finally:
            plugin.uninstall()

    def test_the_router_entrypoint_is_patched(self) -> None:
        """O proxy encaminha por `Router.acompletion`, não pelas funções de módulo.

        `route_llm_request.py:487` faz `getattr(llm_router, route_type)(**data)`. Sem este
        patch, um modelo de subscrição resolvia o deployment e ia para o cliente nativo
        antes de qualquer função de módulo ser tocada. O `sitecustomize.py` original diz-o
        no comentário da linha 2812; o porte só patchava as funções de módulo.
        """
        from litellm.router import Router

        from litellm_mysubs import plugin

        original = Router.acompletion
        try:
            plugin.configure(store=ConnectedStore())
            plugin.install()
            assert Router.acompletion is not original
            assert Router.acompletion.__name__ == "_wrapped_router_acompletion"
        finally:
            plugin.uninstall()
        assert Router.acompletion is original

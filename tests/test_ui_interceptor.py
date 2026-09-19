"""The token interceptor seen from the outside: button, pairing code, deposit.

What these tests protect is the reason the feature exists. The callback server has to open
on the browser's machine, which is not the proxy's when it runs in a container or a
cluster; the deposit is what closes that distance. If the deposit route ends up behind the
administrator guard, or if the client gets to choose the provider, the feature becomes
either decorative or dangerous — and neither failure is visible to the eye.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.ui import mount
from litellm_mysubs.ui.service import MySubsService


class Store:
    """Store with the real signature: `set(provider, credential)`."""

    owns_refresh = True

    def __init__(self) -> None:
        self.creds: dict[str, Credential] = {}

    def get(self, provider: ProviderId) -> Credential | None:
        return self.creds.get(provider)

    def set(self, provider: ProviderId, credential: Credential) -> None:
        self.creds[provider] = credential

    def delete(self, provider: ProviderId) -> None:
        self.creds.pop(provider, None)

    def reload(self) -> bool:
        return False

    def connected(self) -> list[ProviderId]:
        return list(self.creds)  # type: ignore[arg-type]


class Router:
    def __init__(self) -> None:
        self.model_list: list[dict[str, Any]] = []

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list


def build(
    *, guard: Any | None = None, store: Store | None = None
) -> tuple[TestClient, MySubsService, Store]:
    used = store or Store()
    router = Router()
    service = MySubsService(store=used, router_source=lambda: router)
    app = FastAPI()
    mount(app, service, guard=guard)
    return TestClient(app), service, used


def payload(code: str, token: str = "new-token") -> dict[str, Any]:
    return {"pairing_code": code, "credential": {"access_token": token, "refresh_token": "r"}}


def issue(client: TestClient, provider: str = "anthropic") -> str:
    """Issue a code and return it, extracted from the command the page shows."""
    body = client.post(f"/mysubs/pair/{provider}").text
    marker = "--code "
    start = body.index(marker) + len(marker)
    return body[start:].split("<")[0].strip()


class TestPairingButton:
    def test_the_connect_card_offers_the_interceptor(self) -> None:
        """Without the button, whoever runs the proxy in a cluster is left with the paste
        alone and no idea there was another path."""
        client, _, _ = build()
        body = client.get("/mysubs/").text
        assert "pair/anthropic" in body
        assert "interceptor" in body.lower()

    def test_issuing_shows_a_command_ready_to_paste(self) -> None:
        """The user must not have to compose the proxy URL nor the provider's internal id:
        every piece composed by hand is a piece that only fails after the login."""
        client, _, _ = build()
        body = client.post("/mysubs/pair/openai-codex").text
        assert "mysubs-login openai-codex" in body
        assert "--code " in body
        assert "--url " in body

    def test_the_code_never_travels_in_the_url(self) -> None:
        """A code in the query string ends up in the history, in the front proxy's logs and
        in the `Referer` of the next request. That is why the route answers 200 with the
        page, and not a 303 with the code in the destination."""
        client, _, _ = build()
        response = client.post("/mysubs/pair/anthropic", follow_redirects=False)
        assert response.status_code == 200

    def test_buttons_work_from_a_page_served_below_the_root(self) -> None:
        """The pairing page is served at `/mysubs/pair/<p>`, not at the root.

        With a relative `action`, the browser resolves `pair/x` against `/mysubs/pair/` and
        requests `/mysubs/pair/pair/x` — 404. The bug showed up nowhere but here, because
        this is the only 200 response served outside the root: the other buttons redirect.
        """
        client, _, _ = build()
        body = client.post("/mysubs/pair/anthropic").text
        for action in re.findall(r'action="([^"]+)"', body):
            assert action.startswith("/mysubs/"), f"relative action: {action!r}"

    def test_every_button_posts_to_a_route_that_exists(self) -> None:
        """An `action` pointing at a 404 is a dead button, and the user has no way to tell
        that apart from a provider failure.

        Without following redirects: `connect` answers 303 to the **provider**, and following
        it would measure Anthropic instead of this app.
        """
        client, _, _ = build()
        body = client.get("/mysubs/").text
        for action in sorted(set(re.findall(r'action="([^"]+)"', body))):
            status = client.post(action, data={"pasted": "x"}, follow_redirects=False).status_code
            assert status != 404, action

    def test_an_unknown_provider_is_refused(self) -> None:
        client, _, _ = build()
        assert client.post("/mysubs/pair/made-up").status_code == 404


class TestDeposit:
    def test_a_paired_deposit_connects_the_provider(self) -> None:
        client, _, store = build()
        code = issue(client)
        response = client.post("/mysubs/api/deposit", json=payload(code))
        assert response.status_code == 200
        assert store.creds["anthropic"].access_token == "new-token"

    def test_the_code_decides_the_provider_not_the_caller(self) -> None:
        """Letting the body name the provider would turn a code issued for Codex into a write
        over Anthropic's credential.

        It is attempted in both places a `provider` would fit — next to the code and inside
        the credential itself — because only the first is obvious, and it is the second that
        reaches `from_payload`.
        """
        client, _, store = build()
        code = issue(client, "openai-codex")
        body = payload(code)
        body["provider"] = "anthropic"
        body["credential"]["provider"] = "anthropic"
        assert client.post("/mysubs/api/deposit", json=body).json() == {
            "provider": "openai-codex"
        }
        assert "anthropic" not in store.creds
        assert store.creds["openai-codex"].access_token == "new-token"
        assert store.creds["openai-codex"].provider == "openai-codex"

    def test_a_code_works_once(self) -> None:
        """A spent code is a replay: it stays in the terminal scrollback of whoever ran it."""
        client, _, _ = build()
        code = issue(client)
        assert client.post("/mysubs/api/deposit", json=payload(code)).status_code == 200
        assert client.post("/mysubs/api/deposit", json=payload(code)).status_code == 403

    def test_an_invented_code_is_refused(self) -> None:
        client, _, store = build()
        response = client.post("/mysubs/api/deposit", json=payload("XXXX-XXXX-XXXX"))
        assert response.status_code == 403
        assert store.creds == {}

    def test_a_deposit_without_a_token_is_refused(self) -> None:
        """A credential with no `access_token` is not degraded, it is absent: storing it left
        the card saying "connected" about nothing."""
        client, _, store = build()
        code = issue(client)
        response = client.post(
            "/mysubs/api/deposit", json={"pairing_code": code, "credential": {}}
        )
        assert response.status_code == 403
        assert store.creds == {}

    def test_deposit_is_reachable_without_the_admin_guard(self) -> None:
        """This is the test that justifies the separate router. With the dependency declared
        on the app, the deposit answered 403 and the interceptor could never hand over what
        it had fetched — with the rest of the page working, hence with no signal at all."""

        def deny() -> None:
            raise AssertionError("the guard should not run on the deposit")

        from fastapi import Depends

        client, service, store = build(guard=Depends(deny))
        pairing = service.pair("anthropic")
        response = client.post("/mysubs/api/deposit", json=payload(pairing.code))
        assert response.status_code == 200
        assert store.creds["anthropic"].access_token == "new-token"

    def test_the_rest_of_the_page_stays_guarded(self) -> None:
        """The corollary: opening the deposit must not have opened the page."""
        from fastapi import Depends, HTTPException

        def deny() -> None:
            raise HTTPException(status_code=403, detail="no")

        client, _, _ = build(guard=Depends(deny))
        assert client.get("/mysubs/").status_code == 403
        assert client.post("/mysubs/pair/anthropic").status_code == 403


class TestReconnect:
    def test_a_deposit_clears_the_previous_discovery(self) -> None:
        """A reconnected subscription may serve different models. The previous list is no
        longer measured against this credential, and showing it would present as verified
        something that was not."""
        from litellm_mysubs.catalog.discovery import DiscoveredModel

        client, service, _ = build()
        service.discovered["anthropic"] = [
            DiscoveredModel(
                wire_name="claude-opus-4-8",
                suggested_name="claude-opus",
                verified=True,
                note="",
            )
        ]
        code = issue(client)
        client.post("/mysubs/api/deposit", json=payload(code))
        assert "anthropic" not in service.discovered


class TestBackgroundRefresher:
    """The refresher has to start with the proxy, not with the first request.

    These tests exist because of a measured bug: the first version registered the `lifespan`
    on the sub-app, and Starlette **does not propagate it to sub-apps mounted** with
    `app.mount()`. Nothing failed — the page served, the tests passed — and the tokens
    expired all the same. A background feature that does not run is indistinguishable from
    one that runs, except by the symptom it was supposed to have prevented.
    """

    def test_mounting_registers_the_refresher_on_the_host_app(self) -> None:
        from fastapi.testclient import TestClient

        from litellm_mysubs.ui.install import install, shared_service

        host = FastAPI()
        install(host, store=Store())
        service = shared_service()
        assert service is not None
        assert service.refresher is None, "must not start before the server"

        with TestClient(host):
            assert service.refresher is not None, "the proxy startup did not start the refresher"
            assert service.refresher.running

    def test_the_refresher_stops_with_the_proxy(self) -> None:
        """A task that survives the shutdown holds the process and prevents the restart."""
        from fastapi.testclient import TestClient

        from litellm_mysubs.ui.install import install, shared_service

        host = FastAPI()
        install(host, store=Store())
        service = shared_service()
        assert service is not None
        with TestClient(host):
            pass
        assert not service.refresher.running


class TestDisconnect:
    """Disconnecting is the only action on the page that destroys something unrecoverable.

    A deleted refresh token only comes back with a fresh login at the provider — there is no
    local copy and no undo. That is what justifies the confirmation in the request body, and
    not just the browser's `confirm()`: that one does not survive a form resubmission.
    """

    def connected(self) -> Store:
        store = Store()
        for provider in ("anthropic", "openai-codex"):
            store.creds[provider] = Credential(
                provider=provider, access_token="a", refresh_token="r"
            )
        return store

    def test_disconnecting_removes_the_credential(self) -> None:
        client, _, store = build(store=self.connected())
        response = client.post(
            "/mysubs/disconnect/anthropic", data={"confirm": "anthropic"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "anthropic" not in store.creds
        assert "openai-codex" in store.creds, "disconnected a provider that was not asked for"

    def test_a_missing_confirmation_changes_nothing(self) -> None:
        """A stray POST — a double click, a refresh that resubmits — must not delete a token
        that only a fresh login restores."""
        client, _, store = build(store=self.connected())
        client.post("/mysubs/disconnect/anthropic", follow_redirects=False)
        assert "anthropic" in store.creds

    def test_a_confirmation_for_another_provider_is_refused(self) -> None:
        client, _, store = build(store=self.connected())
        client.post(
            "/mysubs/disconnect/anthropic", data={"confirm": "openai-codex"},
            follow_redirects=False,
        )
        assert "anthropic" in store.creds

    def test_disconnecting_pulls_the_models_out_of_the_router(self) -> None:
        """The worst possible state is models advertised in `/v1/models` with no credential to
        serve them: every request would fail with a 401 from the upstream and the user would
        not connect the cause to the button they pressed."""
        from litellm_mysubs.catalog.discovery import DiscoveredModel

        store = self.connected()
        client, service, _ = build(store=store)
        service.discovered["anthropic"] = [
            DiscoveredModel(
                wire_name="claude-opus-4-8", suggested_name="claude-opus", verified=True, note=""
            )
        ]
        service.apply("anthropic", ["claude-opus"])
        router = service.router_source()
        assert len(router.model_list) == 1

        client.post(
            "/mysubs/disconnect/anthropic", data={"confirm": "anthropic"},
            follow_redirects=False,
        )
        assert router.model_list == [], "the models were left orphaned in the Router"

    def test_the_card_offers_the_button_only_when_connected(self) -> None:
        client, _, _ = build(store=self.connected())
        body = client.get("/mysubs/").text
        assert "/mysubs/disconnect/anthropic" in body
        assert "/mysubs/disconnect/google-antigravity" not in body, (
            "offered to disconnect a provider that was never connected"
        )

    def test_reconnecting_after_a_disconnect_works(self) -> None:
        """Disconnecting must not leave state that prevents connecting again."""
        client, service, store = build(store=self.connected())
        client.post(
            "/mysubs/disconnect/anthropic", data={"confirm": "anthropic"},
            follow_redirects=False,
        )
        pairing = service.pair("anthropic")
        assert client.post("/mysubs/api/deposit", json=payload(pairing.code)).status_code == 200
        assert store.creds["anthropic"].access_token == "new-token"


class TestRenewButton:
    """The renew button only exists where the background refresher does not reach.

    Two symmetric failures to avoid: showing it always teaches the user they have to watch
    something already handled; hiding it always leaves with no action at all precisely the
    card that needs one — a token of unknown expiry is never renewed by the loop.
    """

    def store_with(self, expires_at: float) -> Store:
        store = Store()
        store.creds["anthropic"] = Credential(
            provider="anthropic", access_token="a", refresh_token="r", expires_at=expires_at
        )
        return store

    def test_a_healthy_token_hides_the_button(self) -> None:
        import time

        client, _, _ = build(store=self.store_with(time.time() + 7200))
        body = client.get("/mysubs/").text
        assert "/mysubs/refresh/anthropic" not in body
        assert "renews automatically" in body

    def test_an_unknown_expiry_still_offers_the_button(self) -> None:
        """`expires_at <= 0` means "unknown", and the loop gives up on purpose: treating it as
        expired would burn a single-use refresh token on every sweep. Here the manual path is
        the only way out."""
        client, _, _ = build(store=self.store_with(0.0))
        body = client.get("/mysubs/").text
        assert "/mysubs/refresh/anthropic" in body
        assert "renews automatically" not in body

    def test_a_store_that_is_not_the_owner_offers_the_button(self) -> None:
        """`owns_refresh=False` makes the sweep give up entirely — the owner is another
        process. The manual path remains the only action possible from here."""
        import time

        store = self.store_with(time.time() + 7200)
        store.owns_refresh = False
        client, _, _ = build(store=store)
        body = client.get("/mysubs/").text
        assert "/mysubs/refresh/anthropic" in body
        assert "renews automatically" not in body

    def test_the_route_still_works_when_the_button_is_hidden(self) -> None:
        """Hiding is a page decision, not removal of the capability: whoever automates against
        the route must not lose it over a presentation choice."""
        import time

        client, _, _ = build(store=self.store_with(time.time() + 7200))
        assert client.post("/mysubs/refresh/anthropic", follow_redirects=False).status_code == 303


class TestAutomaticConnect:
    """Connecting without pasting anything, when the proxy runs on the browser's machine.

    What makes this possible, and what does not: reading the URL of the window that failed
    is impossible (`SecurityError`, different origin). What does work is the page talking to
    the callback port over `fetch`, **provided it sends CORS** — and that is how it learns
    the login finished.
    """

    def test_connect_returns_the_url_as_json_when_asked(self) -> None:
        """The page needs the URL without leaving: a 303 took the browser away and the probing
        died with the page."""
        client, _, _ = build()
        data = client.post("/mysubs/connect/anthropic?url=1").json()
        assert data["url"].startswith("https://claude.ai/oauth/authorize")
        assert "local" in data

    def test_connect_without_the_flag_still_redirects(self) -> None:
        """The no-JavaScript path must not disappear: it is what makes the button work when
        none of this runs."""
        client, _, _ = build()
        response = client.post("/mysubs/connect/anthropic", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"].startswith("https://claude.ai/")

    def test_a_busy_local_port_does_not_break_connect(self) -> None:
        """On a remote proxy — or with the port taken — the local server does not open. That
        must not block the flow: the user still goes to the provider, and the paste is still
        the way out. The port is really bound instead of the method being patched, because
        that is how the failure happens."""
        import socket

        holder = socket.socket()
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            holder.bind(("127.0.0.1", 54545))
            holder.listen(1)
        except OSError:
            # Already taken by another process — which is exactly the condition under test.
            holder.close()
            holder = None  # type: ignore[assignment]
        try:
            client, _, _ = build()
            data = client.post("/mysubs/connect/anthropic?url=1").json()
            assert data["url"].startswith("https://claude.ai/"), "the login has to go on"
            assert data["local"] is False, "claimed it opened a port that is taken"
        finally:
            if holder is not None:
                holder.close()

    def test_the_page_keeps_the_paste_as_a_way_out(self) -> None:
        """Hiding the paste is a JavaScript decision taken when it detects the interceptor.
        The served HTML has to carry it always: whoever has no JS would be left with no way
        to connect."""
        client, _, _ = build()
        body = client.get("/mysubs/").text
        assert "/mysubs/paste/anthropic" in body
        assert 'name="pasted"' in body


class TestPasteIsAlwaysReachable:
    """The paste box must not depend on anything to exist.

    Measured: no browser mechanism allows capturing the return URL from another origin — 16
    attempts across `window.open` (11 variants), `iframe` (3), the Clipboard API and the
    Performance API, all with `SecurityError`. The three providers further refuse to be
    framed (`X-Frame-Options: SAMEORIGIN`/`DENY`).

    So, on a proxy that does not run on the browser's machine, pasting is the **only** path.
    An earlier version hid it whenever it detected a receiver; if the detection was right
    and the flow failed afterwards, the user was left with no way out mid-login.
    """

    def test_the_paste_box_is_in_the_served_html(self) -> None:
        client, _, _ = build()
        body = client.get("/mysubs/").text
        for provider in ("anthropic", "openai-codex", "google-antigravity"):
            assert f"/mysubs/paste/{provider}" in body, provider
        assert body.count('name="pasted"') == 3

    def test_the_paste_box_is_not_hidden_behind_a_disclosure(self) -> None:
        """Inside a closed `details`, whoever does not know it exists will not find it — and
        that is exactly whoever needs it most."""
        import re

        body = client_body = build()[0].get("/mysubs/").text
        for match in re.finditer(r"<details[^>]*>(.*?)</details>", client_body, re.S):
            assert 'name="pasted"' not in match.group(1), "the box went back inside the details"
        assert "paste" in body.lower()

    def test_the_paste_route_works_without_any_local_receiver(self) -> None:
        """The path has to work with no port open anywhere: that is the remote proxy scenario,
        which is the normal one in a cluster install."""
        client, service, _ = build()
        service.begin("anthropic")
        response = client.post(
            "/mysubs/paste/anthropic", data={"pasted": "garbage"}, follow_redirects=False
        )
        # Refuses with a message, not a 404 nor a 500: the route exists and explains what
        # failed.
        assert response.status_code == 303
        assert "error=" in response.headers["location"]


class TestEveryStyledClassHasARule:
    """Every class used in the HTML has a CSS rule, and vice versa.

    It exists because of a real defect: an edit to the icon rules landed **on top of** the
    `.card` rule, and the symptom was subtle — the cards were still in the HTML, still
    separated by whitespace, they had only lost their outline. Nothing failed, no test
    complained, and it was only seen in a screenshot. The `.lead` rule had disappeared in
    that same edit without anyone noticing.
    """

    def styled(self) -> tuple[set[str], set[str]]:
        import re

        from litellm_mysubs.ui.app import _SHELL

        html = build()[0].get("/mysubs/").text
        # Without the `<script>`s: inside them there is `class="' + cls + '"` built in
        # JavaScript, and the regex picked up pieces of the expression as if they were
        # classes.
        marked = re.sub(r"<script>.*?</script>", "", html, flags=re.S)
        used = set()
        for attr in re.findall(r'class="([^"]+)"', marked):
            used.update(attr.split())
        # Only the class rules from our own CSS; `{{` because the shell is a template.
        defined = set(re.findall(r"\.([a-z][a-z0-9-]*)\{\{", _SHELL))
        return used, defined

    def test_no_class_in_the_page_is_left_without_style(self) -> None:
        used, defined = self.styled()
        # Classes that exist only as a JavaScript hook, with no look of their own.
        hooks = {"paste"}
        orphans = {c for c in used if c not in defined} - hooks
        assert not orphans, f"classes with no CSS rule: {sorted(orphans)}"

    def test_the_card_frame_survives(self) -> None:
        """The frame is what separates one provider from the next: without it the page is a
        run-on list."""
        from litellm_mysubs.ui.app import _SHELL

        assert ".card{{" in _SHELL, "the base card rule disappeared"
        rule = _SHELL.split(".card{{", 1)[1].split("}}", 1)[0]
        assert "border:" in rule, f"the card lost its outline: {rule}"
        assert "border-radius:" in rule, f"the card lost its corners: {rule}"


class TestThePageIsInEnglish:
    """The package lives inside LiteLLM's UI, which is monolingual in English. A string
    left behind in Portuguese breaks nothing — it just looks like a botched install, and no
    other test catches it.

    Outside the `<script>` and `<style>` blocks: inside those there is only markup
    machinery, which never reaches anyone's eyes.
    """

    #: Words that exist only in Portuguese and that appear in prose, not in ids nor in
    #: provider names. Markers, not a complete vocabulary: one is enough to give the whole
    #: sentence away.
    MARKERS = ("não", "subscrição", "código", "página", "máquina", "está", "ligar")

    def visible(self, html: str) -> str:
        without_script = re.sub(r"<script>.*?</script>", "", html, flags=re.S)
        return re.sub(r"<style>.*?</style>", "", without_script, flags=re.S)

    def found(self, html: str) -> list[str]:
        text = self.visible(html).lower()
        return [w for w in self.MARKERS if re.search(rf"\b{w}\b", text)]

    def test_the_connect_card_has_no_portuguese_left(self) -> None:
        body = build()[0].get("/mysubs/").text
        assert not self.found(body), self.found(body)

    def test_the_pairing_page_has_no_portuguese_left(self) -> None:
        body = build()[0].post("/mysubs/pair/anthropic").text
        assert not self.found(body), self.found(body)

    def test_the_connected_card_has_no_portuguese_left(self) -> None:
        """The connected state has prose the unconnected card does not show: expiry, chips,
        discovery and the disconnect block."""
        from litellm_mysubs.catalog.discovery import DiscoveredModel

        store = Store()
        store.creds["openai-codex"] = Credential(
            provider="openai-codex", access_token="a", refresh_token="r"
        )
        client, service, _ = build(store=store)
        service.discovered["openai-codex"] = [
            DiscoveredModel(wire_name="gpt-6", suggested_name="gpt-6", verified=False, note="n")
        ]
        body = client.get("/mysubs/").text
        assert not self.found(body), self.found(body)

    def test_the_refusal_page_has_no_portuguese_left(self) -> None:
        """The refusal is the page that teaches the most, and the one fewest people read
        carefully."""
        from litellm_mysubs.ui.app import _error_page

        body = _error_page(403, "forbidden")
        assert not self.found(body), self.found(body)

    def test_no_portuguese_prose_is_left_in_the_source(self) -> None:
        """The repository is published in English — comments and docstrings included.

        The tests above guard what the browser renders. This one guards the rest: a
        comment left in Portuguese does not break anything and no other test would ever
        notice, which is exactly how the last one survived three passes.

        Deliberate exceptions are listed, not silenced: upstream error text asserted
        verbatim by a test is the thing under test, and translating it would make a
        passthrough indistinguishable from a message of our own.
        """
        import ast

        root = Path(__file__).resolve().parent.parent
        allowed = {
            # Fixtures that assert an upstream message reaches the user unrewritten.
            root / "tests" / "test_credentials_oauth.py",
            # This file: MARKERS below is the Portuguese it is looking for.
            Path(__file__).resolve(),
        }
        accents = re.compile(r"[àáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ]")

        offenders: list[str] = []
        for path in sorted(root.glob("src/**/*.py")) + sorted(root.glob("tests/**/*.py")):
            if path in allowed:
                continue
            text = path.read_text("utf-8")
            tree = ast.parse(text)
            docstrings = {
                doc
                for node in ast.walk(tree)
                if isinstance(
                    node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                )
                and (doc := ast.get_docstring(node, clean=False)) is not None
            }
            for number, line in enumerate(text.splitlines(), start=1):
                stripped = line.lstrip()
                is_comment = stripped.startswith("#")
                in_docstring = any(line.strip() in doc for doc in docstrings if line.strip())
                if (is_comment or in_docstring) and accents.search(line):
                    offenders.append(f"{path.relative_to(root)}:{number}: {stripped[:70]}")

        assert not offenders, "Portuguese prose left in the source:\n" + "\n".join(offenders)


class TestSelectionSurvivesRestart:
    """The applied models have to survive a proxy restart.

    Measured before persistence existed, against the real proxy: apply `gpt-5.5`, restart,
    and `/v1/models` was back to holding only `eco` — with the cards saying
    `connected=True applied=0`. The credentials survived (0600 file); the model choice was
    not stored anywhere.

    The `registry.py` docstring already promised otherwise: "injects directly **and persists
    on its own**". It was half a sentence.
    """

    def service_with(self, tmp_path: Any) -> Any:
        from litellm_mysubs.catalog.discovery import DiscoveredModel
        from litellm_mysubs.catalog.selection import SelectionStore

        router = Router()
        service = MySubsService(
            store=Store(),
            router_source=lambda: router,
            selections=SelectionStore(tmp_path / "models.json"),
        )
        service.discovered["openai-codex"] = [
            DiscoveredModel(wire_name="gpt-6", suggested_name="gpt-6", verified=True, note="")
        ]
        return service, router

    def test_applying_writes_the_selection_to_disk(self, tmp_path: Any) -> None:
        service, _ = self.service_with(tmp_path)
        service.apply("openai-codex", ["gpt-6"])
        saved = service.selections.all()
        assert [s.provider for s in saved] == ["openai-codex"]
        assert saved[0].deployments[0]["model_name"] == "mysubs/codex/gpt-6"

    def test_a_fresh_process_reapplies_what_was_saved(self, tmp_path: Any) -> None:
        """The restart case: fresh process, empty Router, with no discovery run."""
        from litellm_mysubs.catalog.selection import SelectionStore

        service, _ = self.service_with(tmp_path)
        service.apply("openai-codex", ["gpt-6"])

        # Fresh process: nothing in memory, only the file.
        new_router = Router()
        new = MySubsService(
            store=Store(),
            router_source=lambda: new_router,
            selections=SelectionStore(tmp_path / "models.json"),
        )
        assert new.discovered == {}, "premise: the catalogue does not survive"
        assert new.reapply() == 1
        assert [d["model_name"] for d in new_router.model_list] == ["mysubs/codex/gpt-6"]

    def test_reapply_does_not_need_the_network(self, tmp_path: Any) -> None:
        """Going to the network at startup would leave the user without their models whenever
        the upstream was down. The stored deployments are self-sufficient."""
        from litellm_mysubs.catalog.selection import SelectionStore

        service, _ = self.service_with(tmp_path)
        service.apply("openai-codex", ["gpt-6"])

        def explode() -> Any:
            raise AssertionError("startup went to the network")

        new_router = Router()
        new = MySubsService(
            store=Store(),
            router_source=lambda: new_router,
            client_factory=explode,
            selections=SelectionStore(tmp_path / "models.json"),
        )
        assert new.reapply() == 1

    def test_disconnecting_also_forgets_the_saved_models(self, tmp_path: Any) -> None:
        """Without this the next startup re-injected the models of a disconnected
        subscription, and the card said 'not connected' with its models in `/v1/models`."""
        from litellm_mysubs.catalog.selection import SelectionStore

        service, _ = self.service_with(tmp_path)
        service.store.creds["openai-codex"] = Credential(
            provider="openai-codex", access_token="a", refresh_token="r"
        )
        service.apply("openai-codex", ["gpt-6"])
        service.disconnect("openai-codex")

        new_router = Router()
        new = MySubsService(
            store=Store(),
            router_source=lambda: new_router,
            selections=SelectionStore(tmp_path / "models.json"),
        )
        assert new.reapply() == 0
        assert new_router.model_list == []

    def test_reapply_preserves_models_from_the_config(self, tmp_path: Any) -> None:
        """The user's `config.yaml` is not ours to touch."""
        from litellm_mysubs.catalog.selection import SelectionStore

        service, router = self.service_with(tmp_path)
        router.model_list = [{"model_name": "eco", "litellm_params": {"model": "openai/eco"}}]
        service.apply("openai-codex", ["gpt-6"])

        names = {d["model_name"] for d in router.model_list}
        assert "eco" in names, "the model from the config was deleted"
        assert "mysubs/codex/gpt-6" in names

        new_router = Router()
        new_router.model_list = [{"model_name": "eco", "litellm_params": {"model": "openai/eco"}}]
        new = MySubsService(
            store=Store(),
            router_source=lambda: new_router,
            selections=SelectionStore(tmp_path / "models.json"),
        )
        new.reapply()
        assert {d["model_name"] for d in new_router.model_list} == {"eco", "mysubs/codex/gpt-6"}


class TestApplyFeedback:
    """Applying reports what changed, and returns to the card it came from.

    Without an anchor, a `303` reloads the page at the top: whoever pressed a button on the
    third card loses sight of the card and of the message they just caused.
    """

    def ready(self, tmp_path: Any) -> Any:
        from litellm_mysubs.catalog.discovery import DiscoveredModel
        from litellm_mysubs.catalog.selection import SelectionStore

        store = Store()
        store.creds["openai-codex"] = Credential(
            provider="openai-codex", access_token="a", refresh_token="r"
        )
        client, service, _ = build(store=store)
        service.selections = SelectionStore(tmp_path / "models.json")
        service.discovered["openai-codex"] = [
            DiscoveredModel(wire_name=n, suggested_name=n, verified=True, note="")
            for n in ("gpt-6", "gpt-7")
        ]
        return client, service

    def test_apply_reports_what_was_added(self, tmp_path: Any) -> None:
        _, service = self.ready(tmp_path)
        result = service.apply("openai-codex", ["gpt-6"])
        assert result.added == ["mysubs/codex/gpt-6"]
        assert result.removed == []
        assert result.changed

    def test_apply_reports_what_was_removed(self, tmp_path: Any) -> None:
        """This is the question the user has: did what I unchecked really go away?"""
        _, service = self.ready(tmp_path)
        service.apply("openai-codex", ["gpt-6", "gpt-7"])
        result = service.apply("openai-codex", ["gpt-6"])
        assert result.removed == ["mysubs/codex/gpt-7"]
        assert result.added == []

    def test_applying_the_same_selection_reports_no_change(self, tmp_path: Any) -> None:
        """A dialog saying 'added' about what was already there would be a lie."""
        _, service = self.ready(tmp_path)
        service.apply("openai-codex", ["gpt-6"])
        result = service.apply("openai-codex", ["gpt-6"])
        assert not result.changed

    def test_the_redirect_anchors_on_the_card(self, tmp_path: Any) -> None:
        client, _ = self.ready(tmp_path)
        response = client.post(
            "/mysubs/apply/openai-codex", data={"chosen": "gpt-6"}, follow_redirects=False
        )
        target = response.headers["location"]
        assert target.endswith("#openai-codex"), target
        assert "added=" in target, target

    def test_every_action_returns_to_its_card(self, tmp_path: Any) -> None:
        """Not only apply: any button that reloads the page."""
        client, _ = self.ready(tmp_path)
        for route in ("discover", "refresh"):
            target = client.post(
                f"/mysubs/{route}/openai-codex", follow_redirects=False
            ).headers["location"]
            assert target.endswith("#openai-codex"), f"{route}: {target}"

    def test_an_error_also_returns_to_its_card(self, tmp_path: Any) -> None:
        client, _ = self.ready(tmp_path)
        target = client.post(
            "/mysubs/apply/openai-codex", data={"chosen": "made-up"}, follow_redirects=False
        ).headers["location"]
        assert "error=" in target
        assert target.endswith("#openai-codex"), target

    def test_the_card_carries_the_anchor_target(self, tmp_path: Any) -> None:
        """The anchor only works if the card carries the matching `id`."""
        client, _ = self.ready(tmp_path)
        body = client.get("/mysubs/").text
        for provider in ("anthropic", "openai-codex", "google-antigravity"):
            assert f'id="{provider}"' in body, provider

    def test_rediscover_sits_next_to_apply(self, tmp_path: Any) -> None:
        """In separate forms the browser put them on different lines, and «Rediscover» looked
        like it belonged to the next block."""
        client, _ = self.ready(tmp_path)
        body = client.get("/mysubs/").text
        actions = body.split('<div class="actions">', 1)
        assert len(actions) == 2, "the actions block disappeared"
        block = actions[1].split("</div>", 1)[0]
        assert "Apply" in block and "Rediscover" in block, block

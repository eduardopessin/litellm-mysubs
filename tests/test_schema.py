"""Saneamento de schema para o Cloud Code Assist.

Os valores esperados aqui foram apurados por diferenciação contra
``normalizeSchemaForCCA`` do `@oh-my-pi/pi-ai` 18.2.6 a correr em Node (3167 casos,
incluindo fuzz aleatório, todos idênticos). Não são o que se acha que o OMP faz.
"""

from __future__ import annotations

from typing import Any

from litellm_mysubs.wire.schema import CCA_FALLBACK_SCHEMA, normalize_for_cca


def test_schema_simples_atravessa_sem_alteracao() -> None:
    """Se um schema trivial não atravessa intacto, todas as ferramentas perdem a tipagem
    dos argumentos por causa de saneamento que nada tinha para sanear."""
    schema = {
        "type": "object",
        "properties": {"caminho": {"type": "string", "description": "Caminho"}},
        "required": ["caminho"],
    }
    assert normalize_for_cca(schema) == schema


def test_type_array_com_null_vira_tipo_escalar() -> None:
    """``type: ["string","null"]`` não tem representação no proto do CCA: o campo ``type``
    é escalar. Deixá-lo passar como lista fá-lo-ia cair no fallback e a ferramenta perderia
    toda a tipagem por causa de um único argumento opcional."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"a": {"type": ["string", "null"]}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
    )
    assert out["properties"]["a"] == {"type": "string"}


def test_anyof_com_ramo_null_pelado_liberta_o_campo_do_required() -> None:
    """A forma idiomática do Zod para "opcional" é ``anyOf: [T, {type: null}]``.

    O CCA não sabe exprimir nulabilidade, mas sabe exprimir *ausência*: o ramo ``null``
    desaparece e o campo sai do ``required``. Mantê-lo obrigatório forçaria o modelo a
    inventar um valor para um argumento que o cliente declarou poder faltar.
    """
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {
                "a": {"anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]}
            },
            "required": ["a", "b"],
        }
    )
    assert out["properties"]["a"] == {"type": "string", "enum": ["x", "y"]}
    assert out["required"] == ["b"]


def test_anyof_de_enums_do_mesmo_tipo_une_os_membros() -> None:
    """Colapsar para o primeiro ramo perderia os membros dos restantes: o modelo deixaria
    de saber que ``"b"`` é um valor legal e nunca o proporia."""
    out = normalize_for_cca(
        {
            "anyOf": [
                {"type": "string", "enum": ["a", "b"]},
                {"type": "string", "enum": ["b", "c"]},
            ]
        }
    )
    assert out == {"type": "string", "enum": ["a", "b", "c"]}


def test_anyof_de_tipos_mistos_colapsa_e_despeja_as_restricoes_de_cada_ramo() -> None:
    """O campo ``type`` do proto é escalar, logo a união tem de colapsar. As restrições dos
    ramos descartados vão para a descrição — apagá-las em silêncio deixaria o modelo a
    violar um ``minLength`` que não tem como conhecer."""
    out = normalize_for_cca(
        {
            "anyOf": [
                {"type": "string", "minLength": 1},
                {"type": "integer", "minimum": 0},
            ]
        }
    )
    assert out["type"] == "string"
    assert "minLength" not in out
    assert "minLength: 1" in out["description"]
    assert "minimum: 0" in out["description"]


def test_anyof_com_ramo_sem_enum_nao_estreita_aos_membros_do_outro() -> None:
    """Um ramo sem ``enum`` aceita tudo o que o ramo com ``enum`` aceita e mais. Colapsar
    para o ramo enum rejeitaria argumentos que o schema original admite."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"s": {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "string"}]}},
        }
    )
    assert out["properties"]["s"] == {"type": "string"}


def test_anyof_de_objectos_une_propriedades_e_mantem_so_o_required_comum() -> None:
    """``anyOf`` aceita um ramo qualquer. Manter obrigatório o que só um ramo exige
    rejeitaria instâncias válidas; descartar as propriedades do outro ramo esconderia
    argumentos legítimos do modelo."""
    out = normalize_for_cca(
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                    "required": ["a", "b"],
                },
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "c": {"type": "string"}},
                    "required": ["a"],
                },
            ]
        }
    )
    assert set(out["properties"]) == {"a", "b", "c"}
    assert out["required"] == ["a"]


def test_not_residual_cai_no_fallback() -> None:
    """``not`` não tem campo no proto e não há projecção que o exprima. Mandá-lo faria o
    backend devolver 400 e o pedido inteiro — não só esta ferramenta — falharia."""
    out = normalize_for_cca({"type": "object", "properties": {"x": {"not": {"type": "string"}}}})
    assert out == CCA_FALLBACK_SCHEMA


def test_ref_externo_nao_resolvivel_e_removido_sem_arrastar_o_schema() -> None:
    """Um ``$ref`` para fora do documento não se consegue inlinar. O campo ficaria como
    nome desconhecido no protojson, por isso desaparece — a ferramenta sobrevive com o
    argumento sem tipo em vez de o schema todo cair."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"x": {"$ref": "https://exemplo.pt/s.json"}, "y": {"type": "string"}},
        }
    )
    assert out["properties"] == {"x": {}, "y": {"type": "string"}}


def test_ref_local_e_inlinado_a_partir_de_defs() -> None:
    """``$defs`` é dos campos que o proto não tem. Sem inlinar antes de o remover, a
    propriedade que lhe apontava perderia toda a estrutura."""
    out = normalize_for_cca(
        {
            "$defs": {"No": {"type": "object", "properties": {"v": {"type": "string"}}}},
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/No"}},
        }
    )
    assert out["properties"]["n"] == {"type": "object", "properties": {"v": {"type": "string"}}}
    assert "$defs" not in out


def test_restricoes_removidas_vao_para_a_descricao_preservando_a_existente() -> None:
    """``pattern`` e ``minLength`` não existem no proto. Se sumissem sem deixar rasto, o
    modelo geraria argumentos que a ferramenta rejeita e não teria como perceber porquê;
    e se substituíssem a descrição, perder-se-ia o texto do autor."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {
                "p": {"type": "string", "pattern": "^a+$", "minLength": 3, "description": "Nome"}
            },
        }
    )
    p = out["properties"]["p"]
    assert p["type"] == "string"
    assert "pattern" not in p and "minLength" not in p
    assert p["description"] == 'Nome\n\n{pattern: "^a+$", minLength: 3}'


def test_nome_de_campo_em_snake_case_e_renomeado_antes_de_ser_interpretado() -> None:
    """``any_of`` é a forma que o python-genai emite. Sem o renomear não seria reconhecido
    como combinador: passaria cru como campo desconhecido e o pedido levava 400."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"a": {"any_of": [{"type": "string"}, {"type": "number"}]}},
        }
    )
    assert out["properties"]["a"] == {"type": "string"}


def test_no_de_tipo_object_recebe_properties_mesmo_vazio() -> None:
    """Um ``type: object`` sem ``properties`` é lido como objecto opaco, e a ferramenta
    acabaria a receber argumentos que nunca declarou."""
    assert normalize_for_cca({"type": "object"}) == {"type": "object", "properties": {}}


def test_anyof_de_const_vira_enum_com_o_tipo_inferido() -> None:
    """O CCA não tem ``const`` nem combinadores, mas tem ``enum``. Sem esta tradução um
    union de literais — a forma como o Zod escreve enums — caía sempre no fallback."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"modo": {"anyOf": [{"const": "ler"}, {"const": "escrever"}]}},
        }
    )
    assert out["properties"]["modo"] == {"enum": ["ler", "escrever"], "type": "string"}


def test_nome_de_propriedade_igual_a_palavra_chave_nao_e_tratado_como_combinador() -> None:
    """Dentro de ``properties`` as chaves são nomes do utilizador. Tratar uma propriedade
    chamada ``not`` como a palavra-chave mandaria para o fallback um schema perfeitamente
    válido."""
    schema = {
        "type": "object",
        "properties": {"not": {"type": "number"}, "anyOf": {"type": "string"}},
    }
    assert normalize_for_cca(schema) == schema


def test_ciclo_de_referencias_termina_em_vez_de_recorrer_sem_fim() -> None:
    """Um schema auto-referente vindo de um servidor MCP travaria o processo do proxy —
    não só o pedido — se a travessia não cortasse o ciclo."""
    cyclic: dict[str, Any] = {"type": "object", "properties": {}}
    cyclic["properties"]["self"] = cyclic
    assert normalize_for_cca(cyclic) == {"type": "object", "properties": {"self": {}}}


def test_guarda_de_ciclo_nao_confunde_nos_distintos_por_reuso_de_endereco() -> None:
    """O ``id()`` do CPython é o endereço: um nó temporário libertado a meio da travessia
    devolve o mesmo ``id`` ao seguinte, e uma guarda que só guarde inteiros declara esse
    irmão como já visitado, truncando-o para ``{}``.

    Aqui isso faria o ``not`` residual desaparecer antes da verificação de resíduos, e um
    schema que o CCA rejeita com 400 seguia para o fio em vez de cair no fallback. O
    ``WeakMap`` do original não tem o problema porque a chave é o objecto vivo.
    """
    out = normalize_for_cca({"properties": {"p0": {"properties": {"p0": {"not": {}}, "p1": {}}}}})
    assert out == CCA_FALLBACK_SCHEMA


def test_schema_de_entrada_nao_e_mutado() -> None:
    """O schema é propriedade do chamador e é reutilizado em cada pedido. Mutá-lo faria o
    segundo pedido sanear um schema já saneado, acumulando spill na descrição."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "minLength": 2}},
        "required": ["a"],
    }
    before = repr(schema)
    normalize_for_cca(schema)
    assert repr(schema) == before

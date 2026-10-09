"""omp 18.8.6's final tool-call argument parsing (`parseToolCallArguments`,
`replayableToolCallArguments`, pi-utils `parseJsonWithRepair`).

Each expectation was produced by running omp's own TypeScript on the same input under Bun:
``args`` is ``JSON.stringify`` of the parsed arguments (``None`` where omp marks them
``__parseError``), ``replay`` what the native ``function_call`` keeps for replay.
"""

from __future__ import annotations

import json

import pytest

from litellm_mysubs.wire import tool_arguments

#: (raw arguments, JSON.stringify(parseToolCallArguments(raw)), replayableToolCallArguments)
CASES = [
    ("{'a': 'it's', b: [1,2,],}", '{"a":"it\'s","b":[1,2]}', '{"a":"it\'s","b":[1,2]}'),
    (
        '{"a": True, "b": None, "c": False}',
        '{"a":true,"b":null,"c":false}',
        '{"a":true,"b":null,"c":false}',
    ),
    ('{a:1 // c\n, /* x */ b:2}', '{"a":1,"b":2}', '{"a":1,"b":2}'),
    ('{"paths": packages/foo/*}', '{"paths":"packages/foo/*"}', '{"paths":"packages/foo/*"}'),
    ('{"url": http://x.y/z}', '{"url":"http://x.y/z"}', '{"url":"http://x.y/z"}'),
    ('{"a": foo "b": 1}', None, '{"a": foo "b": 1}'),
    ('{"a": 1', None, '{"a": 1'),
    ('{"name": , "b": 1}', '{"name":"","b":1}', '{"name":"","b":1}'),
    (
        '{"n": 0x1F, "m": 0b101, "o": .5, "p": 1., "q": +3, "r": 1e5, "s": -0}',
        '{"n":31,"m":5,"o":0.5,"p":1,"q":3,"r":100000,"s":0}',
        '{"n":31,"m":5,"o":0.5,"p":1,"q":3,"r":100000,"s":0}',
    ),
    ('{"n": NaN}', None, '{"n": NaN}'),
    ('[1,2,3] x', None, '[1,2,3] x'),
    ('{"s": "a\\qb\\u12"}', '{"s":"a\\\\qb\\\\u12"}', '{"s":"a\\\\qb\\\\u12"}'),
    ('{"s": "\\ud83d\\ude00"}', '{"s":"😀"}', '{"s": "\\ud83d\\ude00"}'),
    ('{"s": "tab\there"}', '{"s":"tab\\there"}', '{"s":"tab\\there"}'),
    ('{"a": undefined}', None, '{"a": undefined}'),
    ('{"a":"b"c"}', None, '{"a":"b"c"}'),
    ('{"a": [1, , 2]}', '{"a":[1,2]}', '{"a":[1,2]}'),
    ('{"a": c:d}', None, '{"a": c:d}'),
    ('{"a": 1.0, "b": 1e999}', '{"a":1,"b":null}', '{"a": 1.0, "b": 1e999}'),
    ('', '{}', '{}'),
    ('  ', '{}', '{}'),
    ('{"a":1}', '{"a":1}', '{"a":1}'),
    ('null', 'null', 'null'),
    ('{"x": -Infinity}', None, '{"x": -Infinity}'),
]


@pytest.mark.parametrize(("raw", "args", "replay"), CASES)
def test_matches_omp(raw: str, args: str | None, replay: str) -> None:
    parsed = tool_arguments.parse_tool_call_arguments(raw)
    if args is None:
        assert isinstance(parsed, tool_arguments.InvalidArguments)
    else:
        assert json.dumps(parsed, separators=(",", ":"), ensure_ascii=False) == args
    assert tool_arguments.replayable_tool_call_arguments(raw, parsed) == replay


@pytest.mark.parametrize(
    ("name", "malformed"),
    [
        pytest.param("get_weather", False, id="plain"),
        pytest.param("f" * 128, False, id="128"),
        pytest.param("f" * 129, True, id="129"),
        pytest.param("\U0001f600" * 64, False, id="64-astral-is-128-units"),
        pytest.param("\U0001f600" * 65, True, id="65-astral-is-130-units"),
        pytest.param("", True, id="empty"),
        pytest.param(None, True, id="missing"),
        pytest.param("get weather", True, id="space"),
        pytest.param("get_weather\u00a0", True, id="nbsp"),
        pytest.param("get_weather\ufeff", True, id="bom"),
        pytest.param("get_weather\x7f", True, id="del"),
    ],
)
def test_malformed_tool_call_name(name: object, malformed: bool) -> None:
    """omp's `isMalformedToolCallName`, length in UTF-16 units."""
    assert tool_arguments.is_malformed_tool_call_name(name) is malformed

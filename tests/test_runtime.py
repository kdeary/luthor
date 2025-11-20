"""Unit tests for the Lua runtime helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from luthor import runtime as rt


def test_table_get_uses_index_function():
    table = rt.LuaTable()

    def handler(_tbl, key):
        return f"{key}_meta"

    rt.setmetatable(table, {"__index": handler})
    assert rt.table_get(table, "foo") == "foo_meta"


def test_table_get_uses_index_table():
    table = rt.LuaTable()
    fallback = {"foo": 42}
    rt.setmetatable(table, {"__index": fallback})
    assert rt.table_get(table, "foo") == 42


def test_setmetatable_on_plain_dict_allows_metafields():
    data = {"x": 1}
    mt = {"__index": {"y": 2}}
    rt.setmetatable(data, mt)
    assert rt.table_get(data, "x") == 1
    assert rt.table_get(data, "y") == 2


def test_metatable_lock_blocks_changes_and_hides_value():
    table = rt.LuaTable()
    lock = {"__metatable": "locked"}
    rt.setmetatable(table, lock)
    assert rt.getmetatable(table) == "locked"
    with pytest.raises(ValueError):
        rt.setmetatable(table, {})


def test_lua_len_invokes_metamethod():
    data = rt.LuaTable()

    def handler(tbl):
        assert tbl is data
        return 42

    rt.setmetatable(data, {"__len": handler})
    assert rt.lua_len(data) == 42


def test_generic_for_iter_uses_iter_metamethod():
    data = rt.LuaTable()

    def handler(tbl):
        assert tbl is data
        return lambda: ["iter"]

    rt.setmetatable(data, {"__iter": handler})
    assert rt.generic_for_iter([data]) == ["iter"]


def test_arithmetic_metamethods_are_invoked():
    left = rt.LuaTable()
    right = rt.LuaTable()

    def add_handler(a, b):
        assert a is left and b is right
        return "add"

    rt.setmetatable(left, {"__add": add_handler})
    assert rt.arith_add(left, right) == "add"


def test_arithmetic_fallbacks_still_work():
    assert rt.arith_mul(2, 3) == 6
    assert rt.arith_mod(7, 4) == 3


def test_unary_minus_uses_metamethod():
    value = rt.LuaTable()

    def handler(v):
        assert v is value
        return 99

    rt.setmetatable(value, {"__unm": handler})
    assert rt.arith_unm(value) == 99

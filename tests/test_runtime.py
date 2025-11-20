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


def test_len_invokes_metamethod():
    data = rt.LuaTable()

    def handler(tbl):
        assert tbl is data
        return 42

    mt = rt.LuaTable()
    mt.rawset("__len", handler)
    rt.setmetatable(data, mt)
    assert len(data) == 42


def test_generic_for_iter_uses_iter_metamethod():
    data = rt.LuaTable()

    def handler(tbl):
        assert tbl is data
        return lambda: ["iter"]

    mt = rt.LuaTable()
    mt.rawset("__iter", handler)
    rt.setmetatable(data, mt)
    assert rt.generic_for_iter([data]) == ["iter"]


def test_setmetatable_enables_arithmetic_metamethods():
    left = rt.LuaTable()
    left.rawset(1, 2)
    right = rt.LuaTable()
    right.rawset(1, 3)

    mt = rt.LuaTable()

    def add_op(a, b):
        out = rt.LuaTable()
        out.rawset(1, a.rawget(1) + b.rawget(1))
        return out

    mt.rawset("__add", add_op)
    rt.setmetatable(left, mt)
    rt.setmetatable(right, mt)

    result = left + right
    assert isinstance(result, rt.LuaTable)
    assert result.rawget(1) == 5
    assert rt.getmetatable(left) is mt


def test_pairs_iterates_over_table_entries():
    table = rt.LuaTable()
    table.rawset("name", "alice")
    table.rawset(1, "first")
    seen = list(rt.pairs(table))
    assert ("name", "alice") in seen
    assert (1, "first") in seen

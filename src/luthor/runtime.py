"""Runtime helpers to emulate Lua semantics in the generated Python."""

from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

def truthy(value: Any) -> bool:
    """Lua truthiness: only False and None map to false."""
    return not (value is False or value is None)

def lua_not(value: Any) -> bool:
    return not truthy(value)

def lua_and(left_fn: Callable[[], Any], right_fn: Callable[[], Any]) -> Any:
    left = left_fn()
    return right_fn() if truthy(left) else left

def lua_or(left_fn: Callable[[], Any], right_fn: Callable[[], Any]) -> Any:
    left = left_fn()
    return left if truthy(left) else right_fn()

def concat(left: Any, right: Any) -> str:
    """Lua's `..` operator coerces operands to strings."""
    return f"{left}{right}"

_MISSING = None

def invoke(source: Any, method: Any, *args: Any) -> Any:
    """Implements the Lua colon syntax (implicit self argument)."""
    if callable(method):
        func = method
    else:
        func = getattr(source, method)
    return func(source, *args)

def numeric_for_iter(start: Any, stop: Any, step: Any) -> Iterator[Any]:
    """Inclusive numeric-for loop iterator."""
    current = start
    if step == 0:
        raise ValueError("step must not be zero")
    if step > 0:
        while current <= stop:
            yield current
            current += step
    else:
        while current >= stop:
            yield current
            current += step


def generic_for_iter(expressions: Sequence[Any]) -> Iterable[Any]:
    """Best-effort generic for loop support with __iter fallback."""
    if not expressions:
        return []
    iterable = expressions[0]
    prepared = _prepare_iterable(iterable)
    if callable(prepared):
        return prepared()
    return prepared


@dataclass
class LuaTable:
    """Minimal Lua table emulation with mixed integer/string keys."""

    array: List[Any]
    mapping: dict

    def __init__(self) -> None:
        self.array = []
        self.mapping = {}
        self.mt: Optional[LuaTable] = None

    def append(self, value: Any) -> None:
        self.array.append(value)

    def _raw_lookup(self, key: Any):
        if isinstance(key, int):
            idx = key - 1
            if 0 <= idx < len(self.array):
                return self.array[idx]
        return self.mapping.get(key, _MISSING)

    def rawget(self, key: Any):
        value = self._raw_lookup(key)
        return None if value is _MISSING else value

    def _raw_set(self, key: Any, value: Any) -> None:
        if isinstance(key, int):
            idx = key - 1
            if idx == len(self.array):
                self.array.append(value)
                return
            if 0 <= idx < len(self.array):
                self.array[idx] = value
                return
        if value is None:
            self.mapping.pop(key, None)
        else:
            self.mapping[key] = value

    def rawset(self, key: Any, value: Any) -> None:
        self._raw_set(key, value)

    def __getitem__(self, key: Any) -> Any:
        value = self._raw_lookup(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self._raw_set(key, value)

    def __len__(self) -> int:
        handler = _get_metamethod(self, "__len")
        if handler is not None:
            return handler(self)
        return len(self.array) + len(self.mapping)

    def items(self):
        for idx, value in enumerate(self.array, start=1):
            yield idx, value
        for key, value in self.mapping.items():
            yield key, value

    @classmethod
    def update_dunders(cls, target_cls: Optional[type] = None) -> None:
        target = target_cls or cls

        def make_binary_dunder(lua_name: str):
            def op(self: LuaTable, other: Any):
                mt = getattr(self, "mt", None)
                if mt is not None:
                    fn = mt.rawget(lua_name)
                    if callable(fn):
                        return fn(self, other)
                return NotImplemented

            return op

        mapping = [
            ("__add", "__add__"),
            ("__sub", "__sub__"),
            ("__mul", "__mul__"),
            ("__div", "__truediv__"),
        ]
        for lua_name, py_name in mapping:
            setattr(target, py_name, make_binary_dunder(lua_name))


def table_ctor(fields: Sequence[Sequence[Any]]) -> LuaTable:
    table = LuaTable()
    seq_index = 1
    for flag, key, value in fields:
        if flag:
            table[key] = value
        else:
            table[seq_index] = value
            seq_index += 1
    return table


def _get_metamethod(value: Any, name: str):
    if isinstance(value, LuaTable) and value.mt is not None:
        fn = value.mt.rawget(name)
        return fn if callable(fn) else None
    return None


def _prepare_iterable(value: Any):
    if callable(value):
        return value
    handler = _get_metamethod(value, "__iter")
    if handler is None:
        return value
    return handler(value)


def _coerce_metatable(metatable: Optional[Any]) -> Optional[LuaTable]:
    if metatable is None or isinstance(metatable, LuaTable):
        return metatable
    if isinstance(metatable, dict):
        coerced = LuaTable()
        for key, value in metatable.items():
            coerced.rawset(key, value)
        return coerced
    raise TypeError("metatable must be a LuaTable or mapping")


def setmetatable(table: LuaTable, metatable: Optional[Any]):
    if not isinstance(table, LuaTable):
        raise TypeError("metatables are only supported on LuaTable instances")
    coerced = _coerce_metatable(metatable)
    table.mt = coerced
    if coerced is not None:
        LuaTable.update_dunders()
    return table


def getmetatable(table: LuaTable):
    if not isinstance(table, LuaTable):
        raise TypeError("metatables are only supported on LuaTable instances")
    return table.mt


def pairs(value: Any):
    if isinstance(value, LuaTable):
        return value.items()
    if isinstance(value, dict):
        return value.items()
    if isinstance(value, list):
        def _iter_list():
            for idx, element in enumerate(value, start=1):
                yield idx, element

        return _iter_list()
    raise TypeError("pairs() requires a table or mapping")


RUNTIME_EXPORTS = [
    "truthy",
    "lua_not",
    "lua_and",
    "lua_or",
    "concat",
    "invoke",
    "numeric_for_iter",
    "generic_for_iter",
    "LuaTable",
    "table_ctor",
    "setmetatable",
    "getmetatable",
    "pairs",
]


def create_runtime_namespace():
    namespace = types.SimpleNamespace()
    for name in RUNTIME_EXPORTS:
        setattr(namespace, name, globals()[name])
    return namespace

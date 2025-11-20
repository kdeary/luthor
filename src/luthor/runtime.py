"""Runtime helpers to emulate Lua semantics in the generated Python."""

from __future__ import annotations

from dataclasses import dataclass
import types
import weakref
from typing import Any, Callable, Iterable, Iterator, List, Sequence


def truthy(value: Any) -> bool:
    """Lua truthiness: only False and None map to false."""
    return value is not False and value is not None


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


def lua_len(value: Any) -> int:
    handler = _get_metamethod(value, "__len")
    if handler is not None:
        if callable(handler):
            return handler(value)
    return len(value)


_MISSING = object()


def _sequence_get(seq: Sequence[Any], key: Any):
    if isinstance(key, int):
        idx = key - 1
        if 0 <= idx < len(seq):
            return seq[idx]
    return _MISSING


def _sequence_set(seq: List[Any], key: Any, value: Any) -> bool:
    if not isinstance(key, int):
        return False
    idx = key - 1
    if idx == len(seq):
        seq.append(value)
        return True
    if 0 <= idx < len(seq):
        seq[idx] = value
        return True
    return False


class _MetatableRegistry:
    """Stores metatables for objects that cannot hold attributes."""

    def __init__(self) -> None:
        self._data: dict[int, tuple[weakref.ref | None, object | None, Any]] = {}

    def _cleanup(self, obj_id: int, ref: weakref.ref | None) -> None:
        record = self._data.get(obj_id)
        if not record:
            return
        stored_ref, strong_ref, _ = record
        if stored_ref is ref:
            self._data.pop(obj_id, None)

    def set(self, obj: Any, metatable: Any) -> None:
        if metatable is None:
            self._data.pop(id(obj), None)
            return
        try:
            ref = weakref.ref(obj, lambda r, obj_id=id(obj): self._cleanup(obj_id, r))
            strong_ref = None
        except TypeError:
            ref = None
            strong_ref = obj
        self._data[id(obj)] = (ref, strong_ref, metatable)

    def get(self, obj: Any):
        record = self._data.get(id(obj))
        if not record:
            return None
        ref, strong_ref, metatable = record
        if ref is not None:
            if ref() is None:
                # Object collected, drop stale entry.
                self._data.pop(id(obj), None)
                return None
            if ref() is not obj:
                return None
            return metatable
        if strong_ref is obj:
            return metatable
        return None


_METATABLES = _MetatableRegistry()

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
        self.metatable = None

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
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self._raw_set(key, value)

    def __len__(self) -> int:
        return len(self.array) + len(self.mapping)

    def items(self):
        for idx, value in enumerate(self.array, start=1):
            yield idx, value
        for key, value in self.mapping.items():
            yield key, value


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


def _is_table_like(value: Any) -> bool:
    return isinstance(value, (LuaTable, dict, list))


def _raw_lookup(value: Any, key: Any):
    if isinstance(value, LuaTable):
        return value._raw_lookup(key)
    if isinstance(value, dict):
        return value.get(key, _MISSING)
    if isinstance(value, list):
        return _sequence_get(value, key)
    return _MISSING


def _raw_set(value: Any, key: Any, new_value: Any) -> bool:
    if isinstance(value, LuaTable):
        value._raw_set(key, new_value)
        return True
    if isinstance(value, dict):
        if new_value is None:
            value.pop(key, None)
        else:
            value[key] = new_value
        return True
    if isinstance(value, list):
        if _sequence_set(value, key, new_value):
            return True
    return False


def _get_raw_metatable(table: Any):
    if isinstance(table, LuaTable):
        return getattr(table, "metatable", None)
    return _METATABLES.get(table)


def _set_raw_metatable(table: Any, metatable: Any) -> None:
    if isinstance(table, LuaTable):
        table.metatable = metatable
        return
    _METATABLES.set(table, metatable)


def _metatable_locked(metatable) -> bool:
    if not metatable:
        return False
    lock = _raw_lookup(metatable, "__metatable")
    return lock is not _MISSING


def _get_metamethod(value: Any, name: str):
    metatable = _get_raw_metatable(value)
    if not metatable:
        return None
    handler = _raw_lookup(metatable, name)
    return None if handler is _MISSING else handler


def _binary_metamethod(left: Any, right: Any, name: str, fallback: Callable[[Any, Any], Any]):
    handler = _get_metamethod(left, name)
    if handler is None:
        handler = _get_metamethod(right, name)
    if handler is not None:
        if callable(handler):
            return handler(left, right)
        raise TypeError(f"metamethod {name} must be callable")
    if fallback is not None:
        return fallback(left, right)
    raise TypeError(f"attempt to use operator {name} on incompatible values")


def _unary_metamethod(value: Any, name: str, fallback: Callable[[Any], Any]):
    handler = _get_metamethod(value, name)
    if handler is not None:
        if callable(handler):
            return handler(value)
        raise TypeError(f"metamethod {name} must be callable")
    if fallback is not None:
        return fallback(value)
    raise TypeError(f"attempt to use operator {name} on incompatible value")


def _prepare_iterable(value: Any):
    if callable(value):
        return value
    handler = _get_metamethod(value, "__iter")
    if handler is None:
        return value
    if callable(handler):
        return handler(value)
    return handler


def arith_add(left: Any, right: Any):
    return _binary_metamethod(left, right, "__add", lambda l, r: l + r)


def arith_sub(left: Any, right: Any):
    return _binary_metamethod(left, right, "__sub", lambda l, r: l - r)


def arith_mul(left: Any, right: Any):
    return _binary_metamethod(left, right, "__mul", lambda l, r: l * r)


def arith_div(left: Any, right: Any):
    return _binary_metamethod(left, right, "__div", lambda l, r: l / r)


def arith_idiv(left: Any, right: Any):
    return _binary_metamethod(left, right, "__idiv", lambda l, r: l // r)


def arith_mod(left: Any, right: Any):
    return _binary_metamethod(left, right, "__mod", lambda l, r: l % r)


def arith_pow(left: Any, right: Any):
    return _binary_metamethod(left, right, "__pow", lambda l, r: l ** r)


def arith_unm(value: Any):
    return _unary_metamethod(value, "__unm", lambda v: -v)


def setmetatable(table: Any, metatable: dict | None):
    """Attach a metatable while honoring __metatable locks."""

    current = _get_raw_metatable(table)
    if current and _metatable_locked(current):
        raise ValueError("cannot change a protected metatable")
    _set_raw_metatable(table, metatable)
    return table


def getmetatable(table: Any):
    metatable = _get_raw_metatable(table)
    if metatable is None:
        return None
    lock = _raw_lookup(metatable, "__metatable")
    if lock is not _MISSING:
        return lock
    return metatable


def table_get(value: Any, key: Any):
    existing = _raw_lookup(value, key)
    if existing is not _MISSING:
        return existing
    handler = _get_metamethod(value, "__index")
    if handler is None:
        if _is_table_like(value):
            return None
        raise TypeError("attempt to index a non-table value")
    if callable(handler):
        return handler(value, key)
    return table_get(handler, key)


RUNTIME_EXPORTS = [
    "truthy",
    "lua_not",
    "lua_and",
    "lua_or",
    "concat",
    "lua_len",
    "invoke",
    "numeric_for_iter",
    "generic_for_iter",
    "LuaTable",
    "table_ctor",
    "setmetatable",
    "getmetatable",
]


def create_runtime_namespace():
    namespace = types.SimpleNamespace()
    for name in RUNTIME_EXPORTS:
        setattr(namespace, name, globals()[name])
    return namespace

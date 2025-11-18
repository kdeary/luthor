"""Runtime helpers to emulate Lua semantics in the generated Python."""

from __future__ import annotations

from dataclasses import dataclass
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
    return len(value)

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
    """Best-effort generic for loop support: treat first expr as iterable."""
    if not expressions:
        return []
    iterable = expressions[0]
    if callable(iterable):
        return iterable()
    return iterable


@dataclass
class LuaTable:
    """Minimal Lua table emulation with mixed integer/string keys."""

    array: List[Any]
    mapping: dict

    def __init__(self) -> None:
        self.array = []
        self.mapping = {}

    def append(self, value: Any) -> None:
        self.array.append(value)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            idx = key - 1
            if 0 <= idx < len(self.array):
                return self.array[idx]
        if key in self.mapping:
            return self.mapping[key]
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, int):
            idx = key - 1
            if idx == len(self.array):
                self.array.append(value)
                return
        self.mapping[key] = value

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

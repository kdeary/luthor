# Lua Runtime Metamethod Plan

The goal is to implement the full set of Lua metatable semantics in the shared runtime so transpiled carts behave like the original code. We'll stage the work in digestible chunks.

## 1. Table Access & Assignment
- `__index` handling for both function + table cases.
- `__newindex` handling for function + table cases.
- Respect `__metatable` (lock metatable / hide true value).
- Enforce `setmetatable()` errors when metatable is locked.

## 2. Invocation & Length
- `__call` support so tables can act like functions.
- `__len` for the `#` operator.
- `__iter` for generalized iteration helper.

## 3. Unary & Arithmetic Operators
- `__unm` (unary minus).
- Binary arithmetic metamethods: `__add`, `__sub`, `__mul`, `__div`, `__idiv`, `__mod`, `__pow`.
- `__concat` for string/table concatenation.

## 4. Comparison Operators
- Relational: `__eq`, `__lt`, `__le` (with Lua's requirement that both operands share the same metamethod).
- Mirrored behavior for `>=`, `>` via `__lt`/`__le`.

## 5. Conversion & Stringification
- `__tostring` hook used by `print`/`tostring`.

## 6. Garbage-Collection / Weak Tables (stretch goal)
- Honor `__mode` semantics (`"k"`, `"v"`, `"kv"`).
- Determine feasibility in Python (may require weakref proxies).

Each chunk will:
1. Extend `LuaTable` / runtime helpers with the required dispatch.
2. Update the transpiler to emit runtime calls (rather than direct Python operators).
3. Add unit tests covering happy paths + edge cases.

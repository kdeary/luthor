# Luthor
Luthor is a Lu(a)(Py)tho(n) transpiler with an embedded runtime. It parses Lua code with [`py-lua-parser`](https://github.com/boolangery/py-lua-parser), converts the resulting AST into Python's `ast` module, and emits deterministic Python that leans on a companion runtime for Lua semantics.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
luthor path/to/input.lua -o path/to/output.py
```

The `luthor` CLI reads from a file (or stdin) and prints Python source (or writes to the provided `--output` path) after fixing locations and running the code generator.

## CLI usage

```bash
luthor INPUT.lua \
  --output OUTPUT.py \
  --no-pretty   # optional: emit raw compile() buffer
```

- `INPUT.lua`: defaults to stdin when omitted (handy for piping code).
- `--output`: writes to disk; otherwise the generated Python is printed to stdout.
- `--no-pretty`: skips `ast.unparse` so you get the raw `compile()` buffer when you prefer minimal formatting.

Every invocation injects `from luthor import runtime as __lua_runtime` at the top of the emitted module so the helpers in `luthor/runtime.py` are available.

## Python API

You can drive the transpiler programmatically via `LuaToPythonTranspiler`:

```python
from pathlib import Path
from luthor.transpiler import LuaToPythonTranspiler

transpiler = LuaToPythonTranspiler()
result = transpiler.transpile_text(\"\"\"local data = {foo = 1}\nprint(data.foo)\n\"\"\")

print(result.module)  # Python ast.Module object
print(result.source)  # rendered Python source (with runtime import)

# Or from disk:
game = Path(\"game.lua\")
compiled = transpiler.transpile_file(game)
```

Both helpers return a `TranspileResult` dataclass containing the Python `ast.Module` (useful for inspections or further transforms) and the rendered source. The transformer always fixes locations on the module so you can pass it to `ast.unparse`, `compile`, or downstream tooling without extra work.

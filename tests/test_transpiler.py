"""End-to-end tests for the Lua → Python transpiler facade."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from luthor.transpiler import LuaToPythonTranspiler


def transpile(lua_code: str):
    transpiler = LuaToPythonTranspiler()
    snippet = textwrap.dedent(lua_code).strip()
    return transpiler.transpile_text(snippet)


def test_empty_chunk_emits_import_and_pass():
    result = transpile("")
    body = result.module.body
    assert isinstance(body[0], ast.ImportFrom)
    assert isinstance(body[1], ast.Pass)
    assert result.source.startswith(
        "from luthor import runtime as __lua_runtime"
    )


def test_local_assign_pads_missing_values():
    result = transpile(
        """
        local x, y = 1
        """
    )
    assign = result.module.body[1]
    assert isinstance(assign, ast.Assign)
    assert isinstance(assign.targets[0], ast.Tuple)
    ids = [elt.id for elt in assign.targets[0].elts]
    assert ids == ["x", "y"]

    assert isinstance(assign.value, ast.Tuple)
    values = [elt.value for elt in assign.value.elts]
    assert values == [1, None]


def test_numeric_for_loop_with_default_step():
    result = transpile(
        """
        for i=1,3 do
            x = i
        end
        """
    )
    loop = result.module.body[1]
    assert isinstance(loop, ast.For)
    assert isinstance(loop.target, ast.Name)
    assert loop.target.id == "i"
    assert isinstance(loop.iter, ast.Call)
    assert isinstance(loop.iter.func, ast.Attribute)
    assert loop.iter.func.attr == "numeric_for_iter"
    args = loop.iter.args
    assert len(args) == 3
    assert [arg.value for arg in args] == [1, 3, 1]
    assert isinstance(loop.body[0], ast.Assign)


def test_numeric_for_loop_with_custom_step():
    result = transpile(
        """
        for i=6,2,-2 do
            y = i
        end
        """
    )
    loop = result.module.body[1]
    args = loop.iter.args
    assert isinstance(args[0], ast.Constant) and args[0].value == 6
    assert isinstance(args[1], ast.Constant) and args[1].value == 2
    step = args[2]
    if isinstance(step, ast.Constant):
        assert step.value == -2
    else:
        assert isinstance(step, ast.UnaryOp)
        assert isinstance(step.op, ast.USub)
        assert isinstance(step.operand, ast.Constant)
        assert step.operand.value == 2


def test_generic_for_loop_uses_runtime_iterator():
    result = transpile(
        """
        for key, value in pairs(tbl) do
            seen = key
        end
        """
    )
    loop = result.module.body[1]
    assert isinstance(loop.target, ast.Tuple)
    ids = [elt.id for elt in loop.target.elts]
    assert ids == ["key", "value"]

    iter_call = loop.iter
    assert isinstance(iter_call, ast.Call)
    assert isinstance(iter_call.func, ast.Attribute)
    assert iter_call.func.attr == "generic_for_iter"
    assert len(iter_call.args) == 1
    collection = iter_call.args[0]
    assert isinstance(collection, ast.List)
    call = collection.elts[0]
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "pairs"


def test_while_and_repeat_until_translation():
    result = transpile(
        """
        while flag do
            x = x + 1
        end

        repeat
            x = x - 1
        until done
        """
    )
    while_stmt = result.module.body[1]
    repeat_stmt = result.module.body[2]

    assert isinstance(while_stmt, ast.While)
    assert isinstance(while_stmt.test, ast.Call)
    assert isinstance(repeat_stmt, ast.While)
    assert isinstance(repeat_stmt.test, ast.Constant)
    guard = repeat_stmt.body[-1]
    assert isinstance(guard, ast.If)
    assert isinstance(guard.body[0], ast.Break)


def test_if_elseif_else_chain_produces_nested_ifs():
    result = transpile(
        """
        if a then
            hit = 1
        elseif b then
            hit = 2
        else
            hit = 3
        end
        """
    )
    first_if = result.module.body[1]
    assert isinstance(first_if, ast.If)
    assert len(first_if.orelse) == 1
    nested = first_if.orelse[0]
    assert isinstance(nested, ast.If)
    assert isinstance(nested.orelse[0], ast.Assign)


def test_function_table_assignment_for_dot_notation():
    result = transpile(
        """
        function player.attack(x)
            return x
        end
        """
    )
    fn = result.module.body[1]
    assign = result.module.body[2]
    assert isinstance(fn, ast.FunctionDef)
    assert fn.name == "attack"
    assert isinstance(assign, ast.Assign)
    target = assign.targets[0]
    assert isinstance(target, ast.Subscript)
    assert isinstance(assign.value, ast.Name)
    assert assign.value.id == fn.name


def test_method_definition_injects_self_and_assigns():
    result = transpile(
        """
        function player:jump(power)
            return power
        end
        """
    )
    fn = result.module.body[1]
    assign = result.module.body[2]
    assert isinstance(fn, ast.FunctionDef)
    assert fn.args.args[0].arg == "self"
    assert isinstance(assign.targets[0], ast.Subscript)
    assert isinstance(assign.value, ast.Name)
    assert assign.value.id == fn.name


def test_invoke_uses_runtime_helper():
    result = transpile(
        """
        function tick()
            hero:jump(1)
        end
        """
    )
    fn = result.module.body[1]
    assert isinstance(fn.body[0], ast.Expr)
    call_expr = fn.body[0].value
    assert isinstance(call_expr, ast.Call)
    assert isinstance(call_expr.func, ast.Attribute)
    assert call_expr.func.attr == "invoke"


def test_table_constructor_literal_keys_become_strings():
    result = transpile(
        """
        local data = {x = 1, [idx] = 2, 3}
        """
    )
    assign = result.module.body[1]
    call = assign.value
    assert isinstance(call, ast.Call)
    fields = call.args[0].elts
    literal_field = fields[0]
    assert isinstance(literal_field.elts[1], ast.Constant)
    assert literal_field.elts[1].value == "x"


def test_table_constructor_with_only_keyed_fields_becomes_dict():
    result = transpile(
        """
        local player = {x = 1, y = 2}
        """
    )
    assign = result.module.body[1]
    assert isinstance(assign.value, ast.Dict)
    keys = assign.value.keys
    values = assign.value.values
    assert [key.value for key in keys] == ["x", "y"]
    assert [value.value for value in values] == [1, 2]


def test_table_constructor_with_only_sequence_fields_becomes_list():
    result = transpile(
        """
        local numbers = {1, 2, 3}
        """
    )
    assign = result.module.body[1]
    assert isinstance(assign.value, ast.List)
    assert [elt.value for elt in assign.value.elts] == [1, 2, 3]

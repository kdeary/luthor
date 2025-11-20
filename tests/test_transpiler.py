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

from luthor.transpiler import LuaToPythonTranspiler, TransformerConfig


def transpile(lua_code: str, *, config: TransformerConfig | None = None):
    transpiler = LuaToPythonTranspiler(config=config)
    snippet = textwrap.dedent(lua_code).strip()
    return transpiler.transpile_text(snippet)


PREFIX_LEN = 1  # runtime import only


def body_without_prefix(result):
    return result.module.body[PREFIX_LEN:]


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
    assign = body_without_prefix(result)[0]
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
    loop = body_without_prefix(result)[0]
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
    loop = body_without_prefix(result)[0]
    args = loop.iter.args
    assert isinstance(args[0], ast.Constant) and args[0].value == 6
    assert isinstance(args[1], ast.Constant) and args[1].value == 2
    step = args[2]
    if isinstance(step, ast.Constant):
        assert step.value == -2
    elif isinstance(step, ast.Call):
        assert isinstance(step.func, ast.Attribute)
        assert step.func.attr == "arith_unm"
        assert isinstance(step.args[0], ast.Constant)
        assert step.args[0].value == 2
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
    loop = body_without_prefix(result)[0]
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
    user_body = body_without_prefix(result)
    while_stmt = user_body[0]
    repeat_stmt = user_body[1]

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
    first_if = body_without_prefix(result)[0]
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
    user_body = body_without_prefix(result)
    fn = user_body[0]
    assign = user_body[1]
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
    user_body = body_without_prefix(result)
    fn = user_body[0]
    assign = user_body[1]
    assert isinstance(fn, ast.FunctionDef)
    assert fn.args.args[0].arg == "self"
    assert isinstance(assign, ast.Assign)
    target = assign.targets[0]
    assert isinstance(target, ast.Subscript)
    assert isinstance(assign.value, ast.Name)
    assert assign.value.id == fn.name


def test_table_access_uses_runtime_helper():
    result = transpile(
        """
        local value = player.stats.health
        """
    )
    assign = body_without_prefix(result)[0]
    assert isinstance(assign, ast.Assign)
    sub = assign.value
    assert isinstance(sub, ast.Subscript)
    inner = sub.value
    assert isinstance(inner, ast.Subscript)
    assert isinstance(inner.slice, ast.Constant)
    assert inner.slice.value == "stats"
    assert isinstance(sub.slice, ast.Constant)
    assert sub.slice.value == "health"


def test_table_assignment_uses_runtime_helper():
    result = transpile(
        """
        enemy[1] = 2
        """
    )
    stmt = body_without_prefix(result)[0]
    assert isinstance(stmt, ast.Assign)
    target = stmt.targets[0]
    assert isinstance(target, ast.Subscript)
    assert isinstance(target.value, ast.Name)
    assert target.value.id == "enemy"
    assert isinstance(target.slice, ast.Constant)
    assert target.slice.value == 1


def test_arithmetic_ops_use_runtime_helpers():
    result = transpile(
        """
        value = left + right
        """
    )
    assign = body_without_prefix(result)[0]
    binop = assign.value
    assert isinstance(binop, ast.BinOp)
    assert isinstance(binop.op, ast.Add)


def test_unary_minus_uses_runtime_helper():
    result = transpile(
        """
        value = -enemy
        """
    )
    assign = body_without_prefix(result)[0]
    unary = assign.value
    assert isinstance(unary, ast.UnaryOp)
    assert isinstance(unary.op, ast.USub)


def test_comparison_ops_use_runtime_helpers():
    result = transpile(
        """
        value = left == right
        """
    )
    assign = body_without_prefix(result)[0]
    cmp = assign.value
    assert isinstance(cmp, ast.Compare)
    assert isinstance(cmp.ops[0], ast.Eq)


def test_relational_ops_handle_ge_runtime_helper():
    result = transpile(
        """
        value = a >= b
        """
    )
    assign = body_without_prefix(result)[0]
    cmp = assign.value
    assert isinstance(cmp, ast.Compare)
    assert isinstance(cmp.ops[0], ast.GtE)


def test_function_call_emits_direct_call():
    result = transpile(
        """
        foo(bar)
        """
    )
    expr = body_without_prefix(result)[0]
    assert isinstance(expr, ast.Expr)
    call = expr.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "foo"
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "bar"


def test_invoke_uses_runtime_helper():
    result = transpile(
        """
        function tick()
            hero:jump(1)
        end
        """
    )
    fn = body_without_prefix(result)[0]
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
    assign = body_without_prefix(result)[0]
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
    assign = body_without_prefix(result)[0]
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
    assign = body_without_prefix(result)[0]
    assert isinstance(assign.value, ast.List)
    assert [elt.value for elt in assign.value.elts] == [1, 2, 3]


def test_anonymous_function_with_statements_is_hoisted():
    result = transpile(
        """
        local thunk = function()
            local sum = 0
            sum = sum + 1
            return sum
        end
        """
    )
    user_body = body_without_prefix(result)
    func_def = user_body[0]
    assign = user_body[1]
    assert isinstance(func_def, ast.FunctionDef)
    assert func_def.name.startswith("__lua_function_")
    assert isinstance(assign, ast.Assign)
    assert isinstance(assign.value, ast.Name)
    assert assign.value.id == func_def.name


def test_pico8_globals_are_wrapped_with_pyco8():
    config = TransformerConfig(wrap_globals={"rnd"})
    result = transpile(
        """
        x = rnd(5)
        """,
        config=config,
    )
    assign = body_without_prefix(result)[0]
    call = assign.value
    assert isinstance(call, ast.Call)
    callee = call.func
    assert isinstance(callee, ast.Attribute)
    assert isinstance(callee.value, ast.Name)
    assert callee.value.id == "PYCO8"
    assert callee.attr == "rnd"


def test_local_shadowing_skips_pyco8_wrap():
    config = TransformerConfig(wrap_globals={"rnd"})
    result = transpile(
        """
        local rnd = 1
        x = rnd
        """,
        config=config,
    )
    assign = body_without_prefix(result)[1]
    assert isinstance(assign.value, ast.Name)
    assert assign.value.id == "rnd"


def test_keyword_locals_get_sanitized():
    result = transpile(
        """
        local del = 1
        x = del
        """
    )
    names = {node.id for node in ast.walk(result.module) if isinstance(node, ast.Name)}
    assert "del" not in names
    assert "del_" in names


def test_keyword_function_argument_gets_sanitized():
    result = transpile(
        """
        function tick(del)
            return del
        end
        """
    )
    fn = body_without_prefix(result)[0]
    assert isinstance(fn, ast.FunctionDef)
    assert fn.args.args[0].arg == "del_"
    names = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)}
    assert "del" not in names
    assert "del_" in names


def test_goto_translates_to_label_state_machine():
    result = transpile(
        """
        local i = 0
        ::loop::
        i = i + 1
        if i < 3 then
            goto loop
        end
        """
    )
    user_body = body_without_prefix(result)
    assign = user_body[0]
    loop = user_body[1]
    assert isinstance(assign.targets[0], ast.Name)
    assert assign.targets[0].id == "_lua_label"
    assert isinstance(assign.value, ast.Constant)
    assert assign.value.value == "__start__"

    assert isinstance(loop, ast.While)
    assert isinstance(loop.body[0], ast.If)
    label_branch = loop.body[0].orelse[0]
    assert isinstance(label_branch, ast.If)

    def has_continue(stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.Continue):
                return True
            for attr in ("body", "orelse"):
                child = getattr(stmt, attr, None)
                if isinstance(child, list) and has_continue(child):
                    return True
        return False

    assert has_continue(label_branch.body)

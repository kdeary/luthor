"""Lua AST → Python AST transformer."""

from __future__ import annotations

import ast
import copy
import inspect
import keyword
import textwrap
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from luaparser import astnodes as lua

from . import runtime as runtime_module

_runtime_source = inspect.getsource(runtime_module)
_runtime_lines = [
    line
    for line in textwrap.dedent(_runtime_source).splitlines()
    if not line.startswith("from __future__ import")
]
BUNDLED_RUNTIME_AST = ast.parse("\n".join(_runtime_lines))

class TransformerError(RuntimeError):
    """Raised when the transformer encounters an unsupported node."""


@dataclass
class TransformerConfig:
    """Holds knobs for the transformer."""

    inject_runtime_import: bool = True
    bundle_runtime: bool = False
    runtime_module: str = "luthor"
    runtime_symbol: str = "runtime"
    runtime_alias: str = "__lua_runtime"
    initialize_globals: bool = True
    initialize_all_globals: bool = False
    wrap_globals: Optional[Set[str]] = None
    wrap_container: str = "API"
    wrap_function_name: Optional[str] = None
    wrap_function_args: Optional[List[str]] = None


class LuaToPythonAstTransformer:
    """Walks the Lua AST produced by py-lua-parser and builds Python AST."""

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        self.config = config or TransformerConfig()
        self._function_counter = 0
        self._temp_counter = 0
        self._inline_stack: List[List[ast.stmt]] = []
        self._current_label_var: Optional[str] = None
        self._locals_stack: List[set[str]] = [set()]
        self._function_depth = 0
        self._function_global_stack: List[set[str]] = []
        self._globals_assigned: set[str] = set()
        self._globals_in_functions: set[str] = set()

    # -- entrypoint -----------------------------------------------------

    def transform(self, chunk: lua.Chunk) -> ast.Module:
        if not isinstance(chunk, lua.Chunk):
            raise TypeError(f"Expected lua.Chunk, received {type(chunk)!r}")
        self._globals_assigned.clear()
        self._globals_in_functions.clear()
        body = self.visit_Block(chunk.body)
        runtime_prefix: List[ast.stmt] = []
        if self.config.bundle_runtime:
            runtime_nodes = copy.deepcopy(BUNDLED_RUNTIME_AST.body)
            runtime_prefix.extend(runtime_nodes)
            runtime_prefix.extend(ast.parse("__lua_runtime = create_runtime_namespace()").body)
        elif self.config.inject_runtime_import:
            runtime_prefix.append(
                ast.ImportFrom(
                    module=self.config.runtime_module,
                    names=[ast.alias(name=self.config.runtime_symbol, asname=self.config.runtime_alias)],
                    level=0,
                )
            )
        builtin_nodes = self._lua_builtin_assignments()
        initializer_nodes = self._global_initializers()
        body = builtin_nodes + body
        wrap_name = self.config.wrap_function_name
        if wrap_name:
            arg_names = list(self.config.wrap_function_args or [])
            func_def = ast.FunctionDef(
                name=wrap_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=name) for name in arg_names],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=body,
                decorator_list=[],
                returns=None,
                type_comment=None,
            )
            body = runtime_prefix + initializer_nodes + [func_def]
        else:
            body = runtime_prefix + initializer_nodes + body
        return ast.Module(body=body, type_ignores=[])

    # -- helpers --------------------------------------------------------

    def _runtime_attr(self, name: str) -> ast.Attribute:
        return ast.Attribute(
            value=ast.Name(id=self.config.runtime_alias, ctx=ast.Load()),
            attr=name,
            ctx=ast.Load(),
        )

    def _runtime_call(self, name: str, *args: ast.expr) -> ast.Call:
        return ast.Call(func=self._runtime_attr(name), args=list(args), keywords=[])

    def _lua_builtin_assignments(self) -> List[ast.stmt]:
        mapping = {
            "setmetatable": "setmetatable",
            "getmetatable": "getmetatable",
            "pairs": "pairs",
        }
        assigns: List[ast.stmt] = []
        for target_name, runtime_name in mapping.items():
            assigns.append(
                ast.Assign(
                    targets=[ast.Name(id=target_name, ctx=ast.Store())],
                    value=self._runtime_attr(runtime_name),
                )
            )
        return assigns

    def _wrap_condition(self, expr: ast.expr) -> ast.expr:
        """Lua truthiness: only nil/false are falsy."""
        if self._is_definitely_boolean(expr):
            return expr
        return self._runtime_call("truthy", expr)

    def _is_definitely_boolean(self, expr: ast.expr) -> bool:
        """Return True when Python already guarantees a bool result."""
        if isinstance(expr, ast.Compare):
            return True
        if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
            return True
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            return True
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            func = expr.func
            if isinstance(func.value, ast.Name) and func.value.id == self.config.runtime_alias:
                if func.attr in {"lua_not", "truthy"}:
                    return True
        return False

    def _wrap_bool_op(
        self, runtime_name: str, left: ast.expr, right: ast.expr
    ) -> ast.Call:
        """Short-circuit aware bool op via runtime."""
        return self._runtime_call(
            runtime_name, self._lambda_wrapper(left), self._lambda_wrapper(right)
        )

    def _lambda_wrapper(self, expr: ast.expr) -> ast.Lambda:
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=expr,
        )

    def _tuple_or_single(self, values: Sequence[ast.expr]) -> ast.expr:
        if not values:
            return ast.Constant(None)
        if len(values) == 1:
            return values[0]
        return ast.Tuple(elts=list(values), ctx=ast.Load())

    def _fresh_function_name(self) -> str:
        name = f"__lua_function_{self._function_counter}"
        self._function_counter += 1
        return name

    def _fresh_temp_name(self) -> str:
        name = f"__lua_tmp_{self._temp_counter}"
        self._temp_counter += 1
        return name

    def _function_binding(self, node) -> tuple[str, Optional[Tuple[ast.expr, ast.expr]]]:
        if isinstance(node, lua.Name):
            return self._sanitize_identifier(node.id), None
        if isinstance(node, lua.Index):
            if isinstance(node.idx, lua.Name):
                candidate = node.idx.id
            elif isinstance(node.idx, lua.String):
                candidate = node.idx.s
            else:
                candidate = None
            if candidate:
                candidate = self._sanitize_identifier(candidate)
            func_name = candidate if candidate and candidate.isidentifier() else self._fresh_function_name()
            target = self._index_components(node)
            return func_name, target
        raise TransformerError("Function names with complex expressions are not supported yet.")

    def _apply_default_placeholders(self, args_obj: ast.arguments) -> None:
        args_obj.defaults = [ast.Constant(None) for _ in args_obj.args]

    def _build_function_definition(self, node, *, is_local: bool = False) -> tuple[ast.FunctionDef, Optional[ast.stmt]]:
        if is_local and isinstance(node.name, lua.Name):
            self._declare_local(node.name.id)
        func_name, assignment_target = self._function_binding(node.name)
        if not is_local and isinstance(node.name, lua.Name):
            self._record_global(func_name)
        args_ast, arg_names = self._build_arguments(node.args)
        body = self._function_block(node.body, initial_locals=arg_names)
        func_def = ast.FunctionDef(
            name=func_name,
            args=args_ast,
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        assignment: Optional[ast.stmt] = None
        if assignment_target is not None:
            table_expr, key_expr = assignment_target
            target = ast.Subscript(value=table_expr, slice=key_expr, ctx=ast.Store())
            assignment = ast.Assign(targets=[target], value=ast.Name(id=func_name, ctx=ast.Load()))
        return func_def, assignment

    def _ensure_store(self, node: ast.AST) -> ast.AST:
        """Set ctx=Store for assignment targets."""
        if isinstance(node, ast.Name):
            node.ctx = ast.Store()
        elif isinstance(node, ast.Attribute):
            node.ctx = ast.Store()
        elif isinstance(node, ast.Subscript):
            node.ctx = ast.Store()
        elif isinstance(node, ast.Tuple):
            node.ctx = ast.Store()
            for elt in node.elts:
                self._ensure_store(elt)
        else:
            raise TransformerError(f"Unsupported assignment target: {ast.dump(node)}")
        return node

    def _block(self, block: lua.Block, initial_locals: Optional[Iterable[str]] = None) -> List[ast.stmt]:
        return self.visit_Block(block, initial_locals=initial_locals)

    def _function_block(self, block: lua.Block, initial_locals: Optional[Iterable[str]] = None) -> List[ast.stmt]:
        self._function_global_stack.append(set())
        self._function_depth += 1
        try:
            body = self._block(block, initial_locals=initial_locals)
        finally:
            self._function_depth -= 1
            globals_in_function = self._function_global_stack.pop()
        if globals_in_function:
            body = [ast.Global(names=sorted(globals_in_function))] + body
        return body

    def _expr_list(self, nodes: Iterable[lua.Expression]) -> List[ast.expr]:
        return [self.visit_expression(node) for node in nodes]

    def _push_scope(self, names: Optional[Iterable[str]] = None) -> None:
        scope = set(names or [])
        self._locals_stack.append(scope)

    def _pop_scope(self) -> None:
        self._locals_stack.pop()

    def _declare_local(self, name: str) -> None:
        sanitized = self._sanitize_identifier(name)
        self._locals_stack[-1].add(sanitized)

    def _declare_locals(self, names: Iterable[str]) -> None:
        for name in names:
            self._declare_local(name)

    def _is_local(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self._locals_stack))

    def _should_wrap_global(self, name: str) -> bool:
        globals_set = self.config.wrap_globals
        if not globals_set or name not in globals_set:
            return False
        if name in {self.config.wrap_container, self.config.runtime_alias, "_lua_label"}:
            return False
        return not self._is_local(self._sanitize_identifier(name))

    def _sanitize_identifier(self, name: str) -> str:
        if not name:
            return "_"
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        if keyword.iskeyword(cleaned):
            cleaned = f"{cleaned}_"
        return cleaned

    def _record_global(self, name: str) -> None:
        """Track a non-local binding for later initialization."""
        if name.startswith("__lua_"):
            return
        self._globals_assigned.add(name)
        if self._function_depth > 0:
            self._globals_in_functions.add(name)
            if self._function_global_stack:
                self._function_global_stack[-1].add(name)

    def _track_global_from_target(self, target_node: lua.Lhs) -> None:
        if isinstance(target_node, lua.Name):
            sanitized = self._sanitize_identifier(target_node.id)
            if not self._is_local(sanitized):
                self._record_global(sanitized)

    def _global_initializers(self) -> List[ast.stmt]:
        if not self.config.initialize_globals:
            return []
        names = self._globals_assigned if self.config.initialize_all_globals else self._globals_in_functions
        if not names:
            return []
        return [
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Constant(None),
            )
            for name in sorted(names)
        ]

    def _append_converted(self, body: List[ast.stmt], converted):
        if converted is None:
            return
        if isinstance(converted, list):
            body.extend(converted)
        elif isinstance(converted, ast.stmt):
            body.append(converted)
        elif isinstance(converted, ast.expr):
            body.append(ast.Expr(value=converted))
        else:
            raise TransformerError(f"Unexpected block translation: {converted!r}")

    def _contains_goto(self, stmts: Sequence[lua.Statement]) -> bool:
        return any(isinstance(stmt, lua.Label) for stmt in stmts)

    # why does goto exist man...
    def _translate_goto_block(self, stmts: Sequence[lua.Statement]) -> List[ast.stmt]:
        label_var = "_lua_label"
        default_label = "__start__"
        segments: List[tuple[str, list]] = [(default_label, [])]
        for stmt in stmts:
            if isinstance(stmt, lua.Label):
                label_name = stmt.id.id if isinstance(stmt.id, lua.Name) else str(stmt.id)
                segments.append((label_name, []))
            else:
                segments[-1][1].append(stmt)

        prev_label_var = self._current_label_var
        self._current_label_var = label_var

        self._declare_local(label_var)
        init_assign = ast.Assign(
            targets=[ast.Name(id=label_var, ctx=ast.Store())],
            value=ast.Constant(segments[0][0]),
        )

        next_if: Optional[ast.stmt] = None
        for idx in range(len(segments) - 1, -1, -1):
            label_name, segment_stmts = segments[idx]
            branch_body: List[ast.stmt] = []
            for stmt in segment_stmts:
                converted = self.visit(stmt)
                self._append_converted(branch_body, converted)
            if idx == len(segments) - 1:
                branch_body.append(ast.Break())
            else:
                next_label = segments[idx + 1][0]
                branch_body.append(
                    ast.Assign(
                        targets=[ast.Name(id=label_var, ctx=ast.Store())],
                        value=ast.Constant(next_label),
                    )
                )
                branch_body.append(ast.Continue())

            test = ast.Compare(
                left=ast.Name(id=label_var, ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(label_name)],
            )
            orelse = [next_if] if next_if else [ast.Break()]
            next_if = ast.If(test=test, body=branch_body, orelse=orelse)

        loop = ast.While(test=ast.Constant(True), body=[next_if], orelse=[])
        self._current_label_var = prev_label_var
        return [init_assign, loop]

    # -- dispatcher -----------------------------------------------------

    def visit(self, node, **kwargs):
        if node is None:
            return None
        method = getattr(
            self, f"visit_{node.__class__.__name__}", self.generic_visit
        )
        return method(node, **kwargs)

    def generic_visit(self, node, **_: object):
        raise TransformerError(f"Unsupported node: {node.__class__.__name__}")

    # -- structural nodes ----------------------------------------------

    def visit_Block(self, node: lua.Block, initial_locals: Optional[Iterable[str]] = None) -> List[ast.stmt]:
        self._inline_stack.append([])
        self._push_scope(initial_locals)
        if self._contains_goto(node.body):
            python_body = self._translate_goto_block(node.body)
        else:
            python_body = []
            for stmt in node.body:
                if isinstance(stmt, (lua.Function, lua.LocalFunction, lua.Method)):
                    converted = self.visit(stmt)
                    self._append_converted(python_body, converted)
                else:
                    converted = self.visit(stmt)
                    self._append_converted(python_body, converted)
            if not python_body:
                python_body.append(ast.Pass())
        hoisted = self._inline_stack.pop()
        self._pop_scope()
        if hoisted:
            python_body = hoisted + python_body
        return python_body

    def visit_Goto(self, node: lua.Goto):
        if not self._current_label_var:
            raise TransformerError("goto used outside of a label-enabled block")
        label_name = node.label.id if isinstance(node.label, lua.Name) else str(node.label)
        assign = ast.Assign(
            targets=[ast.Name(id=self._current_label_var, ctx=ast.Store())],
            value=ast.Constant(label_name),
        )
        return [assign, ast.Continue()]

    def visit_Do(self, node: lua.Do) -> List[ast.stmt]:
        # The `do ... end` block simply injects its body.
        return self._block(node.body)

    def visit_Assign(self, node: lua.Assign):
        targets = list(node.targets)
        for target in targets:
            self._track_global_from_target(target)
        target_count = max(1, len(targets))
        values = self._expr_list(node.values or [])
        if not values:
            values = [ast.Constant(None)] * target_count
        else:
            if len(values) < target_count:
                values = values + [ast.Constant(None)] * (target_count - len(values))
            elif len(values) > target_count:
                values = list(values[:target_count])

        has_index_target = any(isinstance(target, lua.Index) for target in targets)
        if not has_index_target:
            py_targets = [self._ensure_store(self.visit_lhs(t)) for t in targets]
            if len(py_targets) == 1:
                final_target: ast.expr = py_targets[0]
            else:
                final_target = ast.Tuple(elts=py_targets, ctx=ast.Store())
            return ast.Assign(targets=[final_target], value=self._tuple_or_single(values))

        if target_count == 1:
            stmts = self._assign_single_target(targets[0], values[0])
            if not stmts:
                return ast.Pass()
            return stmts[0] if len(stmts) == 1 else stmts

        statements: List[ast.stmt] = []
        temp_names: List[Optional[str]] = []
        for value in values:
            if isinstance(value, ast.Constant) and value.value is None:
                temp_names.append(None)
                continue
            temp_name = self._fresh_temp_name()
            self._declare_local(temp_name)
            temp_names.append(temp_name)
            statements.append(
                ast.Assign(
                    targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                    value=value,
                )
            )

        for idx, target in enumerate(targets):
            if idx < len(temp_names) and temp_names[idx] is not None:
                assigned_value = ast.Name(id=temp_names[idx], ctx=ast.Load())
            else:
                assigned_value = ast.Constant(None)
            statements.extend(self._assign_single_target(target, assigned_value))

        if len(statements) == 1:
            return statements[0]
        return statements

    def visit_LocalAssign(self, node: lua.LocalAssign):
        names = [target.id for target in node.targets if isinstance(target, lua.Name)]
        self._declare_locals(names)
        return self.visit_Assign(node)

    def _assign_single_target(self, target_node, value_expr: ast.expr) -> List[ast.stmt]:
        if isinstance(target_node, lua.Index):
            target = self.visit_Index(target_node, ctx=ast.Store())
        else:
            target = self._ensure_store(self.visit_lhs(target_node))
        return [ast.Assign(targets=[target], value=value_expr)]

    def visit_Return(self, node: lua.Return) -> ast.Return:
        values = self._expr_list(node.values or [])
        return ast.Return(value=self._tuple_or_single(values) if values else None)

    def visit_While(self, node: lua.While) -> ast.While:
        test = self._wrap_condition(self.visit_expression(node.test))
        body = self._block(node.body)
        return ast.While(test=test, body=body, orelse=[])

    def visit_Repeat(self, node: lua.Repeat) -> ast.While:
        body = self._block(node.body)
        test = self.visit_expression(node.test)
        guard = ast.If(
            test=self._wrap_condition(test),
            body=[ast.Break()],
            orelse=[],
        )
        body.append(guard)
        return ast.While(test=ast.Constant(True), body=body, orelse=[])

    def visit_Fornum(self, node: lua.Fornum) -> ast.For:
        if isinstance(node.target, lua.Name):
            self._declare_local(self._sanitize_identifier(node.target.id))
        target = self._ensure_store(self.visit_Name(node.target, ctx=ast.Store()))
        if node.step is None:
            step_expr = ast.Constant(1)
        elif isinstance(node.step, (int, float)):
            step_expr = ast.Constant(node.step)
        else:
            step_expr = self.visit_expression(node.step)
        iter_call = self._runtime_call(
            "numeric_for_iter",
            self.visit_expression(node.start),
            self.visit_expression(node.stop),
            step_expr,
        )
        return ast.For(
            target=target,
            iter=iter_call,
            body=self._block(node.body),
            orelse=[],
        )

    def visit_Forin(self, node: lua.Forin) -> ast.For:
        for target in node.targets:
            if isinstance(target, lua.Name):
                self._declare_local(self._sanitize_identifier(target.id))
        targets = [self.visit_Name(t, ctx=ast.Store()) for t in node.targets]
        target_expr: ast.expr
        if len(targets) == 1:
            target_expr = targets[0]
        else:
            target_expr = ast.Tuple(elts=targets, ctx=ast.Store())
        iterables = ast.List(
            elts=[self.visit_expression(expr) for expr in node.iter],
            ctx=ast.Load(),
        )
        iter_call = self._runtime_call("generic_for_iter", iterables)
        return ast.For(target=target_expr, iter=iter_call, body=self._block(node.body), orelse=[])

    def visit_Break(self, _: lua.Break) -> ast.Break:
        return ast.Break()

    def visit_Continue(self, _: lua.Continue) -> ast.Continue:
        return ast.Continue()

    def visit_SemiColon(self, _: lua.SemiColon) -> ast.Pass:
        return ast.Pass()

    def visit_If(self, node: lua.If) -> ast.If:
        return ast.If(
            test=self._wrap_condition(self.visit_expression(node.test)),
            body=self._block(node.body),
            orelse=self._convert_orelse(node.orelse),
        )

    def _convert_orelse(self, node) -> List[ast.stmt]:
        if node is None:
            return []
        if isinstance(node, list):
            return [self.visit(stmt) for stmt in node]
        if isinstance(node, lua.Block):
            return self._block(node)
        if isinstance(node, lua.ElseIf):
            return [
                ast.If(
                    test=self._wrap_condition(self.visit_expression(node.test)),
                    body=self._block(node.body),
                    orelse=self._convert_orelse(node.orelse),
                )
            ]
        raise TransformerError(f"Unexpected orelse payload: {node!r}")

    def visit_Function(self, node: lua.Function):
        func_def, assignment = self._build_function_definition(node)
        if not self._inline_stack:
            raise TransformerError("Function defined outside of a block context.")
        self._inline_stack[-1].append(func_def)
        return assignment

    def visit_LocalFunction(self, node: lua.LocalFunction) -> ast.FunctionDef:
        func_def, assignment = self._build_function_definition(node, is_local=True)
        if not self._inline_stack:
            raise TransformerError("Function defined outside of a block context.")
        self._inline_stack[-1].append(func_def)
        return assignment

    def visit_Call(self, node: lua.Call) -> ast.Call:
        return self._build_call(node.func, node.args)

    def visit_Invoke(self, node: lua.Invoke) -> ast.Call:
        if not isinstance(node.func, lua.Name):
            raise TransformerError("Complex invoke targets are not supported")
        method_name = ast.Constant(node.func.id)
        args = [self.visit_expression(arg) for arg in node.args]
        return self._runtime_call(
            "invoke", self.visit_expression(node.source), method_name, *args
        )

    def visit_Method(self, node: lua.Method):
        method_name = node.name.id if isinstance(node.name, lua.Name) else None
        func_name = method_name if method_name and method_name.isidentifier() else self._fresh_function_name()
        py_args, arg_names = self._build_arguments(node.args)
        py_args.args.insert(0, ast.arg(arg="self"))
        self._apply_default_placeholders(py_args)
        locals_list = ["self"] + arg_names
        body = self._function_block(node.body, initial_locals=locals_list)
        table_expr = self.visit_expression(node.source)
        key_expr = ast.Constant(method_name) if method_name else self.visit_expression(node.name)
        func_def = ast.FunctionDef(
            name=func_name,
            args=py_args,
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        target = ast.Subscript(value=table_expr, slice=key_expr, ctx=ast.Store())
        assign = ast.Assign(
            targets=[target],
            value=ast.Name(id=func_name, ctx=ast.Load()),
        )
        if not self._inline_stack:
            raise TransformerError("Method defined outside of a block context.")
        self._inline_stack[-1].append(func_def)
        return assign

    # -- expressions ----------------------------------------------------

    def visit_expression(self, node: lua.Expression) -> ast.expr:
        expr = self.visit(node)
        if isinstance(expr, ast.stmt):
            raise TransformerError(f"Expected expression, got statement: {node}")
        return expr

    def visit_lhs(self, node: lua.Lhs) -> ast.expr:
        expr = self.visit(node)
        if isinstance(expr, ast.stmt):
            raise TransformerError(f"Expected LHS expression, got statement: {node}")
        return expr

    def visit_Name(self, node: lua.Name, ctx: Optional[ast.expr_context] = None) -> ast.AST:
        ctx = ctx or ast.Load()
        sanitized = self._sanitize_identifier(node.id)
        if isinstance(ctx, ast.Load) and self._should_wrap_global(node.id):
            container = ast.Name(id=self.config.wrap_container, ctx=ast.Load())
            attr = self._sanitize_identifier(node.id)
            return ast.Attribute(value=container, attr=attr, ctx=ctx)
        return ast.Name(id=sanitized, ctx=ctx)

    def visit_Number(self, node: lua.Number) -> ast.Constant:
        return ast.Constant(node.n)

    def visit_String(self, node: lua.String) -> ast.Constant:
        return ast.Constant(node.s)

    def visit_Nil(self, _: lua.Nil) -> ast.Constant:
        return ast.Constant(None)

    def visit_TrueExpr(self, _: lua.TrueExpr) -> ast.Constant:
        return ast.Constant(True)

    def visit_FalseExpr(self, _: lua.FalseExpr) -> ast.Constant:
        return ast.Constant(False)

    def visit_Table(self, node: lua.Table) -> ast.expr:
        field_exprs = []
        dict_keys: List[ast.expr] = []
        dict_values: List[ast.expr] = []
        list_values: List[ast.expr] = []
        has_keyed = False
        has_sequential = False

        for field in node.fields:
            if not isinstance(field, lua.Field):
                raise TransformerError("Unsupported table field")
            is_sequence_field = False
            if field.key is None:
                is_sequence_field = True
            elif isinstance(field.key, lua.Number):
                first_token = getattr(field.key, "_first_token", None)
                last_token = getattr(field.key, "_last_token", None)
                is_sequence_field = first_token is None and last_token is None

            if field.key:
                if isinstance(field.key, lua.Name):
                    key = ast.Constant(field.key.id)
                else:
                    key = self.visit_expression(field.key)
            else:
                key = ast.Constant(None)
            value = self.visit_expression(field.value)
            is_keyed = not is_sequence_field

            if is_keyed:
                has_keyed = True
                dict_keys.append(key)
                dict_values.append(value)
            else:
                has_sequential = True
                list_values.append(value)

            field_exprs.append(
                ast.Tuple(elts=[ast.Constant(is_keyed), key, value], ctx=ast.Load())
            )

        return self._runtime_call(
            "table_ctor", ast.List(elts=field_exprs, ctx=ast.Load())
        )

    def visit_Field(self, node: lua.Field):
        return node  # handled by visit_Table

    def visit_Varargs(self, _: lua.Varargs) -> ast.Name:
        return ast.Name(id="args", ctx=ast.Load())

    def visit_AnonymousFunction(self, node: lua.AnonymousFunction) -> ast.Lambda:
        args, arg_names = self._build_arguments(node.args)
        body = self._function_block(node.body, initial_locals=arg_names)
        func_name = self._fresh_function_name()
        func_def = ast.FunctionDef(
            name=func_name,
            args=args,
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        if not self._inline_stack:
            raise TransformerError("Anonymous functions cannot be hoisted without a block context.")
        self._inline_stack[-1].append(func_def)
        return ast.Name(id=func_name, ctx=ast.Load())

    def _index_components(self, node: lua.Index) -> tuple[ast.expr, ast.expr]:
        value_expr = self.visit_expression(node.value)
        if node.notation == lua.IndexNotation.DOT and isinstance(node.idx, lua.Name):
            key_expr = ast.Constant(node.idx.id)
        else:
            key_expr = self.visit_expression(node.idx)
        return value_expr, key_expr

    def visit_Index(self, node: lua.Index, ctx: Optional[ast.expr_context] = None):
        value_expr, key_expr = self._index_components(node)
        return ast.Subscript(value=value_expr, slice=key_expr, ctx=ctx or ast.Load())

    # -- calls / functions ----------------------------------------------

    def _build_arguments(self, args: Sequence[lua.Expression]) -> tuple[ast.arguments, List[str]]:
        py_args: List[ast.arg] = []
        vararg: Optional[ast.arg] = None
        locals_list: List[str] = []
        for arg in args:
            if isinstance(arg, lua.Dots):
                vararg = ast.arg(arg="args")
                locals_list.append("args")
            elif isinstance(arg, lua.Name):
                sanitized = self._sanitize_identifier(arg.id)
                py_args.append(ast.arg(arg=sanitized))
                locals_list.append(sanitized)
            else:
                raise TransformerError(f"Unsupported function argument {arg!r}")
        arguments = ast.arguments(
            posonlyargs=[],
            args=py_args,
            vararg=vararg or ast.arg(arg="__lua_extra_args"),
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        self._apply_default_placeholders(arguments)
        return arguments, locals_list

    def _build_call(
        self,
        func_expr: lua.Expression,
        args: Sequence[lua.Expression],
    ) -> ast.Call:
        func = self.visit_expression(func_expr)
        return ast.Call(func=func, args=self._expr_list(args), keywords=[])

    # -- operators ------------------------------------------------------

    def visit_AddOp(self, node: lua.AddOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.Add(), right=self.visit_expression(node.right)
        )

    def visit_SubOp(self, node: lua.SubOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.Sub(), right=self.visit_expression(node.right)
        )

    def visit_MultOp(self, node: lua.MultOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.Mult(), right=self.visit_expression(node.right)
        )

    def visit_FloatDivOp(self, node: lua.FloatDivOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.Div(), right=self.visit_expression(node.right)
        )

    def visit_FloorDivOp(self, node: lua.FloorDivOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.FloorDiv(), right=self.visit_expression(node.right)
        )

    def visit_ModOp(self, node: lua.ModOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.Mod(), right=self.visit_expression(node.right)
        )

    def visit_ExpoOp(self, node: lua.ExpoOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.Pow(), right=self.visit_expression(node.right)
        )

    def visit_BAndOp(self, node: lua.BAndOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.BitAnd(), right=self.visit_expression(node.right)
        )

    def visit_BOrOp(self, node: lua.BOrOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.BitOr(), right=self.visit_expression(node.right)
        )

    def visit_BXorOp(self, node: lua.BXorOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.BitXor(), right=self.visit_expression(node.right)
        )

    def visit_BShiftLOp(self, node: lua.BShiftLOp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.LShift(), right=self.visit_expression(node.right)
        )

    def visit_BShiftROp(self, node: lua.BShiftROp) -> ast.BinOp:
        return ast.BinOp(
            left=self.visit_expression(node.left), op=ast.RShift(), right=self.visit_expression(node.right)
        )

    # Relational
    def visit_LessThanOp(self, node: lua.LessThanOp) -> ast.Compare:
        return self._compare(node, ast.Lt())

    def visit_GreaterThanOp(self, node: lua.GreaterThanOp) -> ast.Compare:
        return self._compare(node, ast.Gt())

    def visit_LessOrEqThanOp(self, node: lua.LessOrEqThanOp) -> ast.Compare:
        return self._compare(node, ast.LtE())

    def visit_GreaterOrEqThanOp(self, node: lua.GreaterOrEqThanOp) -> ast.Compare:
        return self._compare(node, ast.GtE())

    def visit_EqToOp(self, node: lua.EqToOp) -> ast.Compare:
        return self._compare(node, ast.Eq())

    def visit_NotEqToOp(self, node: lua.NotEqToOp) -> ast.Compare:
        return self._compare(node, ast.NotEq())

    def _compare(self, node: lua.BinaryOp, op: ast.cmpop) -> ast.Compare:
        return ast.Compare(
            left=self.visit_expression(node.left),
            ops=[op],
            comparators=[self.visit_expression(node.right)],
        )

    def visit_Concat(self, node: lua.Concat) -> ast.Call:
        return self._runtime_call(
            "concat",
            self.visit_expression(node.left),
            self.visit_expression(node.right),
        )

    def visit_UMinusOp(self, node: lua.UMinusOp) -> ast.UnaryOp:
        return ast.UnaryOp(op=ast.USub(), operand=self.visit_expression(node.operand))

    def visit_UBNotOp(self, node: lua.UBNotOp) -> ast.UnaryOp:
        return ast.UnaryOp(op=ast.Invert(), operand=self.visit_expression(node.operand))

    def visit_ULNotOp(self, node: lua.ULNotOp) -> ast.Call:
        return self._runtime_call("lua_not", self.visit_expression(node.operand))

    def visit_ULengthOP(self, node: lua.ULengthOP) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="len", ctx=ast.Load()),
            args=[self.visit_expression(node.operand)],
            keywords=[],
        )

    def visit_AndLoOp(self, node: lua.AndLoOp) -> ast.Call:
        return ast.BoolOp(
            op=ast.And(),
            values=[self.visit_expression(node.left), self.visit_expression(node.right)],
        )

    def visit_OrLoOp(self, node: lua.OrLoOp) -> ast.Call:
        return ast.BoolOp(
            op=ast.Or(),
            values=[self.visit_expression(node.left), self.visit_expression(node.right)],
        )

    # def visit_AndLoOp(self, node: lua.AndLoOp) -> ast.Call:
    #     return self._wrap_bool_op(
    #         "lua_and",
    #         self.visit_expression(node.left),
    #         self.visit_expression(node.right),
    #     )

    # def visit_OrLoOp(self, node: lua.OrLoOp) -> ast.Call:
    #     return self._wrap_bool_op(
    #         "lua_or",
    #         self.visit_expression(node.left),
    #         self.visit_expression(node.right),
    #     )

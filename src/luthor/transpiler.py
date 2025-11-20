"""User-facing Lua→Python transpiler facade."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from antlr4 import CommonTokenStream, InputStream, Token
from antlr4.error.ErrorListener import ConsoleErrorListener
from luaparser.ast import SyntaxException
from luaparser.astnodes import Chunk
from luaparser.builder import BuilderVisitor
from luaparser.parser.LuaLexer import LuaLexer
from luaparser.parser.LuaParser import LuaParser

from .transformer import LuaToPythonAstTransformer, TransformerConfig


@dataclass
class TranspileResult:
    """Holds the resulting python module and its rendered source."""

    module: ast.Module
    source: str


class LuaToPythonTranspiler:
    """Facade combining parsing, AST conversion, and code generation."""

    def __init__(self, *, pretty: bool = True, config: TransformerConfig | None = None) -> None:
        self._pretty = pretty
        self._transformer = LuaToPythonAstTransformer(config=config)

    def transpile_text(self, lua_code: str) -> TranspileResult:
        """Transpile Lua source text into Python source."""
        chunk = _parse_lua(lua_code)
        module = self._transformer.transform(chunk)
        ast.fix_missing_locations(module)
        source = self._render_python(module)
        return TranspileResult(module=module, source=source)

    def transpile_file(self, input_path: Path) -> TranspileResult:
        """Read Lua code from disk and transpile it."""
        return self.transpile_text(input_path.read_text())

    def _render_python(self, module: ast.Module) -> str:
        if self._pretty:
            return ast.unparse(module)
        return compile(module, filename="<luthor>", mode="exec")  # pragma: no cover


def _parse_lua(source: str) -> Chunk:
    """Minimal wrapper around py-lua-parser without debug printing."""
    lexer = LuaLexer(InputStream(source))
    lexer.removeErrorListeners()
    lexer.addErrorListener(ConsoleErrorListener())

    token_stream = CommonTokenStream(lexer, channel=Token.DEFAULT_CHANNEL)
    parser = LuaParser(token_stream)
    parser.addErrorListener(ConsoleErrorListener())
    tree = parser.start_()

    if parser.getNumberOfSyntaxErrors() > 0:
        raise SyntaxException("syntax errors")

    visitor = BuilderVisitor(token_stream)
    return visitor.visit(tree)

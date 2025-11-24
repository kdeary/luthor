"""Command line interface for Luthor, the Lu(a)(Py)tho(n) transpiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .transpiler import LuaToPythonTranspiler, TransformerConfig


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to the Lua file to transpile (defaults to stdin).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for the generated Python (defaults to stdout).",
    )
    parser.add_argument(
        "--init-all-globals",
        action="store_true",
        help='Preinitialize every global variable to "None" before executing user code.',
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = TransformerConfig(initialize_all_globals=args.init_all_globals)
    transpiler = LuaToPythonTranspiler(config=config)

    if args.input:
        source = Path(args.input).read_text()
    else:
        source = sys.stdin.read()

    result = transpiler.transpile_text(source)

    if args.output:
        Path(args.output).write_text(result.source + "\n")
    else:
        sys.stdout.write(result.source + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

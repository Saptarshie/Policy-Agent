"""Arithmetic the agent can actually be held to.

The agent has no calculator in its head: every "12 minus 7" it does by
generating tokens, and nothing records what it computed. This tool makes the
derivation explicit — expression in, value and normalised expression out — so a
reviewer can check the arithmetic instead of trusting the prose, and so a derived
number in the answer has a traceable origin.

Safety: the expression is parsed with :mod:`ast` and walked against an allowlist.
No attribute access, no calls except a small numeric set, no names except the
variables supplied in the same call. Nothing here can run code.
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Any, Dict

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MAX_EXPR_CHARS = 500
MAX_POW = 12          # 10**12 is plenty for money; stops 10**10**10


class CalcError(ValueError):
    pass


def _eval(node: ast.AST, names: Dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError(f"only numbers are allowed, got {type(node.value).__name__}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise CalcError(f"unknown variable '{node.id}' "
                            f"(supply it in variables; known: {sorted(names) or 'none'})")
        return float(names[node.id])
    if isinstance(node, ast.BinOp):
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise CalcError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _eval(node.left, names), _eval(node.right, names)
        if fn is operator.pow and (abs(right) > MAX_POW or abs(left) > 1e9):
            raise CalcError("exponent too large")
        if fn in (operator.truediv, operator.floordiv, operator.mod) and right == 0:
            raise CalcError("division by zero")
        return fn(left, right)
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise CalcError(f"unary {type(node.op).__name__} is not allowed")
        return fn(_eval(node.operand, names))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalcError(f"function not allowed (allowed: {sorted(_FUNCS)})")
        if node.keywords:
            raise CalcError("keyword arguments are not allowed")
        fn_name = node.func.id
        args = [_eval(a, names) for a in node.args]
        if fn_name == "round":
            # ndigits must be an int; the evaluator only ever produces floats
            if len(args) == 2:
                args[1] = int(args[1])
            elif len(args) > 2:
                raise CalcError("round() takes at most 2 arguments")
        return float(_FUNCS[fn_name](*args))
    raise CalcError(f"{type(node).__name__} is not allowed in an expression")


def calculate(expression: str, variables: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Evaluate *expression* with *variables* and return value + derivation."""
    expr = (expression or "").strip()
    if not expr:
        raise CalcError("empty expression")
    if len(expr) > MAX_EXPR_CHARS:
        raise CalcError(f"expression longer than {MAX_EXPR_CHARS} characters")
    names: Dict[str, float] = {}
    for k, v in (variables or {}).items():
        try:
            names[str(k)] = float(v)
        except (TypeError, ValueError):
            raise CalcError(f"variable '{k}' is not a number")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"could not parse '{expr}': {e.msg}")
    value = _eval(tree, names)
    # Compact derivation a reviewer can re-check by hand. Substitute whole
    # identifiers only: naive str.replace turns `days * rate` with a variable `d`
    # into "9ays * rate", and `days - day_off` with `day` into "2s - 2_off".
    shown = _IDENT_RE.sub(
        lambda m: (f"{names[m.group(0)]:g}"
                   if m.group(0) in names and m.group(0) not in _FUNCS
                   else m.group(0)),
        expr,
    )
    return {
        "expression": expr,
        "variables": {k: (int(v) if float(v).is_integer() else v) for k, v in names.items()},
        "value": int(value) if float(value).is_integer() else round(value, 6),
        "resolved": shown,
    }

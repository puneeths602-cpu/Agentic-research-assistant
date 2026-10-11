"""Tools the researcher agent may call. Only allow-listed tools are executable."""
import ast
import operator as op

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}


def calculator(expr: str) -> str:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("unsupported expression")
    return str(ev(ast.parse(expr, mode="eval").body))


ALLOWED_TOOLS = {"calculator": calculator}

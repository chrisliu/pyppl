import ast
import inspect
import random

from typing import Any, Self, TypeAlias, Union


class ProbVar:
    ...


class ProbBool(ProbVar):
    def __bool__(self: Self):
        raise NotImplementedError()


class Flip(ProbBool):
    def __init__(self: Self, prob_true: float = 0.5):
        self._prob_true: float = prob_true
        self._value: bool = random.random() < prob_true

    def __bool__(self: Self) -> bool:
        return self._value

    def __eq__(self: Self, other: Union[ProbBool, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) == bool(other)
        elif isinstance(other, bool):
            return bool(self) == other
        raise TypeError("{mytype} == {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))


PrimNumeric: TypeAlias = Union[int, float]


class Numeric(ProbVar):
    def __int__(self: Self) -> int:
        raise NotImplementedError()

    def __float__(self: Self) -> float:
        raise NotImplementedError()

    def __eq__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        raise NotImplementedError()

    def __lt__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        raise NotImplementedError()

    def __le__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        raise NotImplementedError()

    def __add__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __sub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __mul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __truediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __floordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __mod__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __radd__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __rsub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __rmul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __rtruediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __rfloordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()

    def __rmod__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        raise NotImplementedError()


class Integer(Numeric):
    ...


class Real(Numeric):
    ...


def observe(val):
    return val


class DataflowAnalyis:
    ...


class NotObservable:
    ...


class SamplingTransform(ast.NodeTransformer):
    # def visit_Call(self: Self, node: ast.Call) -> Union[ast.Call, ast.If]:
    #     callee = _get_reference(node.func)
    #     if callee is observe:
    #         print(ast.dump(node, indent=2))
    #         # Replace `observe(...)` with early exit logic.
    #         observe_node = ast.If(node,
    #                               [ast.Return(ast.Name(NotObservable.__name__,
    #                                                    ctx=ast.Load))
    #                                ],
    #                               [])
    #         return ast.copy_location(observe_node, node)
    #     return node

    def visit_Expr(self: Self, stmt_node: ast.Expr) -> Union[ast.Expr, ast.If]:
        expr_node: ast.expr = stmt_node.value
        if isinstance(expr_node, ast.Call):
            callee = _get_reference(expr_node.func)
            if callee is observe:
                # Replace `observe(...)` with early exit logic.
                # ```
                # if not observe(...)
                #   return NotObservable
                # ```
                not_cond = ast.UnaryOp(ast.Not(), expr_node)
                if_observe_node = ast.If(not_cond,
                                         [ast.Return(
                                          ast.Name(NotObservable.__name__,
                                                   ctx=ast.Load()))
                                          ],
                                         [])
                return ast.copy_location(if_observe_node, expr_node)
        return stmt_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = [decorator
                               for decorator in node.decorator_list
                               if not isinstance(decorator, ast.Name) or
                               not decorator.id == pp_compile.__name__]
        return super().generic_visit(node)


def _get_reference(node: ast.expr) -> Any:
    # Warning: this only searches globals (should search in the appropriate
    # call frame instead).
    if isinstance(node, ast.Name):
        return globals()[node.id]
    elif isinstance(node, ast.Attribute):
        raise NotImplementedError("No support for attribute references")
    else:
        raise ValueError(f"Unrecognized refrence type {type(node)}")


def pp_compile(func):
    func_src = inspect.getsource(func)
    src_lines = [line for line in func_src.split('\n') if len(line) != 0]
    leading = [len(line) - len(line.lstrip(' ')) for line in src_lines]
    indent = min(leading)
    src_lines = [line[indent:] for line in src_lines]
    func_src = '\n'.join(src_lines)
    # print(func_src)

    # Insert correct logic for observe statements.
    mode = 'exec'
    func_ast = ast.parse(func_src, mode=mode)
    # print(ast.dump(func_ast, indent=2))
    transformed_ast = SamplingTransform().visit(func_ast)
    transformed_ast = ast.fix_missing_locations(transformed_ast)
    # print(ast.dump(transformed_ast, indent=2))
    # print(ast.unparse(transformed_ast))

    # Compile an executable version of the transformed function.
    # TODO: Use inspect stack frame local scope.
    cc = compile(transformed_ast, filename='<ast>', mode=mode)
    exec(cc)  # Save function in local context.
    transformed_func = locals()[func.__name__]

    # Exact Inference stuff

    # cfg = pycfg.gen_cfg(func_src)
    # sources = [k for k, v in cfg.items() if len(v.parents) == 0]
    # if len(sources) > 1:
    #     raise RuntimeError("PyPPL does not support nested functions")

    # for k, v in cfg.items():
    #     print(ast.dump(v.ast_node))
    #     print(k, len(v.parents), len(v.children))
    # print(cfg)

    # @functools.wraps
    def contextual_execution(*args: Any, **kwargs: Any):
        # Sample logic goes here.
        # TODO: only returns one sample right now, not a distribution.
        return transformed_func(*args, **kwargs)

    return contextual_execution


if __name__ == '__main__':
    @pp_compile
    def test_flip():
        f = Flip()
        observe(f)
        if f and Flip():
            a = True
            b = False
        else:
            a = False
            b = True
        return a, b

    for _ in range(10):
        sample = test_flip()
        if sample is NotObservable:
            print("Not observed")
        else:
            print(sample)

    # for _ in range(8):
    #     if Flip() != Flip():
    #         print("True")
    #     else:
    #         print("False")

    # @compile
    # def test_loop():
    #     success = True
    #     for i in range(RandomInteger(5)):
    #         success &= Flip()
    #     return success

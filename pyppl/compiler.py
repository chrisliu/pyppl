import ast
import inspect
import random
import pycfg.pycfg as pycfg

from typing import Any, Dict, List, Self, TypeAlias, Union


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


def compile(func):
    func_src = inspect.getsource(func)
    src_lines = [line for line in func_src.split('\n') if len(line) != 0]
    leading = [len(line) - len(line.lstrip(' ')) for line in src_lines]
    indent = min(leading)
    src_lines = [line[indent:] for line in src_lines]
    func_src = '\n'.join(src_lines)
    print(func_src)

    cfg = pycfg.gen_cfg(func_src)
    sources = [k for k, v in cfg.items() if len(v.parents) == 0]
    if len(sources) > 1:
        raise RuntimeError("PyPPL does not support nested functions")

    for k, v in cfg.items():
        print(ast.dump(v.ast_node))
        print(k, len(v.parents), len(v.children))
    print(cfg)
    return func


class SamplingTransform(ast.NodeTransformer):
    ...


if __name__ == '__main__':
    @ compile
    def test_flip():
        f = Flip()
        observe(f is True)
        if f and Flip():
            a = True
            b = False
        else:
            a = False
            b = True

        for i in range(RandomInteger()):
            j = Flip()

        return a, b, j

    for _ in range(8):
        if Flip() != Flip():
            print("True")
        else:
            print("False")

    # @compile
    # def test_loop():
    #     success = True
    #     for i in range(RandomInteger(5)):
    #         success &= Flip()
    #     return success

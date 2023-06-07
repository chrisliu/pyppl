import random
import numpy as np
import math

from typing import Union, Self, TypeAlias


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
    def __init__(self: Self, value: int):
        self.val = value

    def __int__(self: Self) -> int:
        return int(self.val)

    def __float__(self: Self) -> float:
        return float(self.val)

    def __eq__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self.val == other
        return self.val == other.val

    def __lt__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self.val < other
        return self.val < other.val

    def __le__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self.val <= other
        return self.val <= other.val

    def __add__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val + other
        return self.val + other.val

    def __sub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val - other
        return self.val - other.val

    def __mul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val * other
        return self.val * other.val

    def __truediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(self) / float(other)

    def __floordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(self) / float(other))

    def __mod__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, float):
            raise NotImplementedError()
        return int(self) % int(other)

    def __radd__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val + other
        return self.val + other.val

    def __rsub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val - other
        return self.val - other.val

    def __rmul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val * other
        return self.val * other.val

    def __rtruediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(other) / float(self)

    def __rfloordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(other) / float(self))

    def __rmod__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, float):
            raise NotImplementedError()
        return int(other) % int(self)


class Real(Numeric):
    def __init__(self: Self, value: int):
        self.val = value

    def __int__(self: Self) -> int:
        return int(self.val)

    def __float__(self: Self) -> float:
        return float(self.val)

    def __eq__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self.val == other
        return self.val == other.val

    def __lt__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self.val < other
        return self.val < other.val

    def __le__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self.val <= other
        return self.val <= other.val

    def __add__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val + other
        return self.val + other.val

    def __sub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val - other
        return self.val - other.val

    def __mul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val * other
        return self.val * other.val

    def __truediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(self) / float(other)

    def __floordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(self) / float(other))

    def __radd__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val + other
        return self.val + other.val

    def __rsub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val - other
        return self.val - other.val

    def __rmul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self.val * other
        return self.val * other.val

    def __rtruediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(other) / float(self)

    def __rfloordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(other) / float(self))

import random
# import numpy as np
import math

from typing import Any, Dict, Union, Self, TypeAlias


class ProbVar:
    @property
    def distribution(self: Self) -> Dict[Any, float]:
        raise NotImplementedError()

    @distribution.setter
    def distribution(self: Self, dist: Dict[Any, float]) -> None:
        # TODO: Update distribution on assign.
        raise NotImplementedError()

    def _is_valid_distribution(self: Self, dist: Dict[Any, float]) -> bool:
        return sum(dist.values()) == 1

    def __repr__(self: Self) -> str:
        dist_as_str = {str(k): str(v) for k, v in self.distribution.items()}
        longest_key_len = max(len(k) for k in dist_as_str.keys())
        dist_as_str = {k.ljust(longest_key_len): v
                       for k, v in dist_as_str.items()}
        return '{{\n{dist}\n}}'.format(
            dist='\n'.join(f'  {k}: {v}' for k, v in dist_as_str.items()))


class ProbBool(ProbVar):
    def __init__(self: Self, prob_true: float = 0.5) -> None:
        self._prob_true: float = prob_true

    def __bool__(self: Self) -> bool:
        raise NotImplementedError()

    @property
    def distribution(self: Self) -> Dict[bool, float]:
        return {
            True: self._prob_true,
            False: (1 - self._prob_true),
        }

    @distribution.setter
    def distribution(self: Self, dist: Dict[bool, float]) -> None:
        if not self._is_valid_distribution(dist):
            raise ValueError(f"Bad distribution {dist}")
        if True in dist:
            self._prob_true = dist[True]
        else:
            raise ValueError(f"Inapprporiate boolean distribution {dist}")
        


class Flip(ProbBool):
    def __init__(self: Self, prob_true: float = 0.5) -> None:
        super().__init__(prob_true)
        self._value: bool = random.random() < self._prob_true

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

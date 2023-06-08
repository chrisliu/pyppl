import functools
import random
import numpy as np
import math

from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, Self, TypeAlias


class ProbVar:
    def __init__(self: Self) -> None:
        self._distribution: Optional['Distribution'] = None

    @property
    def distribution(self: Self) -> Optional['Distribution']:
        return self._distribution

    @distribution.setter
    def distribution(self: Self, distribution: 'Distribution') -> None:
        self._distribution_setter_impl(distribution)

    def _distribution_setter_impl(self: Self, distribution: 'Distribution'
                                  ) -> None:
        self._distribution = distribution

    def sample(self: Self) -> None:
        raise NotImplementedError()

    def __repr__(self: Self) -> str:
        return '{classname}({distribution})'.format(
            classname=type(self).__name__,
            distribution=self._distribution)


class ProbBool(ProbVar):
    def __init__(self: Self, prob_true: float = 0.5) -> None:
        self._prob_true: float = prob_true
        self.distribution = DiscreteDistribution({
            True: prob_true,
            False: (1 - prob_true),
        })
        self.sample()

    def _distribution_setter_impl(self: Self, dist: 'Distribution') -> None:
        if not isinstance(dist, DiscreteDistribution):
            raise ValueError(
                f"Inappropriate boolean distribution {type(dist)}")
        elif True in dist._distribution:
            self._prob_true = dist._distribution[True]
            super()._distribution_setter_impl(dist)
        else:
            raise ValueError(f"Inapprporiate boolean distribution {dist}")

    def sample(self: Self) -> None:
        self._value: bool = random.random() < self._prob_true

    def __bool__(self: Self) -> bool:
        return self._value

    def __eq__(self: Self, other: Union[Self, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) == bool(other)
        elif isinstance(other, bool):
            return bool(self) == other
        raise TypeError("{mytype} == {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))

    def __and__(self: Self, other: Union[Self, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) & bool(other)
        elif isinstance(other, bool):
            return bool(self) & other
        raise TypeError("{mytype} & {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))

    def __or__(self: Self, other: Union[Self, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) | bool(other)
        elif isinstance(other, bool):
            return bool(self) | other
        raise TypeError("{mytype} & {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))

    def __xor__(self: Self, other: Union[Self, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) ^ bool(other)
        elif isinstance(other, bool):
            return bool(self) ^ other
        raise TypeError("{mytype} & {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))

    def __rand__(self: Self, other: Union[Self, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) & bool(other)
        elif isinstance(other, bool):
            return bool(self) & other
        raise TypeError("{mytype} & {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))

    def __ror__(self: Self, other: Union[Self, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) | bool(other)
        elif isinstance(other, bool):
            return bool(self) | other
        raise TypeError("{mytype} & {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))

    def __rxor__(self: Self, other: Union[Self, bool]) -> bool:
        if isinstance(other, ProbBool):
            return bool(self) ^ bool(other)
        elif isinstance(other, bool):
            return bool(self) ^ other
        raise TypeError("{mytype} & {othertype} unsupported".format(
            mytype=type(self), othertype=type(other)))


class Flip(ProbBool):
    ...


PrimNumeric: TypeAlias = Union[int, float]


class Numeric(ProbVar):
    def __init__(self, distribution: Optional['Distribution'] = None) -> None:
        self._distribution = distribution
        self._value: Optional[PrimNumeric] = None
        if self._distribution is not None:
            self._distribution._set_prob_var_type(type(self))
            self._value = self.sample()

    @staticmethod
    def _has_value(numeric_method):
        @functools.wraps(numeric_method)
        def check_value(*args, **kwargs):
            self = args[0]
            if self._value is None:
                raise AttributeError("{classname} not initialized".format(
                    classname=type(self).__name__))
            return numeric_method(*args, **kwargs)
        return check_value

    @property
    def distribution(self: Self) -> Optional['Distribution']:
        return self._distribution

    @distribution.setter
    def distribution(self: Self, distribution: 'Distribution') -> None:
        self._distribution = distribution

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

    def __gt__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        raise NotImplementedError()

    def __ge__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
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
    def __init__(self: Self, distribution: Optional['Distribution'] = None
                 ) -> None:
        super().__init__(distribution)
        self._value: int  # actually is a Optional[int]

    def sample(self: Self) -> int:
        if self._distribution is None:
            raise AttributeError("Distribution is not set")
        return int(self._distribution.sample())

    @Numeric._has_value
    def __int__(self: Self) -> int:
        return int(self._value)

    @Numeric._has_value
    def __float__(self: Self) -> float:
        return float(self._value)

    @Numeric._has_value
    def __eq__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value == other
        return self._value == other._value

    @Numeric._has_value
    def __lt__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value < other
        return self._value < other._value

    @Numeric._has_value
    def __le__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value <= other
        return self._value <= other._value

    @Numeric._has_value
    def __gt__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value > other
        return self._value > other._value

    @Numeric._has_value
    def __ge__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value >= other
        return self._value >= other._value

    @Numeric._has_value
    def __add__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value + other
        return self._value + other._value

    @Numeric._has_value
    def __sub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value - other
        return self._value - other._value

    @Numeric._has_value
    def __mul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value * other
        return self._value * other._value

    @Numeric._has_value
    def __truediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(self) / float(other)

    @Numeric._has_value
    def __floordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(self) / float(other))

    @Numeric._has_value
    def __mod__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, float):
            raise NotImplementedError()
        return int(self) % int(other)

    @Numeric._has_value
    def __radd__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value + other
        return self._value + other._value

    @Numeric._has_value
    def __rsub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value - other
        return self._value - other._value

    @Numeric._has_value
    def __rmul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value * other
        return self._value * other._value

    @Numeric._has_value
    def __rtruediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(other) / float(self)

    @Numeric._has_value
    def __rfloordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(other) / float(self))

    @Numeric._has_value
    def __rmod__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, float):
            raise NotImplementedError()
        return int(other) % int(self)


class Real(Numeric):
    def __init__(self: Self, distribution: 'Distribution') -> None:
        super().__init__(distribution)
        self._value: float

    def sample(self: Self) -> float:
        if self._distribution is None:
            raise AttributeError("Distribution is not set")
        return float(self._distribution.sample())

    def __int__(self: Self) -> int:
        return int(self._value)

    def __float__(self: Self) -> float:
        return float(self._value)

    def __eq__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value == other
        return self._value == other._value

    def __lt__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value < other
        return self._value < other._value

    def __le__(self: Self, other: Union[Self, PrimNumeric]) -> bool:
        if isinstance(other, PrimNumeric):
            return self._value <= other
        return self._value <= other._value

    def __add__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value + other
        return self._value + other._value

    def __sub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value - other
        return self._value - other._value

    def __mul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value * other
        return self._value * other._value

    def __truediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(self) / float(other)

    def __floordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(self) / float(other))

    def __radd__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value + other
        return self._value + other._value

    def __rsub__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value - other
        return self._value - other._value

    def __rmul__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        if isinstance(other, PrimNumeric):
            return self._value * other
        return self._value * other._value

    def __rtruediv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return float(other) / float(self)

    def __rfloordiv__(self: Self, other: Union[Self, PrimNumeric]) -> PrimNumeric:
        return math.floor(float(other) / float(self))


T = TypeVar('T')


class Distribution(Generic[T]):
    def __init__(self):
        self._prob_var_type: Optional[Type[ProbVar]] = None

    def _set_prob_var_type(self: Self, prob_ty: Type[ProbVar]) -> None:
        if not isinstance(prob_ty, type) or not issubclass(prob_ty, ProbVar):
            raise ValueError(
                f"{prob_ty} is not a probabilistic variable type.")
        self._prob_var_type = prob_ty

    def sample(self: Self) -> T:
        raise NotImplementedError

    def __repr__(self: Self) -> str:
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self: Self, a: PrimNumeric, b: PrimNumeric):
        """A uniform distribution of range [a, b]."""
        self._start = a
        self._stop = b

    def sample(self: Self) -> float:
        if self._prob_var_type is not None and\
                issubclass(self._prob_var_type, Integer):
            return random.randint(int(self._start), int(self._stop))
        else:
            return random.uniform(float(self._start), float(self._stop))

    def __repr__(self: Self) -> str:
        return '{classname}(start={start}, stop={stop})'.format(
            classname=type(self).__name__,
            start=self._start,
            stop=self._stop)


class GaussianDistribution(Distribution):
    def __init__(self: Self, mu: PrimNumeric, sigma: PrimNumeric):
        """A gaussian distribution of range [a, b]."""
        self._mu = mu
        self._sigma = sigma

    def sample(self: Self) -> float:
        return random.gauss(float(self._mu), float(self._sigma))

    def __repr__(self: Self) -> str:
        return '{classname}(mu={mu}, sigma={sigma})'.format(
            classname=type(self).__name__,
            mu=self._mu,
            sigma=self._sigma)


class DiscreteDistribution(Distribution):
    def __init__(self: Self, distribution: Dict[Any, float]) -> None:
        if not DiscreteDistribution._is_valid_distribution(distribution):
            raise ValueError(f"Bad distribution {distribution}")
        self._distribution = distribution

    @staticmethod
    def _is_valid_distribution(distribution: Dict[Any, float]) -> bool:
        return math.isclose(sum(distribution.values()), 1)

    def sample(self: Self) -> Any:
        assert len(self._distribution) != 0
        rand_val = random.random()
        postfix_sum = 0
        for k, v in self._distribution.items():
            postfix_sum += v
            if rand_val < postfix_sum:
                return k
        print("Warning: discrete distribution sample shouldn't be out of bounds")
        return k

    def __repr__(self: Self) -> str:
        ordered_keys = sorted(self._distribution.keys())
        dist_as_str = {str(k): f'{self._distribution[k]:0.3f}'
                       for k in ordered_keys}
        longest_key_len = max(len(k) for k in dist_as_str.keys())
        dist_as_str = {k.ljust(longest_key_len): v
                       for k, v in dist_as_str.items()}
        return '{classname}({{\n{dist}\n}})'.format(
            classname=type(self).__name__,
            dist='\n'.join(f'  {k}: {v}' for k, v in dist_as_str.items()))

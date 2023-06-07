import contextvars
import collections

from pyppl.lang import NotObservable
from pyppl.types import ProbVar
from types import TracebackType
from typing import Any, Callable, Optional, Self, Type, Union

_inference_technique: contextvars.ContextVar[Optional['InferenceTechnique']] =\
    contextvars.ContextVar("inf_tech", default=None)


def _cur_inference_technique() -> Optional['InferenceTechnique']:
    return _inference_technique.get()


class InferenceTechnique:
    def __enter__(self: Self) -> None:
        self._token = _inference_technique.set(self)

    def __exit__(self: Self, exc_type: Union[Type[BaseException], None],
                 exc_val: Union[BaseException, None],
                 exc_tb: Union[TracebackType, None]) -> None:
        _inference_technique.reset(self._token)


class SamplingInference(InferenceTechnique):
    __default_num_samples = int(1e4)

    def __init__(self: Self, num_samples: Optional[int] = None) -> None:
        if num_samples is None:
            num_samples = SamplingInference.__default_num_samples
        if num_samples <= 0:
            raise ValueError(
                "Sampling inference must take at least one sample.")
        self._num_samples = num_samples

    # TODO: Only assumes one return type for now.
    def sample(self: Self, func: Callable, return_types: Type[ProbVar],
               *args: Any, **kwargs: Any,) -> ProbVar:
        raise NotImplementedError()


class RejectionSampling(SamplingInference):
    def __init__(self, num_samples: Optional[int] = None) -> None:
        super().__init__(num_samples)

    def sample(self: Self, func: Callable, return_types: Type[ProbVar],
               *args: Any, **kwargs: Any,) -> ProbVar:
        # Sample self._num_samples executions.
        distribution = collections.defaultdict(int)
        for _ in range(self._num_samples):
            sample = func(*args, **kwargs)
            if sample is not NotObservable:
                distribution[sample] += 1

        # Normalize.
        actual_num_samples = sum(distribution.values())
        distribution = {k: v / actual_num_samples
                        for k, v in distribution.items()}

        prob_var = return_types()
        prob_var.distribution = distribution

        return prob_var


class ExactInference(InferenceTechnique):
    def __init__(self: Self) -> None:
        pass

    def exact_prob(self: Self, contextual_execution):
        record = collections.defaultdict(float)

        """
        Requires code to be more complete but general thought process as follows

        Construct execution graph to determine all paths
        Take probability of all paths from flip/numeric, Ex.
        if flip():
            return a
        else:
            return b
        constructs record probabiltiy of if path = probability of true flip, other false flip for else path
        Add values directly into record and return it
        """
        return None

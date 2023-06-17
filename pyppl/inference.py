import contextvars
import collections

from pyppl.lang import NotObservable
from pyppl.types import ProbVar, Flip, DiscreteDistribution
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Self, Type, Union

import numpy as np

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
        prob_var.distribution = DiscreteDistribution(distribution)

        return prob_var


class MCMC(SamplingInference):
    def __init__(self, num_samples: Optional[int] = None) -> None:
        super().__init__(num_samples)
        self.mu = 0
        self.sigma = 1
        self.target_dist = lambda x: np.exp(
            -0.5 * ((x - self.mu) / self.sigma) ** 2) / (np.sqrt(2 * np.pi) * self.sigma)
        self.state = 0

    def sample(self: Self, func: Callable, return_types: Type[ProbVar],
               *args: Any, **kwargs: Any,) -> ProbVar:
        distribution = collections.defaultdict(int)

        for _ in range(self._num_samples):
            proposal = func(*args, **kwargs)

            if proposal is not NotObservable:
                acceptance = min(1, self.target_dist(
                    proposal) / self.target_dist(self.state))
                if np.random.uniform(0, 1) < acceptance:
                    self.state = proposal
                distribution[self.state] += 1

        actual_num_samples = sum(distribution.values())
        distribution = {k: v / actual_num_samples
                        for k, v in distribution.items()}
        prob_var = return_types()
        prob_var.distribution = DiscreteDistribution(distribution)

        return prob_var


class ExactInference(InferenceTechnique):
    def __init__(self: Self) -> None:
        pass

    def infer(self: Self,
              sat_cond: Union[List[Dict[str, int]], bool],
              sat_lbls: List[str],
              get_prob_vars: Callable,
              *args: Any,
              **kwargs: Any
              ) -> Flip:
        # If always true or false, return that.
        if isinstance(sat_cond, bool):
            prob_true = 1 if sat_cond else 0
            return Flip(prob_true)

        prob_vars: List[Flip] = get_prob_vars(*args, **kwargs)
        sat_refs = dict(zip(sat_lbls, prob_vars))

        prob_true = 0
        for cond in sat_cond:
            prob_cond = 1
            for lbl, true_or_false in cond.items():
                true_or_false = true_or_false == 1
                prob_var = sat_refs[str(lbl)]
                prob_var_true = prob_var._prob_true
                prob_cond *= prob_var_true if true_or_false else \
                    (1 - prob_var_true)
            prob_true += prob_cond
        return Flip(prob_true)

import contextvars
import collections

from pyppl.lang import NotObservable
from typing import Optional

_inference_technique: contextvars.ContextVar[Optional['Technique']] = \
    contextvars.ContextVar("inf_tech", default=None)


class Technique:
    def __enter__(self):
        self._token = _inference_technique.set(self)

    def __exit__(self):
        _inference_technique.reset(self._token)


class Sampling(Technique):
    __default_num_samples = int(1e4)

    def __init__(self, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = Sampling.__default_num_samples
        assert num_samples > 0
        self.num_samples = num_samples

    def sample(self, func, *args, **kwargs):
        record = collections.defaultdict(int)
        for _ in range(self.num_samples):
            res = func(*args, **kwargs)
            if not isinstance(res, NotObservable):
                record[res] += 1
        return self._normalize(record)

    def _normalize(self, record):
        for key in record.keys():
            record[key] /= self.num_samples
        return record


class ExactInference(Technique):
    def __init__(self):
        pass

    def exact_prob(self, contextual_execution):
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

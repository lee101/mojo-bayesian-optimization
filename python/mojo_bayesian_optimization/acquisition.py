"""Acquisition functions with the elementwise kernels in Mojo.

The shapes and formulas mirror `bayes_opt.acquisition`, and the tests compare
`base_acq` against the real classes directly. What differs is mechanical:

- `suggest` minimises over a batch of uniform random candidates instead of
  running upstream's L-BFGS-B refinement afterwards. That refinement is a
  scipy call on a non-convex objective; porting it would mean porting scipy.
  The random sweep is the part that touches every candidate's acquisition
  value, and that is the part in the compiled library.
- The kernels take a thread count, so a large candidate batch can be split
  across cores. At the default 10 000 candidates and a few dozen observations
  that is still a small amount of work, and the default is one worker.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import _lib
from .gp import GaussianProcess

__all__ = [
    "AcquisitionFunction",
    "ExpectedImprovement",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
]


class AcquisitionFunction:
    """Base class: a function of the predictive mean and standard deviation.

    Targets are treated as values to **minimise**, which is the convention
    `bayes_opt` uses internally by storing the negation of the objective.
    """

    def __init__(self, n_workers: int = 1) -> None:
        self.i = 0
        self.n_workers = int(n_workers)

    def base_acq(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_acquisition_params(self) -> dict:
        raise NotImplementedError(
            "custom AcquisitionFunction subclasses must implement "
            "get_acquisition_params"
        )

    def set_acquisition_params(self, params: dict) -> None:
        raise NotImplementedError(
            "custom AcquisitionFunction subclasses must implement "
            "set_acquisition_params"
        )

    def decay_exploration(self) -> None:
        """Hook called once per `suggest`; the base class does nothing."""

    def suggest(
        self,
        gp: GaussianProcess,
        bounds,
        n_random: int = 10_000,
        random_state=None,
        n_workers: int | None = None,
    ) -> np.ndarray:
        """Return the candidate minimising the (negated) acquisition value.

        `bounds` is an `(d, 2)` array of lower and upper limits per dimension.
        """
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (d, 2)")
        if gp.X_train_ is None:
            raise ValueError("fit the GaussianProcess before calling suggest")
        self.i += 1
        rng = np.random.RandomState(random_state)
        candidates = rng.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_random, bounds.shape[0])
        )
        mean, std = gp.predict(candidates, return_std=True, n_workers=n_workers)
        values = self.base_acq(mean, std)
        best, _ = _lib.argmin(values, n_workers=gp.n_workers)
        if best < 0:
            raise ValueError("no candidates were generated")
        self.decay_exploration()
        return candidates[best]


class UpperConfidenceBound(AcquisitionFunction):
    r"""Upper confidence bound, :math:`\mu(x) + \kappa \sigma(x)`.

    Lower `kappa` prefers exploitation, higher prefers exploration.
    """

    def __init__(
        self,
        kappa: float = 2.576,
        exploration_decay: float | None = None,
        exploration_decay_delay: int | None = None,
        n_workers: int = 1,
    ) -> None:
        if kappa < 0:
            raise ValueError("kappa must be greater than or equal to 0.")
        if exploration_decay is not None and not (0 < exploration_decay <= 1):
            raise ValueError(
                "exploration_decay must be greater than 0 and less than or equal to 1."
            )
        if exploration_decay_delay is not None and (
            not isinstance(exploration_decay_delay, int) or exploration_decay_delay < 0
        ):
            raise ValueError(
                "exploration_decay_delay must be an integer greater than or equal to 0."
            )
        super().__init__(n_workers=n_workers)
        self.kappa = kappa
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay

    def base_acq(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return _lib.acq_ucb(mean, std, self.kappa, n_workers=self.n_workers)

    def decay_exploration(self) -> None:
        """Reduce kappa by a constant rate once the delay has passed."""
        if self.exploration_decay is not None and (
            self.exploration_decay_delay is None
            or self.exploration_decay_delay <= self.i
        ):
            self.kappa = self.kappa * self.exploration_decay

    def get_acquisition_params(self) -> dict:
        return {
            "kappa": self.kappa,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        self.kappa = params["kappa"]
        self.exploration_decay = params["exploration_decay"]
        self.exploration_decay_delay = params["exploration_decay_delay"]


class _Improvement(AcquisitionFunction):
    """Shared xi bookkeeping for the two improvement-based acquisitions."""

    def __init__(
        self,
        xi: float,
        exploration_decay: float | None = None,
        exploration_decay_delay: int | None = None,
        n_workers: int = 1,
    ) -> None:
        if xi < 0:
            raise ValueError("xi must be greater than or equal to 0.")
        if exploration_decay is not None and not (0 < exploration_decay <= 1):
            raise ValueError(
                "exploration_decay must be greater than 0 and less than or equal to 1."
            )
        if exploration_decay_delay is not None and (
            not isinstance(exploration_decay_delay, int) or exploration_decay_delay < 0
        ):
            raise ValueError(
                "exploration_decay_delay must be an integer greater than or equal to 0."
            )
        super().__init__(n_workers=n_workers)
        self.xi = xi
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay
        self.y_max = None

    def decay_exploration(self) -> None:
        if self.exploration_decay is not None and (
            self.exploration_decay_delay is None
            or self.exploration_decay_delay <= self.i
        ):
            self.xi = self.xi * self.exploration_decay

    def get_acquisition_params(self) -> dict:
        return {
            "xi": self.xi,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        self.xi = params["xi"]
        self.exploration_decay = params["exploration_decay"]
        self.exploration_decay_delay = params["exploration_decay_delay"]

    def _resolve_y_max(self, y: Iterable[float]) -> None:
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("cannot suggest a point without previous samples")
        self.y_max = float(values.min())


class ProbabilityOfImprovement(_Improvement):
    r"""Probability of improvement, :math:`\Phi((\mu - y_{best} - \xi)/\sigma)`."""

    def base_acq(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        if self.y_max is None:
            raise ValueError(
                "y_max is not set. If you are calling this method outside of "
                "suggest(), you must set y_max manually."
            )
        return _lib.acq_poi(mean, std, self.y_max, self.xi, n_workers=self.n_workers)

    def suggest(
        self,
        gp: GaussianProcess,
        bounds,
        y=None,
        n_random: int = 10_000,
        random_state=None,
        n_workers: int | None = None,
    ) -> np.ndarray:
        if y is None:
            y = gp.y_train_
        self._resolve_y_max(y)
        return super().suggest(gp, bounds, n_random, random_state, n_workers)


class ExpectedImprovement(_Improvement):
    r"""Expected improvement.

    :math:`a \Phi(a/\sigma) + \sigma \phi(a/\sigma)` with
    :math:`a = \mu - y_{best} - \xi`. The PDF term is what distinguishes this
    from the probability of improvement: a wide posterior can be worth more
    even when the probability of beating the incumbent is low.
    """

    def base_acq(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        if self.y_max is None:
            raise ValueError(
                "y_max is not set. If you are calling this method outside of "
                "suggest(), ensure y_max is set, or set it manually."
            )
        return _lib.acq_ei(mean, std, self.y_max, self.xi, n_workers=self.n_workers)

    def suggest(
        self,
        gp: GaussianProcess,
        bounds,
        y=None,
        n_random: int = 10_000,
        random_state=None,
        n_workers: int | None = None,
    ) -> np.ndarray:
        if y is None:
            y = gp.y_train_
        self._resolve_y_max(y)
        return super().suggest(gp, bounds, n_random, random_state, n_workers)

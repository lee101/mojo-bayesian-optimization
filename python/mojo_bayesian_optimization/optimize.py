"""The Bayesian optimisation loop, with the GP and acquisition in Mojo.

`maximize` is the shape of `bayes_opt.BayesianOptimization.maximize`: seed with
a few random samples, then repeatedly fit the surrogate, ask the acquisition
for the next point, and evaluate. Only the numeric core is ported; the domain
machinery around it (parameter types, discrete/categorical bounds, domain
reduction, logging, persistence) is `bayes_opt`'s and should stay there.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from .acquisition import AcquisitionFunction, ExpectedImprovement
from .gp import GaussianProcess

__all__ = ["Result", "maximize"]


class Result(NamedTuple):
    """Outcome of a `maximize` run."""

    x: np.ndarray
    """Best point found, in the original bounds."""
    y: float
    """Objective value at `x` (the minimum found)."""
    X: np.ndarray
    """Every evaluated point, in evaluation order."""
    Y: np.ndarray
    """Every evaluated objective value, in evaluation order."""

    @property
    def n_evaluations(self) -> int:
        return int(self.X.shape[0])


def maximize(
    func: Callable[[np.ndarray], float],
    bounds,
    n_iterations: int = 20,
    *,
    initial_points: int = 5,
    acq: AcquisitionFunction | None = None,
    gp: GaussianProcess | None = None,
    random_state=None,
) -> Result:
    """Minimise `func` over `bounds` by Gaussian-process Bayesian optimisation.

    Parameters
    ----------
    func :
        Called with a length-`d` array, returns a scalar to be **minimised**.
    bounds :
        `(d, 2)` array of lower and upper limits per dimension.
    n_iterations :
        Number of acquisition-driven probes after the initial random seed.
    initial_points :
        Number of uniform random probes used to seed the surrogate.
    acq, gp :
        Surrogate and acquisition to use; the defaults match `bayes_opt`'s
        Matern(5/2) GP with an expected-improvement acquisition.
    random_state :
        Seed, so a run is reproducible.

    Returns
    -------
    Result
        The best point, the full evaluation history, and both in order.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (d, 2)")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")
    if initial_points < 1:
        raise ValueError("initial_points must be at least 1")
    d = bounds.shape[0]
    if acq is None:
        acq = ExpectedImprovement(xi=0.01)
    if gp is None:
        gp = GaussianProcess(kernel="matern", length_scale=1.0, alpha=1e-6)
    rng = np.random.RandomState(random_state)

    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(initial_points, d))
    Y = np.array([float(func(row)) for row in X], dtype=np.float64)
    for _ in range(n_iterations):
        gp.fit(X, Y)
        nxt = acq.suggest(gp, bounds, random_state=rng.randint(0, 2**31 - 1))
        X = np.vstack([X, nxt.reshape(1, d)])
        Y = np.append(Y, float(func(nxt)))
    best = int(np.argmin(Y))
    return Result(x=X[best].copy(), y=float(Y[best]), X=X, Y=Y)

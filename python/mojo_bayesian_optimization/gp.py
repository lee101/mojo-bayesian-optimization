"""Gaussian process regression with the factorisation in Mojo.

This is the subset of `sklearn.gaussian_process.GaussianProcessRegressor` that
Bayesian optimisation actually exercises: a stationary kernel with a fixed
per-dimension length scale, a fixed noise level on the diagonal, optional
target standardisation, and the predictive mean and standard deviation.

What is deliberately not here is hyperparameter optimisation. Upstream
`bayes_opt` builds its regressor with scikit-learn's default
`optimizer="fmin_l_bfgs_b"`, which refits the kernel length scale by maximising
the marginal likelihood on every call. That is an L-BFGS-B loop over a
non-convex objective and it belongs to scipy; this port fixes the length scale
instead and says so. The marginal likelihood itself *is* computed, in Mojo, so
a caller who wants to drive their own optimiser has the objective.
"""

from __future__ import annotations

import numpy as np

from . import _lib

__all__ = ["GaussianProcess"]


class GaussianProcess:
    """Exact GP regression over a stationary kernel.

    Parameters
    ----------
    kernel :
        One of `"matern"` (nu = 5/2, the `bayes_opt` default), `"matern32"`,
        `"matern12"` or `"rbf"`.
    length_scale :
        Scalar, or one entry per input dimension for automatic relevance
        determination.
    alpha :
        Noise added to the diagonal of the kernel matrix. `bayes_opt` uses
        `1e-6`.
    normalize_y :
        Standardise the targets before fitting and un-standardise the predictive
        mean, matching scikit-learn's handling including its guard against a
        degenerate target spread.
    n_workers :
        Thread count for the candidate sweep in `predict`. 1 keeps the sweep on
        one core, which is the right default at realistic sizes.
    """

    def __init__(
        self,
        kernel: str = "matern",
        length_scale=1.0,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        n_workers: int = 1,
    ) -> None:
        if kernel not in _lib.KERNELS:
            raise ValueError(
                f"unknown kernel {kernel!r}; expected one of {sorted(_lib.KERNELS)}"
            )
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        ls = np.asarray(length_scale, dtype=np.float64)
        if ls.ndim > 1 or (ls.size > 1 and ls.ndim == 1 and np.any(ls <= 0.0)):
            raise ValueError("length_scale entries must be positive")
        if ls.ndim == 0 and float(ls) <= 0.0:
            raise ValueError("length_scale entries must be positive")
        self.kernel = kernel
        self.length_scale = length_scale
        self.alpha = float(alpha)
        self.normalize_y = bool(normalize_y)
        self.n_workers = int(n_workers)
        self.X_train_ = None
        self.y_train_ = None
        self.L_ = None
        self.alpha_ = None
        self.beta_ = None
        self._y_train_mean = 0.0
        self._y_train_std = 1.0

    # -- fitting ---------------------------------------------------------

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Cross-covariance between two point sets, shape (len(A), len(B))."""
        A = np.ascontiguousarray(A, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        na, d = A.shape
        nb = B.shape[0]
        if B.shape[1] != d:
            raise ValueError("X and X_new must have the same number of columns")
        return _lib.cross_kernel(A, B, self.length_scale, _lib.KERNELS[self.kernel])

    def fit(self, X, y) -> "GaussianProcess":
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if X.shape[0] != y.size:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("cannot fit a GP on an empty sample")
        self.X_train_ = X
        self.y_train_ = y
        if self.normalize_y:
            self._y_train_mean = float(y.mean())
            std = float(y.std())
            self._y_train_std = std if std > 1e-12 else 1.0
        else:
            self._y_train_mean = 0.0
            self._y_train_std = 1.0
        y_norm = (y - self._y_train_mean) / self._y_train_std
        K = self._kernel(X, X)
        K[np.diag_indices_from(K)] += self.alpha
        self.L_ = _lib.cholesky(K)
        self.alpha_, self.beta_ = _lib.cholesky_solve(self.L_, y_norm)
        return self

    # -- prediction ------------------------------------------------------

    def predict(self, X, return_std: bool = False, n_workers: int | None = None):
        """Predictive mean, and standard deviation when asked for."""
        if self.L_ is None:
            raise ValueError("fit must be called before predict")
        Xs = np.ascontiguousarray(X, dtype=np.float64)
        if Xs.ndim == 1:
            Xs = Xs.reshape(1, -1)
        m = Xs.shape[0]
        Kt = self._kernel(self.X_train_, Xs)
        # A stationary kernel has a constant diagonal, so k(x, x) is a scalar
        # broadcast rather than another O(n^2 * d) evaluation.
        kss = np.ones(m, dtype=np.float64)
        mu, std = _lib.gp_predict(
            self.L_, self.beta_, Kt, kss,
            n_workers=self.n_workers if n_workers is None else n_workers,
        )
        mu = mu * self._y_train_std + self._y_train_mean
        if return_std:
            # The variance comes out of the factorisation in normalised target
            # units, so it has to be rescaled by the target spread squared. The
            # mean needs the same factor linearly; doing one and not the other
            # is the classic way to get a plausible-looking but wrong
            # uncertainty, and it is invisible whenever normalize_y is False.
            return mu, std * self._y_train_std
        return mu

    def log_marginal_likelihood(self):
        """Value and gradient-free objective for the fitted hyperparameters.

        Returns the negative log marginal likelihood including the `n/2 log 2pi`
        constant, so it lines up with
        `GaussianProcessRegressor.log_marginal_likelihood_value_`.
        """
        if self.L_ is None:
            raise ValueError("fit must be called before log_marginal_likelihood")
        y_norm = (self.y_train_ - self._y_train_mean) / self._y_train_std
        return _lib.nll(self.L_, y_norm, self.alpha_)

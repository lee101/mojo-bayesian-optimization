"""Correctness-gated benchmark for mojo-bayesian-optimization.

The reference for every case is the strongest available implementation of the
same computation: scikit-learn for the factorisation and the predictive sweep
(it goes through BLAS), SciPy for the normal CDF and PDF, and NumPy for the
argmin. The `bayes_opt` package itself is not the baseline because the work
happens in the regressor it delegates to.

Every case checks agreement with that reference before timing, so a broken
kernel shows up as a correctness failure rather than a suspiciously good number.
"""

from __future__ import annotations

import pathlib
import sys
import time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from scipy.stats import norm  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import Matern  # noqa: E402

import mojo_bayesian_optimization as mbo  # noqa: E402
from mojo_bayesian_optimization import _lib  # noqa: E402


def _time(fn, repeats=9):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _data(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, d))
    y = np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2 - 0.3 * X[:, 2]
    return X, y


def bench_fit(n: int = 200, d: int = 4, repeats: int = 7):
    """Kernel matrix, Cholesky and the two triangular solves, against sklearn."""
    X, y = _data(n, d)
    ours = mbo.GaussianProcess(length_scale=1.0, alpha=1e-6)
    theirs = GaussianProcessRegressor(
        kernel=Matern(length_scale=1.0, nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
    )
    ours.fit(X, y)
    theirs.fit(X, y)
    np.testing.assert_allclose(
        ours.predict(X, return_std=True)[0],
        theirs.predict(X, return_std=True)[0],
        rtol=1e-8,
        atol=1e-9,
    )
    return (
        f"fit n={n} d={d}",
        _time(lambda: theirs.fit(X, y), repeats),
        _time(lambda: mbo.GaussianProcess(length_scale=1.0, alpha=1e-6).fit(X, y), repeats),
    )


def bench_predict(n: int = 60, m: int = 20000, d: int = 3, repeats: int = 7):
    """Predictive mean and standard deviation over a large candidate batch."""
    X, y = _data(n, d)
    Xs, _ = _data(m, d, seed=1)
    ours = mbo.GaussianProcess(length_scale=1.0, alpha=1e-6).fit(X, y)
    theirs = GaussianProcessRegressor(
        kernel=Matern(length_scale=1.0, nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
    ).fit(X, y)
    mu, sd = ours.predict(Xs, return_std=True)
    mu_t, sd_t = theirs.predict(Xs, return_std=True)
    np.testing.assert_allclose(mu, mu_t, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(sd, sd_t, rtol=1e-7, atol=1e-9)

    par_mu, par_sd = ours.predict(Xs, return_std=True, n_workers=8)
    np.testing.assert_array_equal(mu, par_mu)

    ref = _time(lambda: theirs.predict(Xs, return_std=True), repeats)
    ours_t = _time(lambda: ours.predict(Xs, return_std=True), repeats)
    par_t = _time(lambda: ours.predict(Xs, return_std=True, n_workers=8), repeats)
    return ref, ours_t, par_t, f"predict {n}->{m}"


def bench_acquisition(m: int = 20000, repeats: int = 5):
    """Expected improvement over a candidate batch, against SciPy's normal laws."""
    rng = np.random.default_rng(2)
    mean = rng.normal(size=m)
    std = np.abs(rng.normal(size=m)) + 0.05
    y_max, xi = -0.5, 0.01
    got = _lib.acq_ei(mean, std, y_max, xi)
    a = mean - y_max - xi
    z = a / std
    expect = a * norm.cdf(z) + std * norm.pdf(z)
    np.testing.assert_allclose(got, expect, rtol=1e-10, atol=1e-13)

    def theirs():
        return a * norm.cdf(z) + std * norm.pdf(z)

    return (
        f"EI over {m}",
        _time(theirs, repeats),
        _time(lambda: _lib.acq_ei(mean, std, y_max, xi), repeats),
    )


def bench_one_iteration(n: int = 60, m: int = 20000, d: int = 3, repeats: int = 7):
    """A whole acquisition step: predict, score, take the argmin."""
    X, y = _data(n, d)
    bounds = np.array([[-2.0, 2.0]] * d)
    candidates = np.random.RandomState(4).uniform(
        bounds[:, 0], bounds[:, 1], size=(m, d)
    )
    y_max = float(y.min())
    ours = mbo.GaussianProcess(length_scale=1.0, alpha=1e-6).fit(X, y)
    theirs = GaussianProcessRegressor(
        kernel=Matern(length_scale=1.0, nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
    ).fit(X, y)

    def ref():
        mu, sd = theirs.predict(candidates, return_std=True)
        a = mu - y_max - 0.01
        z = a / sd
        values = a * norm.cdf(z) + sd * norm.pdf(z)
        return candidates[int(np.argmin(values))]

    def mine(workers=1):
        mu, sd = ours.predict(candidates, return_std=True, n_workers=workers)
        values = _lib.acq_ei(mu, sd, y_max, 0.01, n_workers=workers)
        best, _ = _lib.argmin(values, n_workers=workers)
        return candidates[best]

    assert np.allclose(ref(), mine())
    assert np.array_equal(mine(), mine())
    return (
        f"one EI step {n}->{m}",
        _time(ref, repeats),
        _time(lambda: mine(1), repeats),
    )


def bench_threaded_predict(n: int = 60, m: int = 200000, d: int = 3, repeats: int = 5):
    """Whether splitting the candidate sweep across cores pays at all."""
    X, y = _data(n, d)
    Xs, _ = _data(m, d, seed=1)
    gp = mbo.GaussianProcess(length_scale=1.0, alpha=1e-6, n_workers=8).fit(X, y)
    a = gp.predict(Xs, n_workers=1)
    b = gp.predict(Xs, n_workers=8)
    np.testing.assert_array_equal(a, b)
    return (
        f"predict {n}->{m} serial/8",
        _time(lambda: gp.predict(Xs, n_workers=1), repeats),
        _time(lambda: gp.predict(Xs, n_workers=8), repeats),
    )


def main():
    print(f"{'case':<24}{'reference':>12}{'mojo 1 core':>14}{'mojo 8 core':>14}{'x':>8}")
    print("-" * 72)
    ref, ours_t, par_t, name = bench_predict()
    print(
        f"{name:<24}{ref*1e3:>10.2f}ms{ours_t*1e3:>12.2f}ms{par_t*1e3:>12.2f}ms"
        f"{ref/min(ours_t, par_t):>7.2f}x"
    )
    for fn in (bench_fit, bench_acquisition, bench_one_iteration):
        name, ref, ours_t = fn()
        print(
            f"{name:<24}{ref*1e3:>10.2f}ms{ours_t*1e3:>12.2f}ms{'-':>14}"
            f"{ref/ours_t:>7.2f}x"
        )
    name, serial, par = bench_threaded_predict()
    print(
        f"{name:<24}{'-':>12}{serial*1e3:>12.2f}ms{par*1e3:>12.2f}ms"
        f"{serial/par:>7.2f}x"
    )
    print()
    print("ratios above 1.00x favour mojo-bayesian-optimization; below is a loss")


if __name__ == "__main__":
    main()

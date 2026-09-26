"""Parity against `bayes_opt` for the acquisitions and scikit-learn for the GP.

Mojo emits FMA, so the GP results are compared with an explicit tolerance: the
factorisation and the triangular solves reorder the sums, and a 12-point
Cholesky is enough for the last bits to move. The acquisition values are
transcendental but well conditioned in the tested range, so they are held to a
tighter relative tolerance than the GP. Exact equality is asserted only where
the operation genuinely is exact: the Cholesky against NumPy's, the argmin
index, and the parameter bookkeeping.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

import mojo_bayesian_optimization as mbo
from mojo_bayesian_optimization import _lib

# `bayes_opt` 3.x exposes the acquisitions as classes in `bayes_opt.acquisition`.
# The shared test venv has been seen with 1.x as well, which has no such module,
# so the upstream half of each comparison is resolved per test rather than at
# import: a missing upstream class skips that one comparison and leaves the rest
# of the suite, and in particular the scikit-learn GP parity, running.
def _upstream(name, **kwargs):
    try:
        from bayes_opt import acquisition
    except ImportError as exc:  # pragma: no cover - venv dependent
        pytest.skip(f"this venv has a bayes_opt without bayes_opt.acquisition: {exc}")
    return getattr(acquisition, name)(**kwargs)

KERNELS = {
    "matern": (lambda ls: Matern(length_scale=ls, nu=2.5), 3),
    "matern32": (lambda ls: Matern(length_scale=ls, nu=1.5), 2),
    "matern12": (lambda ls: Matern(length_scale=ls, nu=0.5), 1),
    "rbf": (lambda ls: RBF(length_scale=ls), 0),
}


def _sample(n=40, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=n) * scale
    std = np.abs(rng.normal(size=n)) * scale + 1e-3
    return mean, std


# -- acquisitions --------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.5, 2.576, 10.0])
def test_ucb_matches_bayes_opt(kappa):
    mean, std = _sample()
    ours = mbo.UpperConfidenceBound(kappa=kappa).base_acq(mean, std)
    theirs = _upstream("UpperConfidenceBound", kappa=kappa).base_acq(mean, std)
    np.testing.assert_allclose(ours, theirs, rtol=1e-13, atol=1e-15)


def test_ucb_is_mean_plus_kappa_std():
    mean, std = _sample()
    np.testing.assert_allclose(
        mbo.UpperConfidenceBound(kappa=2.576).base_acq(mean, std),
        mean + 2.576 * std,
        rtol=1e-14,
    )


def test_poi_matches_bayes_opt():
    mean, std = _sample(seed=1, scale=3.0)
    ours = mbo.ProbabilityOfImprovement(xi=0.01)
    theirs = _upstream("ProbabilityOfImprovement", xi=0.01)
    ours.y_max = theirs.y_max = -0.5
    np.testing.assert_allclose(
        ours.base_acq(mean, std), theirs.base_acq(mean, std), rtol=1e-10, atol=1e-14
    )


def test_ei_matches_bayes_opt():
    mean, std = _sample(seed=2, scale=3.0)
    ours = mbo.ExpectedImprovement(xi=0.01)
    theirs = _upstream("ExpectedImprovement", xi=0.01)
    ours.y_max = theirs.y_max = -0.5
    np.testing.assert_allclose(
        ours.base_acq(mean, std), theirs.base_acq(mean, std), rtol=1e-10, atol=1e-13
    )


@pytest.mark.parametrize("y_max", [-5.0, -0.5, 0.0, 2.0])
@pytest.mark.parametrize("xi", [0.0, 0.01, 0.5])
def test_ei_matches_the_closed_form(y_max, xi):
    """EI is a*Phi(a/sigma) + sigma*phi(a/sigma); the PDF term is the whole
    difference from the probability of improvement, so losing it still yields a
    monotone function and still looks plausible."""
    mean, std = _sample(seed=3, scale=2.0)
    a = mean - y_max - xi
    z = a / std
    expect = a * norm.cdf(z) + std * norm.pdf(z)
    acq = mbo.ExpectedImprovement(xi=xi)
    acq.y_max = y_max
    np.testing.assert_allclose(acq.base_acq(mean, std), expect, rtol=1e-10, atol=1e-14)


def test_poi_matches_the_closed_form():
    mean, std = _sample(seed=4, scale=2.0)
    y_max, xi = -0.25, 0.01
    acq = mbo.ProbabilityOfImprovement(xi=xi)
    acq.y_max = y_max
    expect = norm.cdf((mean - y_max - xi) / std)
    np.testing.assert_allclose(acq.base_acq(mean, std), expect, rtol=1e-10, atol=1e-14)


def test_ei_rewards_uncertainty_where_poi_does_not():
    """A wider posterior at the same mean is worth more under EI, not less.

    Both acquisitions are monotone in `sigma` at first glance, but only EI has
    the `sigma * phi` term, which grows without bound. Dropping that term, or
    computing `std` without the target rescaling, collapses the gap and makes
    EI a monotone transform of POI.
    """
    mean = np.array([1.0, 1.0])
    std = np.array([1.0, 8.0])
    poi = mbo.ProbabilityOfImprovement(xi=0.0)
    poi.y_max = 0.0
    ei = mbo.ExpectedImprovement(xi=0.0)
    ei.y_max = 0.0
    poi_vals = poi.base_acq(mean, std)
    ei_vals = ei.base_acq(mean, std)
    assert poi_vals[1] < poi_vals[0]
    assert ei_vals[1] > ei_vals[0]
    assert ei_vals[1] > 3.0 * poi_vals[1]
    # The closed form must agree, so the growth is not an artefact.
    a = mean - 0.0
    expect = a * norm.cdf(a / std) + std * norm.pdf(a / std)
    np.testing.assert_allclose(ei_vals, expect, rtol=1e-12)


def test_ei_at_the_incumbent_is_the_spread_times_the_normal_density():
    """At the incumbent the improvement is zero, so only the PDF term is left.

    An EI that forgets the PDF term returns exactly zero here, which looks like
    a plausible answer and is the whole reason the function exists.
    """
    ei = mbo.ExpectedImprovement(xi=0.0)
    ei.y_max = 0.0
    got = float(ei.base_acq(np.array([0.0]), np.array([0.25]))[0])
    assert got == pytest.approx(0.25 * norm.pdf(0.0), rel=1e-12)
    assert got > 0.09


def test_improvement_acquisitions_require_y_max():
    for cls in (mbo.ExpectedImprovement, mbo.ProbabilityOfImprovement):
        acq = cls(xi=0.01)
        mean, std = _sample(seed=6, n=3)
        with pytest.raises(ValueError):
            acq.base_acq(mean, std)


def test_ucb_and_improvement_reject_bad_parameters():
    with pytest.raises(ValueError):
        mbo.UpperConfidenceBound(kappa=-1.0)
    with pytest.raises(ValueError):
        mbo.UpperConfidenceBound(exploration_decay=0.0)
    with pytest.raises(ValueError):
        mbo.ExpectedImprovement(xi=-0.1)
    with pytest.raises(ValueError):
        mbo.ExpectedImprovement(xi=0.0, exploration_decay=1.5)
    with pytest.raises(ValueError):
        mbo.ExpectedImprovement(xi=0.0, exploration_decay_delay=-1)


def test_acquisition_parameters_round_trip():
    acq = mbo.UpperConfidenceBound(kappa=1.5, exploration_decay=0.8)
    params = acq.get_acquisition_params()
    other = mbo.UpperConfidenceBound()
    other.set_acquisition_params(params)
    assert other.kappa == 1.5
    assert other.exploration_decay == 0.8
    ei = mbo.ExpectedImprovement(xi=0.25)
    restored = mbo.ExpectedImprovement(xi=0.0)
    restored.set_acquisition_params(ei.get_acquisition_params())
    assert restored.xi == 0.25


def test_exploration_decays_only_after_the_delay():
    X = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
    gp = mbo.GaussianProcess().fit(X, np.sin(6 * X[:, 0]))
    bounds = np.array([[-1.0, 2.0]])
    acq = mbo.UpperConfidenceBound(
        kappa=1.0, exploration_decay=0.5, exploration_decay_delay=3
    )
    for _ in range(2):
        acq.suggest(gp, bounds, n_random=64, random_state=0)
    assert acq.i == 2
    assert acq.kappa == pytest.approx(1.0)
    acq.suggest(gp, bounds, n_random=64, random_state=0)
    assert acq.kappa == pytest.approx(0.5)
    acq.suggest(gp, bounds, n_random=64, random_state=0)
    assert acq.kappa == pytest.approx(0.25)


def test_base_acquisition_rejects_the_missing_hook():
    acq = mbo.AcquisitionFunction()
    with pytest.raises(NotImplementedError):
        acq.base_acq(np.zeros(1), np.ones(1))
    with pytest.raises(NotImplementedError):
        acq.get_acquisition_params()
    with pytest.raises(NotImplementedError):
        acq.set_acquisition_params({})


# -- Gaussian process ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(KERNELS))
@pytest.mark.parametrize("normalize_y", [True, False])
@pytest.mark.parametrize("alpha", [1e-6, 1e-2])
def test_gp_predict_matches_sklearn(name, normalize_y, alpha):
    make_kernel, _ = KERNELS[name]
    ls = [1.0, 0.7, 1.3] if name != "rbf" else 0.9
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(12, 3))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 - 0.3 * X[:, 2]
    ours = mbo.GaussianProcess(
        kernel=name, length_scale=ls, alpha=alpha, normalize_y=normalize_y
    ).fit(X, y)
    theirs = GaussianProcessRegressor(
        kernel=make_kernel(ls), alpha=alpha, normalize_y=normalize_y, optimizer=None
    ).fit(X, y)
    Xs = rng.uniform(-2.0, 2.0, size=(64, 3))
    mu, sd = ours.predict(Xs, return_std=True)
    mu_t, sd_t = theirs.predict(Xs, return_std=True)
    np.testing.assert_allclose(mu, mu_t, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(sd, sd_t, rtol=1e-8, atol=1e-11)


def test_gp_predict_mean_only_returns_one_array():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(8, 2))
    gp = mbo.GaussianProcess().fit(X, X[:, 0] ** 2)
    out = gp.predict(rng.uniform(size=(5, 2)))
    assert out.shape == (5,)


def test_gp_interpolates_its_training_points_at_low_noise():
    """With a small noise level the posterior mean must pass through the data.

    The classic failure here is a solve that is self-consistent but solves the
    wrong system; it would still produce a smooth surface, just not one through
    the observations.
    """
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, size=(10, 2))
    y = X[:, 0] ** 2 + 3.0 * X[:, 1]
    gp = mbo.GaussianProcess(length_scale=0.5, alpha=1e-10).fit(X, y)
    mu = gp.predict(X)
    np.testing.assert_allclose(mu, y, rtol=1e-5, atol=1e-6)


def test_predictive_variance_matches_the_closed_form_for_one_observation():
    """With a single training point the posterior variance is analytic:

        var(x) = k(x, x) - k(x, x0)**2 / (k(x0, x0) + alpha)

    so the whole factorisation path can be checked without any reference
    implementation in the loop.
    """
    x0 = np.array([[0.5, -0.25]])
    y0 = np.array([1.75])
    alpha = 1e-3
    length_scale = [0.8, 1.3]
    gp = mbo.GaussianProcess(
        length_scale=length_scale, alpha=alpha, normalize_y=False
    ).fit(x0, y0)
    Xs = np.array([[0.0, 0.0], [1.5, 1.0], [-2.0, 0.75], [0.5, -0.25]])
    _, sd = gp.predict(Xs, return_std=True)
    # A stationary kernel has k(x, x) = 1, so the posterior variance collapses
    # to 1 - k(x, x0)**2 / (k(x0, x0) + alpha), with the Matern 5/2 shape
    # spelled out here rather than borrowed from a reference implementation.
    r2 = np.sum(((Xs - x0) / np.asarray(length_scale)) ** 2, axis=1)
    t = np.sqrt(5.0 * r2)
    kx0 = (1.0 + t + 5.0 * r2 / 3.0) * np.exp(-t)
    k00 = 1.0 + alpha
    expect = np.sqrt(np.maximum(0.0, 1.0 - kx0**2 / k00))
    np.testing.assert_allclose(sd, expect, rtol=1e-10, atol=1e-12)
    assert sd[0] > 0.0
    # At the observation itself the posterior collapses to the noise level.
    np.testing.assert_allclose(sd[3], np.sqrt(alpha / k00), rtol=1e-10)


def test_target_standardisation_scales_the_uncertainty():
    """The variance comes out of the factorisation in normalised units.

    Rescaling the mean but not the standard deviation is invisible whenever the
    targets happen to have unit spread, and quietly wrong everywhere else, so it
    is pinned by the ratio rather than by an absolute value.
    """
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(10, 2))
    y = 4.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5
    Xs = rng.uniform(size=(20, 2))
    raw = mbo.GaussianProcess(normalize_y=False).fit(X, y)
    normed = mbo.GaussianProcess(normalize_y=True).fit(X, y)
    _, sd_raw = raw.predict(Xs, return_std=True)
    _, sd_norm = normed.predict(Xs, return_std=True)
    np.testing.assert_allclose(sd_norm / sd_raw, y.std(), rtol=1e-9)


def test_standardisation_is_invariant_to_a_target_shift():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(9, 2))
    y = X[:, 0] ** 2 - X[:, 1]
    shifted = mbo.GaussianProcess(normalize_y=True).fit(X, y + 100.0)
    plain = mbo.GaussianProcess(normalize_y=True).fit(X, y)
    Xs = rng.uniform(size=(15, 2))
    mu_s, _ = shifted.predict(Xs, return_std=True)
    mu_p, _ = plain.predict(Xs, return_std=True)
    np.testing.assert_allclose(mu_s, mu_p + 100.0, rtol=1e-9, atol=1e-9)


def test_cholesky_matches_numpy():
    """A blocked LAPACK `potrf` and a scalar pivot loop sum in different orders,
    so this is a tight tolerance rather than bit equality. The strict upper
    triangle check is what actually pins the in-place hazard: a kernel that
    blanks it still returns a lower-triangular array full of plausible numbers.
    """
    rng = np.random.default_rng(5)
    A = rng.normal(size=(9, 9))
    K = A @ A.T + 9.0 * np.eye(9)
    L = _lib.cholesky(K)
    np.testing.assert_allclose(L, np.linalg.cholesky(K), rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(np.triu(L, 1), np.zeros((9, 9)))


def test_cholesky_reconstructs_the_matrix():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(14, 14))
    K = A @ A.T + 14.0 * np.eye(14)
    L = _lib.cholesky(K)
    np.testing.assert_allclose(L @ L.T, K, rtol=1e-12, atol=1e-12)


def test_cholesky_rejects_a_non_positive_definite_matrix():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        _lib.cholesky(bad)


def test_ill_conditioned_kernel_matrix_raises():
    """Duplicate points plus no noise give a singular matrix, which is what a
    degenerate design looks like. Failing loudly beats returning a factor that
    silently predicts a constant."""
    X = np.zeros((4, 1))
    y = np.zeros(4)
    with pytest.raises(np.linalg.LinAlgError):
        mbo.GaussianProcess(alpha=0.0).fit(X, y)


def test_log_marginal_likelihood_is_the_negative_of_sklearn():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(11, 2))
    y = np.cos(2.0 * X[:, 0]) + X[:, 1]
    ours = mbo.GaussianProcess(length_scale=0.9, alpha=1e-4).fit(X, y)
    theirs = GaussianProcessRegressor(
        kernel=Matern(length_scale=0.9, nu=2.5),
        alpha=1e-4,
        normalize_y=True,
        optimizer=None,
    ).fit(X, y)
    assert ours.log_marginal_likelihood() == pytest.approx(
        -theirs.log_marginal_likelihood_value_, rel=1e-9
    )


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_kernel_matrix_matches_sklearn(name):
    make_kernel, _ = KERNELS[name]
    ls = [1.0, 0.6] if name != "rbf" else 0.75
    rng = np.random.default_rng(8)
    X = rng.uniform(-3.0, 3.0, size=(7, 2))
    np.testing.assert_allclose(
        _lib.kernel_matrix(X, ls, _lib.KERNELS[name]),
        make_kernel(ls)(X),
        rtol=1e-10,
        atol=1e-13,
    )


def test_cross_kernel_is_not_the_square_matrix():
    """The candidate sweep needs `kernel(X_train, X_candidates)`, shape
    (n, m). Passing the training set twice gives an (n, n) block that is the
    right shape only when n happens to equal m, and silently wrong otherwise."""
    rng = np.random.default_rng(9)
    A = rng.uniform(-1.0, 1.0, size=(5, 2))
    B = rng.uniform(-1.0, 1.0, size=(11, 2))
    block = _lib.cross_kernel(A, B, 0.8, _lib.KERNELS["matern"])
    assert block.shape == (5, 11)
    np.testing.assert_allclose(
        block, Matern(length_scale=0.8, nu=2.5)(A, B), rtol=1e-10, atol=1e-13
    )
    assert not np.array_equal(block, _lib.kernel_matrix(A, 0.8, _lib.KERNELS["matern"]))


# -- plumbing ------------------------------------------------------------


def test_serial_and_threaded_sweeps_agree():
    """The candidate sweep is split across a thread pool when asked. Each chunk
    owns its own scratch, so a shared buffer would corrupt the tail of the
    range; comparing the two paths catches that."""
    rng = np.random.default_rng(10)
    X = rng.uniform(-2.0, 2.0, size=(30, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    Xs = rng.uniform(-2.0, 2.0, size=(2000, 3))
    acq = mbo.ExpectedImprovement(xi=0.01, n_workers=1)
    acq.y_max = float(y.min())
    gp = mbo.GaussianProcess().fit(X, y)
    serial_mu, serial_sd = gp.predict(Xs, return_std=True, n_workers=1)
    par_mu, par_sd = gp.predict(Xs, return_std=True, n_workers=4)
    np.testing.assert_array_equal(serial_mu, par_mu)
    np.testing.assert_array_equal(serial_sd, par_sd)
    np.testing.assert_array_equal(
        acq.base_acq(serial_mu, serial_sd), _lib.acq_ei(serial_mu, serial_sd, acq.y_max, 0.01, 4)
    )


def test_threaded_sweep_agrees_on_a_ragged_chunk_count():
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(11, 2))
    y = X.sum(axis=1)
    Xs = rng.uniform(size=(1001, 2))
    gp = mbo.GaussianProcess().fit(X, y)
    a = gp.predict(Xs, n_workers=1)
    b = gp.predict(Xs, n_workers=3)
    np.testing.assert_array_equal(a, b)


def test_argmin_returns_the_earliest_minimum():
    values = np.array([1.0, 0.5, 0.5, 0.2, 0.9])
    assert _lib.argmin(values) == (3, pytest.approx(0.2))
    tied = np.array([0.5, 0.5, 0.5])
    assert _lib.argmin(tied)[0] == 0
    assert _lib.argmin(np.array([]))[0] == -1


def test_length_scale_must_be_positive_and_match_the_dimension():
    with pytest.raises(ValueError):
        mbo.GaussianProcess(length_scale=[1.0, -1.0])
    with pytest.raises(ValueError):
        mbo.GaussianProcess(length_scale=0.0)
    with pytest.raises(ValueError):
        mbo.GaussianProcess(kernel="not-a-kernel")
    with pytest.raises(ValueError):
        mbo.GaussianProcess(alpha=-1.0)
    # The width is only known once the data arrives, so a mismatched ARD
    # length is a fit-time error. Passing it through silently would broadcast
    # the wrong dimension into the squared distance.
    gp = mbo.GaussianProcess(length_scale=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gp.fit(np.zeros((4, 2)), np.zeros(4))


def test_fit_and_predict_validate_their_inputs():
    with pytest.raises(ValueError):
        mbo.GaussianProcess().predict(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        mbo.GaussianProcess().fit(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError):
        mbo.GaussianProcess().fit(np.zeros((3, 2)), np.zeros(2))
    gp = mbo.GaussianProcess().fit(np.zeros((3, 2)), np.zeros(3))
    with pytest.raises(ValueError):
        gp.predict(np.zeros((4, 3)))


# -- the loop ------------------------------------------------------------


def test_suggest_stays_inside_the_bounds():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.0, 1.0, size=(10, 2))
    y = X[:, 0] ** 2 + X[:, 1]
    gp = mbo.GaussianProcess().fit(X, y)
    bounds = np.array([[-3.0, 3.0], [0.5, 1.5]])
    for acq in (
        mbo.ExpectedImprovement(xi=0.01),
        mbo.ProbabilityOfImprovement(xi=0.01),
        mbo.UpperConfidenceBound(kappa=2.0),
    ):
        point = acq.suggest(gp, bounds, n_random=500, random_state=1)
        assert point.shape == (2,)
        assert np.all(point >= bounds[:, 0]) and np.all(point <= bounds[:, 1])


def test_suggest_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(13)
    X = rng.uniform(-1.0, 1.0, size=(8, 1))
    y = np.cos(4 * X[:, 0])
    gp = mbo.GaussianProcess().fit(X, y)
    bounds = np.array([[-2.0, 2.0]])
    a = mbo.ExpectedImprovement(xi=0.01).suggest(gp, bounds, n_random=300, random_state=5)
    b = mbo.ExpectedImprovement(xi=0.01).suggest(gp, bounds, n_random=300, random_state=5)
    np.testing.assert_array_equal(a, b)


def test_suggest_requires_a_fitted_gp():
    with pytest.raises(ValueError):
        mbo.UpperConfidenceBound().suggest(
            mbo.GaussianProcess(), np.array([[-1.0, 1.0]])
        )


def test_suggest_rejects_malformed_bounds():
    gp = mbo.GaussianProcess().fit(np.zeros((3, 1)), np.zeros(3))
    with pytest.raises(ValueError):
        mbo.UpperConfidenceBound().suggest(gp, np.array([-1.0, 1.0]))


def _random_search(f, bounds, n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))
    return min(f(p) for p in pts)


@pytest.mark.parametrize(
    "func,bounds,tolerance",
    [
        (lambda p: float(np.sum((p - 0.3) ** 2)), np.array([[-2.0, 2.0], [-2.0, 2.0]]), 1e-2),
        # A badly conditioned objective: a single isotropic length scale cannot
        # resolve both the wide and the narrow direction at once, so the bar here
        # is beating random search rather than hitting the optimum.
        (
            lambda p: float((p[0] - 1.0) ** 2 + 100.0 * (p[1] - 0.5) ** 2),
            np.array([[-2.0, 2.0], [-2.0, 2.0]]),
            0.1,
        ),
        (
            lambda p: float(np.sum(np.sin(3.0 * p) + p**2)),
            np.array([[-3.0, 3.0]] * 4),
            None,
        ),
    ],
)
def test_maximize_beats_random_search(func, bounds, tolerance):
    """End-to-end check against the analytic optimum and a random baseline.

    `bayes_opt.BayesianOptimization.maximize` is not used as the reference here:
    its driver wraps a queue, a logger and the domain machinery, and the
    numeric contract worth testing is the acquisition and the surrogate, both
    of which are compared against the real package above.
    """
    result = mbo.maximize(func, bounds, n_iterations=25, random_state=17)
    assert result.n_evaluations == 30
    assert result.X.shape == (30, bounds.shape[0])
    baseline = _random_search(func, bounds, 30, 17)
    assert result.y <= baseline
    assert result.y == pytest.approx(func(result.x))
    assert np.allclose(result.x, result.X[int(np.argmin(result.Y))])
    if tolerance is not None:
        assert result.y < tolerance


def test_maximize_validates_its_arguments():
    f = lambda p: 0.0
    with pytest.raises(ValueError):
        mbo.maximize(f, np.array([-1.0, 1.0]), n_iterations=1)
    with pytest.raises(ValueError):
        mbo.maximize(f, np.array([[-1.0, 1.0]]), n_iterations=-1)
    with pytest.raises(ValueError):
        mbo.maximize(f, np.array([[-1.0, 1.0]]), initial_points=0)


def test_maximize_with_a_supplied_acquisition_and_gp():
    result = mbo.maximize(
        lambda p: float(np.sum((p - 0.1) ** 2)),
        np.array([[-1.0, 1.0]]),
        n_iterations=10,
        acq=mbo.UpperConfidenceBound(kappa=2.0),
        gp=mbo.GaussianProcess(kernel="rbf", length_scale=0.5),
        random_state=2,
    )
    assert result.n_evaluations == 15
    assert result.y >= 0.0
    assert math.isfinite(result.y)

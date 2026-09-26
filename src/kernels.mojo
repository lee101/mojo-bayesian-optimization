"""Gaussian-process linear algebra and acquisition functions in compiled Mojo.

The numeric core of Bayesian optimisation is a small dense factorisation and
the per-candidate evaluation that follows it. Once `K = LL^T` exists, the
predictive mean and variance at a new point `x` collapse to two triangular
solves and two dot products:

    l      = L^-1 k(x)          O(n) forward substitution
    mu(x)  = l . beta           beta = L^-T y, computed once
    var(x) = k(x,x) - l . l

so a sweep over `m` candidates costs `O(m n)` rather than the `O(m n^2)` a
naive `K^-1` product would. The acquisition functions are then elementwise over
`m` values, which is where the `erfc` and `exp` calls live.

Every loop here is serial on purpose. These are small, latency-bound kernels
and 1.2.0 cannot carry a pointer into a parallel body; the Python layer instead
splits the candidate range across a thread pool and calls the range-taking
entry points once per chunk.

Buffers cross the C ABI as 64-bit addresses and are rebuilt inside each body,
because `@export` rejects a function whose parameter types are inferred.
Nothing here allocates, so no exported symbol is `raises`.
"""

from std.math import erfc, exp, log, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]

comptime INV_SQRT2 = 0.70710678118654752440
comptime INV_SQRT2PI = 0.39894228040143267794
comptime LOG2PI = 1.83787706640934548356


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


@export("bo_cholesky")
def bo_cholesky(k_addr: Int, n: Int, l_addr: Int) abi("C") -> Int:
    """In-place lower Cholesky of a symmetric positive definite `n` by `n` matrix.

    `k` and `l` are both row-major with stride `n`. Returns 0 on success, or
    `-(i + 2)` when the pivot at row `i` is not positive, which is what a
    non-positive-definite kernel matrix looks like in practice.
    """
    if n <= 0:
        return -1
    var k = fp(k_addr)
    var l = fp(l_addr)
    # The source matrix is read, never overwritten, and only the lower triangle
    # of `l` is written. An in-place factorisation that blanks the strict upper
    # triangle destroys the entries later rows still need, which shows up as a
    # plausible-looking but wrong factor.
    for i in range(n):
        for j in range(i):
            var s = k[unsafe_offset=i * n + j]
            for t in range(j):
                s = s - l[unsafe_offset=i * n + t] * l[unsafe_offset=j * n + t]
            l[unsafe_offset=i * n + j] = s / l[unsafe_offset=j * n + j]
        var d = k[unsafe_offset=i * n + i]
        for t in range(i):
            d = d - l[unsafe_offset=i * n + t] * l[unsafe_offset=i * n + t]
        if d <= 0.0:
            return -(i + 2)
        l[unsafe_offset=i * n + i] = sqrt(d)
    return 0


@export("bo_cholesky_solve")
def bo_cholesky_solve(l_addr: Int, y_addr: Int, n: Int, alpha_addr: Int,
                      beta_addr: Int) abi("C"):
    """Write alpha = K^-1 y and beta = L^-1 y, both from one factorisation.

    `beta` is the vector the predictive mean contracts against. With
    `k = L z` and `beta = L^-1 y`,

        mu = k^T K^-1 y = (L z)^T L^-T L^-1 y = z^T L^-1 y = z . beta

    so one forward substitution per candidate is enough. Contracting against
    `L^-T y` instead is a one-character slip that still returns plausible
    numbers, which is why the parity test pins the predictive mean against
    scikit-learn rather than against a self-consistent round trip.
    """
    var l = fp(l_addr)
    var y = fp(y_addr)
    var a = fp(alpha_addr)
    var b = fp(beta_addr)
    if n <= 0:
        return
    # Forward substitution: b = L^-1 y.
    for i in range(n):
        var s = y[unsafe_offset=i]
        for t in range(i):
            s = s - l[unsafe_offset=i * n + t] * b[unsafe_offset=t]
        b[unsafe_offset=i] = s / l[unsafe_offset=i * n + i]
    # Back substitution: a = L^-T beta = K^-1 y.
    for i in range(n - 1, -1, -1):
        var s = b[unsafe_offset=i]
        for t in range(i + 1, n):
            s = s - l[unsafe_offset=t * n + i] * a[unsafe_offset=t]
        a[unsafe_offset=i] = s / l[unsafe_offset=i * n + i]


@export("bo_gp_predict")
def bo_gp_predict(kt_addr: Int, n: Int, m: Int, l_addr: Int, beta_addr: Int,
                  kss_addr: Int, mu_addr: Int, var_addr: Int, lo: Int, hi: Int,
                  scratch_addr: Int) abi("C"):
    """Predictive mean and variance for candidates `lo` to `hi` of `m`.

    `kt` is the `n` by `m` cross-covariance matrix, row-major, so the covariance
    with candidate `i` is the strided column `kt[j * m + i]`. `scratch` holds at
    least `n` doubles and belongs to the caller, which lets several chunks run
    concurrently without sharing state. A negative variance is clamped to zero:
    rounding can push a noiseless diagonal slightly below the true value.
    """
    if hi <= lo:
        return
    var kt = fp(kt_addr)
    var l = fp(l_addr)
    var beta = fp(beta_addr)
    var kss = fp(kss_addr)
    var mu = fp(mu_addr)
    var vr = fp(var_addr)
    var z = fp(scratch_addr)
    for i in range(lo, hi):
        for r in range(n):
            var s = kt[unsafe_offset=r * m + i]
            for t in range(r):
                s = s - l[unsafe_offset=r * n + t] * z[unsafe_offset=t]
            z[unsafe_offset=r] = s / l[unsafe_offset=r * n + r]
        var sm = 0.0
        var sq = 0.0
        for t in range(n):
            sm = sm + z[unsafe_offset=t] * beta[unsafe_offset=t]
            sq = sq + z[unsafe_offset=t] * z[unsafe_offset=t]
        mu[unsafe_offset=i] = sm
        var v = kss[unsafe_offset=i] - sq
        if v < 0.0:
            v = 0.0
        vr[unsafe_offset=i] = v


@export("bo_acq_ucb")
def bo_acq_ucb(mean_addr: Int, std_addr: Int, kappa: Float64, n: Int,
               out_addr: Int, lo: Int, hi: Int) abi("C"):
    """Upper confidence bound, mean + kappa * std."""
    var mean = fp(mean_addr)
    var sd = fp(std_addr)
    var o = fp(out_addr)
    for i in range(lo, hi):
        o[unsafe_offset=i] = mean[unsafe_offset=i] + kappa * sd[unsafe_offset=i]


@export("bo_acq_poi")
def bo_acq_poi(mean_addr: Int, std_addr: Int, y_max: Float64, xi: Float64,
               n: Int, out_addr: Int, lo: Int, hi: Int) abi("C"):
    """Probability of improvement, Phi((mean - y_max - xi) / std)."""
    var mean = fp(mean_addr)
    var sd = fp(std_addr)
    var o = fp(out_addr)
    for i in range(lo, hi):
        var z = (mean[unsafe_offset=i] - y_max - xi) / sd[unsafe_offset=i]
        o[unsafe_offset=i] = 0.5 * erfc(-z * INV_SQRT2)


@export("bo_acq_ei")
def bo_acq_ei(mean_addr: Int, std_addr: Int, y_max: Float64, xi: Float64,
              n: Int, out_addr: Int, lo: Int, hi: Int) abi("C"):
    """Expected improvement, a * Phi(a / std) + std * phi(a / std).

    `a` is the raw improvement `mean - y_max - xi`. The PDF term is what makes
    this different from the probability of improvement: a wide posterior far
    from the incumbent can be worth more even at a low probability.
    """
    var mean = fp(mean_addr)
    var sd = fp(std_addr)
    var o = fp(out_addr)
    for i in range(lo, hi):
        var a = mean[unsafe_offset=i] - y_max - xi
        var z = a / sd[unsafe_offset=i]
        var cdf = 0.5 * erfc(-z * INV_SQRT2)
        var pdf = exp(-0.5 * z * z) * INV_SQRT2PI
        o[unsafe_offset=i] = a * cdf + sd[unsafe_offset=i] * pdf


@export("bo_argmin_range")
def bo_argmin_range(v_addr: Int, lo: Int, hi: Int, idx_addr: Int,
                    val_addr: Int) abi("C") -> Int:
    """Index of the smallest value in `[lo, hi)`, or -1 if the range is empty.

    Ties keep the earliest index, so the choice is deterministic and a repeated
    run suggests the same point.
    """
    if hi <= lo:
        return -1
    var v = fp(v_addr)
    var best = lo
    var bv = v[unsafe_offset=lo]
    for i in range(lo + 1, hi):
        if v[unsafe_offset=i] < bv:
            bv = v[unsafe_offset=i]
            best = i
    fp(idx_addr)[unsafe_offset=0] = Float64(best)
    fp(val_addr)[unsafe_offset=0] = bv
    return best


@export("bo_kernel_ard")
def bo_kernel_ard(a_addr: Int, na: Int, b_addr: Int, nb: Int, d: Int,
                  ls_addr: Int, kind: Int, sf2: Float64, out_addr: Int,
                  out_stride: Int) abi("C"):
    """Cross-covariance between two point sets, shape (na, nb).

    `kind` selects RBF (0), Matern 1/2 (1), Matern 3/2 (2) or Matern 5/2 (3);
    5/2 is the `bayes_opt` default. `out_stride` allows writing into a wider row
    layout, which is what the training-set-by-candidates block needs.
    """
    if d <= 0:
        return
    var a = fp(a_addr)
    var b = fp(b_addr)
    var ls = fp(ls_addr)
    var o = fp(out_addr)
    for i in range(na):
        for j in range(nb):
            var r2 = 0.0
            for t in range(d):
                var s = ls[unsafe_offset=t]
                var diff = a[unsafe_offset=i * d + t] - b[unsafe_offset=j * d + t]
                r2 = r2 + diff * diff / (s * s)
            var at = i * out_stride + j
            if kind == 0:
                o[unsafe_offset=at] = sf2 * exp(-0.5 * r2)
            elif kind == 1:
                o[unsafe_offset=at] = sf2 * exp(-sqrt(r2))
            elif kind == 2:
                var r = sqrt(3.0 * r2)
                o[unsafe_offset=at] = sf2 * (1.0 + r) * exp(-r)
            else:
                var r = sqrt(5.0 * r2)
                o[unsafe_offset=at] = sf2 * (1.0 + r + 5.0 * r2 / 3.0) * exp(-r)


@export("bo_nll")
def bo_nll(l_addr: Int, y_addr: Int, alpha_addr: Int, n: Int) abi("C") -> Float64:
    """Negative log marginal likelihood, -0.5 y^T K^-1 y - sum log L_ii + n/2 log 2pi.

    The constant is included so the value can be compared with
    `GaussianProcessRegressor.log_marginal_likelihood_value_` directly.
    """
    var l = fp(l_addr)
    var y = fp(y_addr)
    var a = fp(alpha_addr)
    var quad = 0.0
    var logdet = 0.0
    for i in range(n):
        quad = quad + y[unsafe_offset=i] * a[unsafe_offset=i]
        logdet = logdet + log(l[unsafe_offset=i * n + i])
    return 0.5 * quad + logdet + 0.5 * Float64(n) * LOG2PI

"""ctypes bridge to the compiled Gaussian-process kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults. Scratch buffers are allocated here, on the Python
side, which keeps the exported Mojo symbols free of allocation and of `raises`.

The candidate sweep is the one place where threading pays: it is an
`O(m * n)` loop of independent work with a serial dependency only inside each
`n`-long triangular solve. ctypes releases the GIL for the foreign call, so the
shim splits the candidate range and calls the range-taking entry point once per
chunk. `max_workers` defaults to 1 because at the sizes a real BO run uses the
thread hand-off costs more than the loop saves.
"""

from __future__ import annotations

import ctypes
import math
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-bayesian-optimization.so"

# RBF, Matern 1/2, Matern 3/2, Matern 5/2. 5/2 is the bayes_opt default.
KERNELS = {"rbf": 0, "matern": 3, "matern12": 1, "matern32": 2}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    i, f, d = ctypes.c_int64, ctypes.c_double, ctypes.c_double
    lib.bo_cholesky.restype = i
    lib.bo_cholesky.argtypes = [i, i, i]
    lib.bo_cholesky_solve.restype = None
    lib.bo_cholesky_solve.argtypes = [i, i, i, i, i]
    lib.bo_gp_predict.restype = None
    lib.bo_gp_predict.argtypes = [i, i, i, i, i, i, i, i, i, i, i]
    lib.bo_acq_ucb.restype = None
    lib.bo_acq_ucb.argtypes = [i, i, d, i, i, i, i]
    lib.bo_acq_poi.restype = None
    lib.bo_acq_poi.argtypes = [i, i, d, d, i, i, i, i]
    lib.bo_acq_ei.restype = None
    lib.bo_acq_ei.argtypes = [i, i, d, d, i, i, i, i]
    lib.bo_argmin_range.restype = i
    lib.bo_argmin_range.argtypes = [i, i, i, i, i]
    lib.bo_kernel_ard.restype = None
    lib.bo_kernel_ard.argtypes = [i, i, i, i, i, i, i, d, i, i]
    lib.bo_nll.restype = d
    lib.bo_nll.argtypes = [i, i, i, i]
    return lib


lib = _load()


def workers(requested: int | None = None) -> int:
    if requested is not None:
        return max(1, int(requested))
    return max(1, min(8, (os.cpu_count() or 1)))


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _chunks(count: int, parts: int):
    parts = max(1, min(parts, count)) if count else 1
    step = -(-count // parts) if count else 0
    bounds = []
    for k in range(parts):
        lo = k * step
        if lo >= count and count:
            break
        hi = min(lo + step, count)
        bounds.append((lo, hi))
    return bounds or [(0, 0)]


def cholesky(K: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a symmetric positive definite matrix."""
    K = _f64(K)
    n = K.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    rc = lib.bo_cholesky(K.ctypes.data, n, L.ctypes.data)
    if rc != 0:
        raise np.linalg.LinAlgError(
            "kernel matrix is not positive definite at row "
            f"{-rc - 2 if rc < -1 else 0}"
        )
    return L


def cholesky_solve(L: np.ndarray, y: np.ndarray):
    """alpha = K^-1 y and beta = L^-T y from a Cholesky factor."""
    L = _f64(L)
    y = _f64(y)
    n = L.shape[0]
    alpha = np.zeros(n, dtype=np.float64)
    beta = np.zeros(n, dtype=np.float64)
    lib.bo_cholesky_solve(
        L.ctypes.data, y.ctypes.data, n, alpha.ctypes.data, beta.ctypes.data
    )
    return alpha, beta


def kernel_matrix(X: np.ndarray, length_scale, kind: int) -> np.ndarray:
    """Stationary kernel matrix with per-dimension lengthscales (ARD)."""
    X = _f64(X)
    n, d = X.shape
    ls = _length_scales(length_scale, d)
    K = np.zeros((n, n), dtype=np.float64)
    lib.bo_kernel_ard(
        X.ctypes.data, n, X.ctypes.data, n, d, ls.ctypes.data, int(kind), 1.0,
        K.ctypes.data, n,
    )
    return K


def cross_kernel(A: np.ndarray, B: np.ndarray, length_scale, kind: int) -> np.ndarray:
    """Kernel matrix between two different point sets, shape (len(A), len(B))."""
    A = _f64(A)
    B = _f64(B)
    na, d = A.shape
    nb = B.shape[0]
    if B.shape[1] != d:
        raise ValueError("A and B must have the same number of columns")
    ls = _length_scales(length_scale, d)
    K = np.zeros((na, nb), dtype=np.float64)
    lib.bo_kernel_ard(
        A.ctypes.data, na, B.ctypes.data, nb, d, ls.ctypes.data, int(kind), 1.0,
        K.ctypes.data, nb,
    )
    return K


def nll(L: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
    L = _f64(L)
    y = _f64(y)
    alpha = _f64(alpha)
    return float(lib.bo_nll(L.ctypes.data, y.ctypes.data, alpha.ctypes.data, L.shape[0]))


def _length_scales(length_scale, d: int) -> np.ndarray:
    ls = np.asarray(length_scale, dtype=np.float64)
    if ls.ndim == 0:
        ls = np.full(d, float(ls))
    if ls.size != d:
        raise ValueError("length_scale must be a scalar or have one entry per dimension")
    if np.any(ls <= 0.0):
        raise ValueError("length_scale entries must be positive")
    return np.ascontiguousarray(ls)


def gp_predict(L, beta, Kt, kss, n_workers: int = 1):
    """Predictive mean and standard deviation for every column of `Kt`."""
    L = _f64(L)
    beta = _f64(beta)
    Kt = _f64(Kt)
    kss = _f64(kss)
    n, m = Kt.shape
    mu = np.empty(m, dtype=np.float64)
    var = np.empty(m, dtype=np.float64)
    if m == 0:
        return mu, np.sqrt(var)
    parts = _chunks(m, workers(n_workers))
    scratch = np.zeros(max(1, n * len(parts)), dtype=np.float64)

    def run(slot, bounds):
        lo, hi = bounds
        base = scratch.ctypes.data + slot * n * 8
        lib.bo_gp_predict(
            Kt.ctypes.data, n, m, L.ctypes.data, beta.ctypes.data, kss.ctypes.data,
            mu.ctypes.data, var.ctypes.data, lo, hi, base,
        )

    if len(parts) == 1:
        run(0, parts[0])
    else:
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            list(pool.map(lambda kv: run(*kv), enumerate(parts)))
    return mu, np.sqrt(var)


def _acq_apply(fn, mean, std, *params, n_workers: int = 1):
    m = mean.size
    out = np.empty(m, dtype=np.float64)
    if m == 0:
        return out
    parts = _chunks(m, workers(n_workers))

    def run(bounds):
        lo, hi = bounds
        fn(mean.ctypes.data, std.ctypes.data, *params, m, out.ctypes.data, lo, hi)

    if len(parts) == 1:
        run(parts[0])
    else:
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            list(pool.map(run, parts))
    return out


def acq_ucb(mean, std, kappa: float, n_workers: int = 1) -> np.ndarray:
    mean = _f64(mean)
    std = _f64(std)
    return _acq_apply(
        lambda ma, sa, k, n, oa, lo, hi: lib.bo_acq_ucb(
            ma, sa, ctypes.c_double(k), n, oa, lo, hi
        ),
        mean, std, kappa, n_workers=n_workers,
    )


def acq_poi(mean, std, y_max: float, xi: float, n_workers: int = 1) -> np.ndarray:
    mean = _f64(mean)
    std = _f64(std)
    return _acq_apply(
        lambda ma, sa, ym, x, n, oa, lo, hi: lib.bo_acq_poi(
            ma, sa, ctypes.c_double(ym), ctypes.c_double(x), n, oa, lo, hi
        ),
        mean, std, y_max, xi, n_workers=n_workers,
    )


def acq_ei(mean, std, y_max: float, xi: float, n_workers: int = 1) -> np.ndarray:
    mean = _f64(mean)
    std = _f64(std)
    return _acq_apply(
        lambda ma, sa, ym, x, n, oa, lo, hi: lib.bo_acq_ei(
            ma, sa, ctypes.c_double(ym), ctypes.c_double(x), n, oa, lo, hi
        ),
        mean, std, y_max, xi, n_workers=n_workers,
    )


def argmin(values: np.ndarray, n_workers: int = 1):
    """Index and value of the smallest entry, earliest index on a tie."""
    v = _f64(values)
    if v.size == 0:
        return -1, float("inf")
    best_idx = -1
    best_val = math.inf
    slot = np.zeros(1, dtype=np.int64)
    vslot = np.zeros(1, dtype=np.float64)
    for bounds in _chunks(v.size, workers(n_workers)):
        lo, hi = bounds
        got = lib.bo_argmin_range(
            v.ctypes.data, lo, hi, slot.ctypes.data, vslot.ctypes.data
        )
        if got >= 0 and float(vslot[0]) < best_val:
            best_idx, best_val = int(got), float(vslot[0])
    return best_idx, best_val

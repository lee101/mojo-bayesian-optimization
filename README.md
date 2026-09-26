# mojo-bayesian-optimization

`mojo-bayesian-optimization` is the compute core of
[bayesian-optimization](https://pypi.org/project/bayesian-optimization/) 3.3.0
with the Gaussian-process factorisation, the candidate sweep and the
acquisition functions running as compiled Mojo. The acquisition classes keep
upstream's names, parameter validation and formulas, so the parity tests compare
`base_acq` against the real classes directly; the GP is compared against
scikit-learn, which is the regressor `bayes_opt` itself delegates to.

```python
import numpy as np
import mojo_bayesian_optimization as mbo

result = mbo.maximize(
    lambda p: float(np.sum((p - 0.3) ** 2)),
    bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
    n_iterations=25,
    random_state=17,
)
result.x, result.y
# (array([0.32616772, 0.30969796]), 0.0007788000104132174)
```

## Why this is a real port

Bayesian optimisation is linear algebra and transcendental functions in a tight
loop, run once per iteration over every candidate. The structure is worth
stating because it is what the kernels exploit:

    K = LL^T                     Cholesky, once per iteration
    alpha = K^-1 y               two triangular solves
    beta  = L^-1 y               one more, and the reason the sweep is cheap
    z     = L^-1 k(x)            one forward substitution per candidate
    mu(x) = z . beta             O(n)
    var(x)= k(x,x) - z . z       O(n)

The last three lines are the whole predictive sweep. Written the obvious way,
`K^-1 k` is a full triangular solve against a vector per candidate and the
formulation invites an `O(n^2)` product; collapsing it to one substitution and
two dot products is what makes a 20 000-candidate sweep cost `O(m n)`. On top of
that sit the acquisition functions, one `erfc` and one `exp` per candidate.

The candidate sweep is also the one loop in the package with no serial
dependency across candidates, so it is exported as a range-taking symbol and
split across a thread pool from the Python shim. Measured 1.8x on eight workers
at 200 000 candidates.

## Covered subset

| area | implemented API | where the work happens |
| --- | --- | --- |
| Surrogate | `GaussianProcess.fit`, `predict`, `log_marginal_likelihood` | Mojo: ARD kernel matrix, Cholesky, triangular solves, predictive sweep, marginal likelihood |
| Kernels | `rbf`, `matern` (5/2, the `bayes_opt` default), `matern32`, `matern12`, scalar or per-dimension length scales | Mojo |
| Acquisitions | `UpperConfidenceBound`, `ProbabilityOfImprovement`, `ExpectedImprovement` | Mojo: the elementwise formulas; Python: parameter validation and the exploration decay |

A note on the shared test venv: it was seen with `bayes_opt` 3.3.0 during
development, when every acquisition comparison below agreed exactly, and later
with 1.4.0, which has no `bayes_opt.acquisition` module at all. The parity tests
resolve the upstream class per test, so on a 1.x venv those six comparisons skip
with a stated reason and the rest of the suite — including the scikit-learn GP
parity, which is unaffected — still runs.
| Suggestion | `AcquisitionFunction.suggest` | Mojo: the whole random candidate sweep; Python: candidate drawing and bounds |
| Loop | `maximize` | Python; every numeric step inside it is a Mojo call |

Not implemented, and not invented:

- **Hyperparameter optimisation.** Upstream builds its regressor with
  scikit-learn's default `optimizer="fmin_l_bfgs_b"`, which refits the kernel
  length scale by maximising the marginal likelihood on every call. That is an
  L-BFGS-B loop over a non-convex objective; porting it would mean porting
  scipy. This port fixes the length scale and computes the marginal likelihood
  so a caller can drive their own optimiser.
- **The L-BFGS-B refinement inside `suggest`.** Upstream samples 10 000 random
  candidates and then refines the best with L-BFGS-B. Only the sweep is ported,
  which is the part that touches every candidate's acquisition value.
- **`TargetSpace`, parameter types, domain reduction, logging, persistence,
  `GPHedge`, `ConstantLiar`, constrained optimisation.** That is the domain
  machinery around the surrogate, not the surrogate. Use the real
  `bayes_opt` for it.
- **`BayesianOptimization.maximize` as a parity reference.** The driver wraps a
  queue, a logger and the domain machinery; the numeric contract worth testing
  is the acquisition and the surrogate, and both are compared against the real
  packages. `maximize` itself is tested against the analytic optimum of problems
  with a known minimum and against a random-search baseline.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-bayesian-optimization.so`. Set
`PYTHONPATH=python` when using the package outside a Pixi task.

## Performance

Best-of-9 wall clock in one process. The reference is the strongest available
implementation of the same computation: scikit-learn for the factorisation and
the predictive sweep (it goes through BLAS), SciPy for the normal CDF and PDF,
NumPy for the argmin. Every case checks agreement with that reference before
timing.

| case | reference | mojo, 1 core | mojo, 8 cores | result |
| --- | ---: | ---: | ---: | ---: |
| fit, n=200 d=4 | 6.61 ms | 4.53 ms | — | 1.46x faster |
| predict, 60 -> 20000 candidates | 230.61 ms | 133.51 ms | 99.99 ms | 2.31x faster |
| expected improvement over 20000 | 2.60 ms | 1.14 ms | — | 2.28x faster |
| one full EI step, 60 -> 20000 | 263.95 ms | 118.06 ms | — | 2.24x faster |
| predict, 60 -> 200000 candidates | — | 1550.13 ms | 855.90 ms | 1.81x faster |

The last row has no single-core reference because the point of the row is the
threading, not the comparison; it reports serial against eight workers.

This machine is shared, and the reference column moves by 20-50% between runs
while the Mojo columns stay put, so the ratios are indicative rather than
precise. Reproduce with:

```bash
pixi run bench
```

Where the win comes from: scikit-learn's predictive path builds an `n x m`
block and calls BLAS twice, and the second call re-reads the whole block to
extract a diagonal. The kernel here walks the same data once per candidate in a
single pass with no temporary, which is why the sweep wins even though the
arithmetic is identical. The acquisition row is a straight `erfc` loop against
SciPy's ufunc pair and wins on call overhead and vectorisation, not on
arithmetic.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-bayesian-optimization.so`.

The Python layer owns every array. Scratch buffers are allocated per call, which
keeps the exported symbols free of allocation and therefore free of `raises` — an
`@export ... abi("C")` function cannot be `raises`. Buffers cross the C ABI as
64-bit addresses and are rebuilt in Mojo as `Pointer[Float64, AnyOrigin[mut=True]]`,
because `@export` rejects a function whose parameter types are inferred.

The loops are serial on purpose. 1.2.0 cannot carry a pointer into a parallel
body, and these kernels are small and latency-bound; the candidate sweep is the
one place where chunking from the Python side is worth it, and it is done with a
`ThreadPoolExecutor` over the range-taking entry points, with one scratch buffer
per chunk so the chunks share no state.

Three details are easy to get wrong, and each is pinned by a test:

- **The in-place Cholesky must not blank the strict upper triangle.** Rows
  further down still need those entries, and the failure is a
  perfectly-shaped lower-triangular array full of plausible wrong numbers.
- **The predictive mean contracts against `L^-1 y`, not `L^-T y`.** With
  `k = L z`, `mu = k^T K^-1 y = z^T L^-1 y`. The two differ by one transpose,
  and both produce smooth, plausible-looking surfaces.
- **The predictive variance has to be rescaled by the target spread.** It comes
  out of the factorisation in normalised target units, exactly like the mean.
  Rescaling the mean and forgetting the variance is invisible whenever the
  targets have unit spread, so the test pins the ratio rather than a value.

Mojo emits FMA, and a Cholesky plus two triangular solves reorder the sums, so
the GP results are compared with an explicit tolerance (`rtol=1e-9` on the mean,
`rtol=1e-8` on the standard deviation) and the acquisitions with a tighter one
(`rtol=1e-10`) because a single `erfc` and a single `exp` are well conditioned.
Exact equality is asserted only where the operation genuinely is exact: the
argmin index, the kernel constants, the parameter bookkeeping, and the
zeroed upper triangle of the Cholesky.

## Tests

```bash
pixi run test
```

72 tests. `base_acq` for all three acquisitions against
`bayes_opt.acquisition`; EI and POI against their closed forms; GP `predict` and
the marginal likelihood against scikit-learn across four kernels, both
`normalize_y` settings and two noise levels; the Cholesky against NumPy plus a
`L L^T = K` reconstruction; the predictive variance against a hand-derived
one-observation closed form; interpolation at low noise; serial and threaded
sweeps compared bit for bit; and `maximize` against the analytic optimum of
three problems and against a random-search baseline.

## License

MIT

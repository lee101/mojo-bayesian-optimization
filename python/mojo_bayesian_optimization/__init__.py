"""mojo-bayesian-optimization: Bayesian optimisation with the GP in Mojo.

Installable alongside the real `bayesian-optimization` package, which the
parity tests compare against.
"""

from .acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from .gp import GaussianProcess
from .optimize import Result, maximize

__all__ = [
    "AcquisitionFunction",
    "ExpectedImprovement",
    "GaussianProcess",
    "ProbabilityOfImprovement",
    "Result",
    "UpperConfidenceBound",
    "maximize",
]
__version__ = "0.1.0"

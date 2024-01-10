from typing import Callable
import numpy as np


def check_grad(
    f: Callable[[np.ndarray], float],
    dfdx: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> float:
    """
    Given a function `f(x)` for a vector-valued input `x` and a
    gradient function `dfdx` representing `∂f/∂x`, return
    the error between `dfdx(x)` and `~dfdx(x)` where `~dfdx` is
    an finite-difference approximation of `∂f/∂x`.
    """

    approx_dfdx = np.zeros_like(x)
    supposed_dfdx = dfdx(x)

    for i in range(x.shape[0]):
        dx = np.zeros_like(x)
        dx[i] = epsilon

        y1 = f(x - dx)
        y2 = f(x + dx)

        approx_dfdx[i] = (y2 - y1) / (2 * epsilon)

    return float(
        np.linalg.norm(approx_dfdx - supposed_dfdx) / np.linalg.norm(approx_dfdx + supposed_dfdx)
    )

"""
Utility functions for DDA computations.
"""

import numpy as np
from numpy.typing import NDArray


def rmse(y_true: NDArray, y_pred: NDArray) -> float:
    """
    Compute Root Mean Square Error between true and predicted values.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE as a scalar float
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

"""
Parallel Processing Utilities for DDA Analysis

This module provides parallelization support for DDA computations
using multiprocessing to process windows in parallel.
"""

from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .dda_functions import deriv_all
from .utils import rmse


def process_window_st(
    wn: int,
    data: NDArray,
    TAU: list,
    dm: int,
    WL: int,
    WS: int,
    TM: int,
) -> Tuple[int, NDArray]:
    """
    Process a single window for ST analysis.

    Args:
        wn: Window number
        data: Input timeseries data
        TAU: List of two delay values
        dm: Derivative method parameter
        WL: Window length
        WS: Window shift
        TM: Maximum delay value

    Returns:
        Tuple of (window_number, coefficients array of shape (4,))
    """
    anf = wn * WS
    ende = anf + WL + TM + 2 * dm - 1

    window_data = data[anf : ende + 1]
    ddata = deriv_all(window_data, dm)
    window_data = window_data[dm:-dm]

    STD = np.std(window_data, ddof=1)
    DATA = (window_data - np.mean(window_data)) / STD
    dDATA = ddata / STD

    # Build design matrix M with delay coordinates
    M = np.column_stack(
        [
            DATA[TM - TAU[0] : len(DATA) - TAU[0]],
            DATA[TM - TAU[1] : len(DATA) - TAU[1]],
            (DATA[TM - TAU[0] : len(DATA) - TAU[0]]) ** 3,
        ]
    )

    # Slice derivative data
    dDATA_sliced = dDATA[TM:]

    # Solve for coefficients using lstsq (matches Julia's \ operator)
    coeffs = np.zeros(4)
    coeffs[:3] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]

    # Compute residual error
    coeffs[3] = rmse(dDATA_sliced, M @ coeffs[:3])

    return wn, coeffs


def process_window_ct(
    wn: int,
    data1: NDArray,
    data2: NDArray,
    TAU: list,
    dm: int,
    WL: int,
    WS: int,
    TM: int,
) -> Tuple[int, NDArray]:
    """
    Process a single window for CT analysis.

    Args:
        wn: Window number
        data1: First timeseries data
        data2: Second timeseries data
        TAU: List of two delay values
        dm: Derivative method parameter
        WL: Window length
        WS: Window shift
        TM: Maximum delay value

    Returns:
        Tuple of (window_number, coefficients array of shape (4,))
    """
    anf = wn * WS
    ende = anf + WL + TM + 2 * dm - 1

    # Process first timeseries
    window_data1 = data1[anf : ende + 1]
    ddata1 = deriv_all(window_data1, dm)
    window_data1 = window_data1[dm:-dm]

    STD1 = np.std(window_data1, ddof=1)
    DATA1 = (window_data1 - np.mean(window_data1)) / STD1
    dDATA1 = ddata1 / STD1

    # Process second timeseries
    window_data2 = data2[anf : ende + 1]
    ddata2 = deriv_all(window_data2, dm)
    window_data2 = window_data2[dm:-dm]

    STD2 = np.std(window_data2, ddof=1)
    DATA2 = (window_data2 - np.mean(window_data2)) / STD2
    dDATA2 = ddata2 / STD2

    # Build design matrices
    M1 = np.column_stack(
        [
            DATA1[TM - TAU[0] : len(DATA1) - TAU[0]],
            DATA1[TM - TAU[1] : len(DATA1) - TAU[1]],
            (DATA1[TM - TAU[0] : len(DATA1) - TAU[0]]) ** 3,
        ]
    )

    M2 = np.column_stack(
        [
            DATA2[TM - TAU[0] : len(DATA2) - TAU[0]],
            DATA2[TM - TAU[1] : len(DATA2) - TAU[1]],
            (DATA2[TM - TAU[0] : len(DATA2) - TAU[0]]) ** 3,
        ]
    )

    # Slice derivatives
    dDATA1_sliced = dDATA1[TM:]
    dDATA2_sliced = dDATA2[TM:]

    # Combine matrices and data
    M = np.vstack([M1, M2])
    dDATA_combined = np.concatenate([dDATA1_sliced, dDATA2_sliced])

    # Solve for coefficients using lstsq (matches Julia's \ operator)
    coeffs = np.zeros(4)
    coeffs[:3] = np.linalg.lstsq(M, dDATA_combined, rcond=None)[0]

    # Compute residual error
    coeffs[3] = rmse(dDATA_combined, M @ coeffs[:3])

    return wn, coeffs


def parallel_process_windows(
    process_func: Callable,
    num_windows: int,
    n_processes: Optional[int] = None,
    **kwargs,
) -> List[Tuple[int, NDArray]]:
    """
    Process windows in parallel using multiprocessing.

    Args:
        process_func: Function to process a single window
        num_windows: Total number of windows to process
        n_processes: Number of parallel processes (default: cpu_count)
        **kwargs: Additional arguments to pass to process_func

    Returns:
        List of (window_index, result) tuples
    """
    if n_processes is None:
        n_processes = min(cpu_count(), num_windows)
    elif n_processes <= 0:
        n_processes = cpu_count()

    # Create partial function with fixed arguments
    func = partial(process_func, **kwargs)

    # Process windows in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(func, range(num_windows))

    return results


def get_optimal_n_processes(num_windows: int, overhead_threshold: int = 10) -> int:
    """
    Determine optimal number of processes based on workload.

    Args:
        num_windows: Number of windows to process
        overhead_threshold: Minimum windows per process to justify parallelization

    Returns:
        Optimal number of processes
    """
    n_cores = cpu_count()

    # Don't parallelize if too few windows
    if num_windows < overhead_threshold:
        return 1

    # Use all cores if enough work
    if num_windows >= n_cores * overhead_threshold:
        return n_cores

    # Scale processes based on workload
    return max(1, min(n_cores, num_windows // overhead_threshold))

"""
Single Timeseries DDA Analysis Module

This module provides functions for performing DDA (Delay Differential Analysis)
on single timeseries data.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .dda_functions import build_design_matrix, deriv_all
from .dda_output_converter import convert_dda_output
from .dda_parallel import (
    get_optimal_n_processes,
    parallel_process_windows,
    process_window_st,
)
from .utils import rmse


def _process_window_st(
    wn: int,
    data: NDArray,
    TAU: list,
    dm: int,
    WL: int,
    WS: int,
    TM: int,
    model: list,
    nr_tau: int,
    order: int,
) -> Tuple[int, NDArray]:
    """
    Process a single window for ST DDA.

    Args:
        wn: Window number
        data: Input timeseries data
        TAU: List of delay values
        dm: Derivative method parameter
        WL: Window length
        WS: Window shift
        TM: Maximum delay
        model: List of model indices
        nr_tau: Number of tau values for monomial list
        order: Order of DDA

    Returns:
        Tuple containing the window number and the computed coefficients
    """
    # Extract relevant window data
    bounds = wn * WS, wn * WS + WL + TM + 2 * dm - 1
    wn_data = data[bounds[0] : bounds[1]]
    wn_model = model

    # Compute DDA coefficients for the window
    coeffs = np.zeros(len(wn_model) + 1)
    ddata = deriv_all(wn_data, dm)
    wn_data = wn_data[dm:-dm]

    STD = np.std(wn_data, ddof=1)  # Use sample std to match Julia
    DATA = (wn_data - np.mean(wn_data)) / STD
    dDATA = ddata / STD

    # Build design matrix from MODEL indices
    M = build_design_matrix(DATA, TAU, TM, model, nr_tau=nr_tau, order=order)

    # Slice derivative data AFTER matrix construction
    dDATA_sliced = dDATA[TM:]

    # Solve for coefficients using lstsq (matches Julia's \ operator)
    coeffs = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]

    # Compute residual error
    error = rmse(dDATA_sliced, M @ coeffs)

    return coeffs, error


def compute_st_single(
    data: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
    return_dict: bool = True,
    sampling_rate: Optional[float] = None,
    units: Optional[str] = None,
    parallel: bool = False,
    n_processes: Optional[int] = None,
    model: Optional[list] = None,
    nr_tau: int = 2,
) -> Union[NDArray, Dict[str, Any]]:
    """
    Compute single timeseries DDA structure coefficients.

    Args:
        data: Input timeseries data (1D array)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)
        return_dict: Return standardized dictionary format (default: True)
        sampling_rate: Sampling rate in Hz (optional)
        units: Physical units of the data (optional)
        parallel: Enable parallel processing of windows (default: False)
        n_processes: Number of parallel processes (default: auto-detect)
        model: List of 1-based model indices (default: None uses [1, 2, 3] for order=3)
        nr_tau: Number of tau values for monomial list (default: 2)

    Returns:
        If return_dict=True: Dictionary with coefficients, errors, and metadata
        If return_dict=False: Array of shape (WN, n_coeffs+1) containing ST coefficients
    """
    TM = max(TAU)
    WN = int(1 + np.floor((len(data) - (WL + TM + 2 * dm - 1)) / WS))

    n_coeffs = len(model)
    ST = np.full((WN, n_coeffs + 1), np.nan)

    if parallel and WN > 1:
        # Use parallel processing
        if n_processes is None:
            n_processes = get_optimal_n_processes(WN)

        # Process windows in parallel
        from functools import partial

        process_func = partial(process_window_st, data=data, TAU=TAU, dm=dm, WL=WL, WS=WS, TM=TM)

        results = parallel_process_windows(
            process_func,
            num_windows=WN,
            n_processes=n_processes,
            data=data,
            TAU=TAU,
            dm=dm,
            WL=WL,
            WS=WS,
            TM=TM,
        )

        # Store results
        for wn, coeffs in results:
            ST[wn, :] = coeffs
    else:
        # Sequential processing
        for wn in range(WN):
            _process_window_st()

    if return_dict:
        return convert_dda_output(
            coefficients_matrix=ST,
            algorithm="DDA_ST",
            delays=TAU,
            window_length=WL,
            window_shift=WS,
            derivative_method=dm,
            order=order,
            sampling_rate=sampling_rate,
            units=units,
        )
    else:
        return ST


def compute_st_multiple(
    Y: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
    return_dict: bool = True,
    sampling_rate: Optional[float] = None,
    units: Optional[str] = None,
    channel_names: Optional[list] = None,
    parallel: bool = False,
    n_processes: Optional[int] = None,
    model: Optional[list] = None,
    nr_tau: int = 2,
) -> Union[NDArray, Dict[str, Any]]:
    """
    Compute single timeseries DDA for multiple timeseries.

    Args:
        Y: Input data matrix (samples x channels)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)
        return_dict: Return standardized dictionary format (default: True)
        sampling_rate: Sampling rate in Hz (optional)
        units: Physical units of the data (optional)
        channel_names: Names of channels (optional)
        parallel: Enable parallel processing (default: False)
        n_processes: Number of parallel processes (default: auto-detect)
        model: List of 1-based model indices
        nr_tau: Number of tau values for monomial list (default: 2)

    Returns:
        If return_dict=True: Dictionary with coefficients, errors, and metadata
        If return_dict=False: Array of shape (WN, n_coeffs+1, n_channels) containing ST features
    """
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    TM = max(TAU)
    WN = int(1 + np.floor((Y.shape[0] - (WL + TM + 2 * dm - 1)) / WS))
    n_channels = Y.shape[1]
    n_coeffs = len(model)

    ST = np.full((WN, n_coeffs + 1, n_channels), np.nan)

    for n_Y in range(n_channels):
        if parallel and WN > 1:
            # Use parallel processing for this channel
            if n_processes is None:
                n_processes = get_optimal_n_processes(WN)

            # Process windows in parallel
            from functools import partial

            process_func = partial(
                process_window_st, data=Y[:, n_Y], TAU=TAU, dm=dm, WL=WL, WS=WS, TM=TM
            )

            results = parallel_process_windows(
                process_func,
                num_windows=WN,
                n_processes=n_processes,
                data=Y[:, n_Y],
                TAU=TAU,
                dm=dm,
                WL=WL,
                WS=WS,
                TM=TM,
            )

            # Store results
            for wn, coeffs in results:
                ST[wn, :, n_Y] = coeffs
        else:
            # Sequential processing
            for wn in range(WN):
                anf = wn * WS
                ende = anf + WL + TM + 2 * dm - 1

                data = Y[anf : ende + 1, n_Y]
                ddata = deriv_all(data, dm)
                data = data[dm:-dm]

                STD = np.std(data, ddof=1)  # Use sample std to match Julia
                DATA = (data - np.mean(data)) / STD
                dDATA = ddata / STD

                # Build design matrix from MODEL indices
                M = build_design_matrix(DATA, TAU, TM, model, nr_tau=nr_tau, order=order)

                # Slice derivative data AFTER matrix construction
                dDATA_sliced = dDATA[TM:]

                # Solve for coefficients using lstsq (matches Julia's \ operator)
                ST[wn, :n_coeffs, n_Y] = np.linalg.lstsq(M, dDATA_sliced, rcond=None)[0]

                # Compute residual error
                ST[wn, n_coeffs, n_Y] = rmse(dDATA_sliced, M @ ST[wn, :n_coeffs, n_Y])

    if return_dict:
        return convert_dda_output(
            coefficients_matrix=ST,
            algorithm="DDA_ST",
            delays=TAU,
            window_length=WL,
            window_shift=WS,
            derivative_method=dm,
            order=order,
            sampling_rate=sampling_rate,
            units=units,
            channel_names=channel_names,
        )
    else:
        return ST


def run_dda_st_external(
    FN_DATA: str,
    FN_DDA: str,
    MODEL: NDArray,
    TAU: list,
    dm: int = 4,
    DDAorder: int = 2,
    nr_delays: int = 2,
    WL: int = 2000,
    WS: int = 1000,
    CH_list: Optional[list] = None,
    platform_system: Optional[str] = None,
) -> Tuple[str, NDArray]:
    """
    Run external DDA executable for ST analysis.

    Args:
        FN_DATA: Input data filename
        FN_DDA: Output DDA filename
        MODEL: Model specification array
        TAU: List of delay values
        dm: Derivative method parameter
        DDAorder: Order of DDA
        nr_delays: Number of delays
        WL: Window length
        WS: Window shift
        CH_list: Optional channel list
        platform_system: Platform (Windows/Unix), auto-detect if None

    Returns:
        Tuple of (command string, loaded ST results)
    """
    import platform
    import subprocess

    if platform_system is None:
        platform_system = platform.system()

    # Platform-specific executable handling
    if platform_system == "Windows":
        executable = Path("run_DDA_AsciiEdf.exe")
        if not executable.exists():
            import shutil

            shutil.copy("run_DDA_AsciiEdf", str(executable))
        CMD = str(executable)
    else:
        CMD = str(Path("run_DDA_AsciiEdf"))

    # Build command
    CMD += " -ASCII"
    CMD += f" -MODEL {' '.join(map(str, MODEL))}"
    CMD += f" -TAU {' '.join(map(str, TAU))}"
    CMD += f" -dm {dm} -order {DDAorder} -nr_tau {nr_delays}"
    CMD += f" -DATA_FN {FN_DATA} -OUT_FN {FN_DDA}"
    CMD += f" -WL {WL} -WS {WS}"
    CMD += " -SELECT 1 0 0 0"  # ST only

    if CH_list:
        CMD += f" -CH_list {' '.join(map(str, CH_list))}"

    # Execute command
    if platform_system == "Windows":
        subprocess.run(CMD.split())
    else:
        subprocess.run(CMD, shell=True)

    # Load results
    ST_results = np.loadtxt(f"{FN_DDA}_ST")
    ST_results = ST_results[:, 2:]  # Skip first 2 columns

    return CMD, ST_results

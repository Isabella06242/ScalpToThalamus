"""
Dynamical Ergodicity (DE) DDA Analysis Module

This module provides functions for computing dynamical ergodicity measures
from DDA analysis results. It quantifies the relationship between single
timeseries and cross-timeseries analysis.
"""

from typing import Optional, Tuple

import os
import numpy as np
from numpy.typing import NDArray

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def compute_dynamical_ergodicity(
    ST: NDArray,
    CT: NDArray,
    channel_pairs: NDArray,
) -> NDArray:
    """
    Compute dynamical ergodicity matrix from ST and CT results.

    The dynamical ergodicity measure quantifies how well the coupling
    between two timeseries can be explained by their individual structures.

    Args:
        ST: Single timeseries results, shape (WN, 4, n_channels)
            Last dimension (index 3) contains the error values
        CT: Cross-timeseries results, shape (WN, 4, n_pairs)
            Last dimension (index 3) contains the error values
        channel_pairs: Array of channel pairs, shape (n_pairs, 2)
            0-based indices indicating which channels are paired

    Returns:
        Ergodicity matrix E of shape (n_channels, n_channels)
        E[i,j] = |mean([st_i, st_j])/ct_ij - 1|
    """
    n_channels = ST.shape[2]

    # Compute mean errors over windows
    st = np.mean(ST[:, -1, :], axis=0)  # Mean over windows, last column (errors)
    ct = np.mean(CT[:, -1, :], axis=0)  # Mean over windows, last column (errors)

    # Initialize ergodicity matrix
    E = np.full((n_channels, n_channels), np.nan)

    # Fill the matrix with ergodicity values
    for n_pair, (ch1, ch2) in enumerate(channel_pairs):
        # Ergodicity measure: |mean([st_i, st_j])/ct_ij - 1|
        E[ch1, ch2] = abs(np.mean([st[ch1], st[ch2]]) / ct[n_pair] - 1)
        E[ch2, ch1] = E[ch1, ch2]  # Symmetric matrix

    return E


def compute_dynamical_ergodicity_windowed(
    ST: NDArray,
    CT: NDArray,
    channel_pairs: NDArray,
) -> NDArray:
    """
    Compute a time-resolved ergodicity matrix — one matrix per window, instead
    of averaging the errors over windows first (cf. compute_dynamical_ergodicity).

    Args:
        ST: Single timeseries results, shape (WN, n_coeffs+1, n_channels).
            Last column (index -1) holds the per-window error.
        CT: Cross timeseries results, shape (WN, n_coeffs+1, n_pairs).
        channel_pairs: Array of channel pairs, shape (n_pairs, 2), 0-based.

    Returns:
        E of shape (WN, n_channels, n_channels) where
        E[w, i, j] = |mean([st_i(w), st_j(w)]) / ct_ij(w) - 1|.
    """
    WN = ST.shape[0]
    n_channels = ST.shape[2]

    st = ST[:, -1, :]  # (WN, n_channels), per-window error
    ct = CT[:, -1, :]  # (WN, n_pairs), per-window error

    E = np.full((WN, n_channels, n_channels), np.nan)
    for wn in range(WN):
        for n_pair, (ch1, ch2) in enumerate(channel_pairs):
            val = abs(np.mean([st[wn, ch1], st[wn, ch2]]) / ct[wn, n_pair] - 1)
            E[wn, ch1, ch2] = val
            E[wn, ch2, ch1] = val
    return E


def save_de_windowed_csv(
    E_windowed: NDArray,
    channel_pairs: NDArray,
    csv_path: str,
    channel_names: Optional[list] = None,
    WS: Optional[int] = None,
    sampling_rate: Optional[float] = None,
) -> None:
    """
    Save the time-resolved ergodicity to CSV in long format: one row per
    window per pair (window, ch1, ch2, ergodicity), so the temporal change
    is preserved.

    Args:
        E_windowed: Array of shape (WN, n_channels, n_channels) from
            compute_dynamical_ergodicity_windowed.
        channel_pairs: Array of shape (n_pairs, 2), 0-based channel indices.
        csv_path: Output CSV path.
        channel_names: Optional labels for channels (defaults to 0-based index).
        WS: Window shift in samples. If given (with sampling_rate), a
            't_sec' column is added marking each window's start time.
        sampling_rate: Sampling rate in Hz, used with WS for the time column.
    """
    import pandas as pd

    WN = E_windowed.shape[0]

    def _name(idx):
        return channel_names[idx] if channel_names is not None else idx

    rows = []
    for wn in range(WN):
        for ch1, ch2 in channel_pairs:
            n1, n2 = _name(ch1), _name(ch2)
            row = {"window": wn, "pair": f"{n1}-{n2}", "ch1": n1, "ch2": n2,
                   "ergodicity": E_windowed[wn, ch1, ch2]}
            if WS is not None and sampling_rate:
                row["t_sec"] = wn * WS / sampling_rate
            rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  saved {csv_path} ({len(rows)} rows)")


def save_st_csv(
    ST: NDArray,
    csv_path: str,
    channel_names: Optional[list] = None,
    average_windows: bool = False,
) -> None:
    """
    Save single-timeseries (ST) DDA results to CSV, one row per channel/window.

    Args:
        ST: ST array of shape (WN, n_coeffs+1, n_channels).
            The first n_coeffs columns are structure coefficients (a1..aN);
            the last column is the residual error.
        csv_path: Output CSV path.
        channel_names: Optional labels for channels (defaults to 0-based index).
        average_windows: If True, average over windows so there is one row per
            channel (matching what compute_dynamical_ergodicity uses). If False,
            keep every window (long format with a 'window' column).
    """
    import pandas as pd

    WN, n_cols, n_channels = ST.shape
    n_coeffs = n_cols - 1
    coeff_cols = [f"a{i + 1}" for i in range(n_coeffs)]

    rows = []
    for ch in range(n_channels):
        label = channel_names[ch] if channel_names is not None else ch
        if average_windows:
            vals = np.nanmean(ST[:, :, ch], axis=0)
            row = {"channel": label}
            row.update({c: vals[i] for i, c in enumerate(coeff_cols)})
            row["error"] = vals[-1]
            rows.append(row)
        else:
            for wn in range(WN):
                vals = ST[wn, :, ch]
                row = {"channel": label, "window": wn}
                row.update({c: vals[i] for i, c in enumerate(coeff_cols)})
                row["error"] = vals[-1]
                rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  saved {csv_path} ({len(rows)} rows)")


def save_ct_csv(
    CT: NDArray,
    channel_pairs: NDArray,
    csv_path: str,
    channel_names: Optional[list] = None,
    average_windows: bool = False,
) -> None:
    """
    Save cross-timeseries (CT) DDA results to CSV, one row per pair/window.

    Args:
        CT: CT array of shape (WN, n_coeffs+1, n_pairs).
            The first n_coeffs columns are structure coefficients (a1..aN);
            the last column is the residual error.
        channel_pairs: Array of shape (n_pairs, 2) with 0-based channel indices.
        csv_path: Output CSV path.
        channel_names: Optional labels for channels (defaults to 0-based index).
        average_windows: If True, average over windows so there is one row per
            pair (matching what compute_dynamical_ergodicity uses).
    """
    import pandas as pd

    WN, n_cols, n_pairs = CT.shape
    n_coeffs = n_cols - 1
    coeff_cols = [f"a{i + 1}" for i in range(n_coeffs)]

    def _name(idx):
        return channel_names[idx] if channel_names is not None else idx

    rows = []
    for p in range(n_pairs):
        ch1, ch2 = channel_pairs[p]
        base = {"pair": p, "ch1": _name(ch1), "ch2": _name(ch2)}
        if average_windows:
            vals = np.nanmean(CT[:, :, p], axis=0)
            row = dict(base)
            row.update({c: vals[i] for i, c in enumerate(coeff_cols)})
            row["error"] = vals[-1]
            rows.append(row)
        else:
            for wn in range(WN):
                vals = CT[wn, :, p]
                row = dict(base, window=wn)
                row.update({c: vals[i] for i, c in enumerate(coeff_cols)})
                row["error"] = vals[-1]
                rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  saved {csv_path} ({len(rows)} rows)")


def plot_ergodicity_heatmap(
    E: NDArray,
    channel_labels: Optional[list] = None,
    title: str = "Dynamical Ergodicity Heatmap",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = "ergodicity_heatmap.png",
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".2e",
) -> None:
    """
    Create and display a heatmap of the dynamical ergodicity matrix.

    Args:
        E: Ergodicity matrix
        channel_labels: Optional labels for channels
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (None to not save)
        cmap: Colormap name
        annot: Whether to annotate cells with values
        fmt: Format string for annotations

    Raises:
        ImportError: If matplotlib or seaborn are not installed
    """
    if not HAS_PLOTTING:
        raise ImportError(
            "Plotting requires matplotlib and seaborn.\n"
            "Install with: pip install matplotlib seaborn"
        )

    plt.figure(figsize=figsize)

    # Create heatmap
    # Handle None labels
    xticklabels = channel_labels if channel_labels is not None else True
    yticklabels = channel_labels if channel_labels is not None else True

    sns.heatmap(
        E,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        square=True,
        cbar_kws={"label": "Ergodicity"},
    )

    plt.title(title)
    plt.xlabel("Channel")
    plt.ylabel("Channel")

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def analyze_ergodicity_statistics(
    E: NDArray,
    threshold: float = 0.1,
) -> dict:
    """
    Compute statistics about the ergodicity matrix.

    Args:
        E: Ergodicity matrix
        threshold: Threshold for considering channels as ergodic

    Returns:
        Dictionary containing various statistics:
        - mean: Mean ergodicity value
        - std: Standard deviation
        - min: Minimum value (excluding diagonal)
        - max: Maximum value
        - n_ergodic: Number of ergodic pairs (below threshold)
        - ergodic_pairs: List of ergodic channel pairs
    """
    # Extract upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(E, k=1)
    upper_values = E[upper_triangle_indices]

    # Remove NaN values
    valid_values = upper_values[~np.isnan(upper_values)]

    # Find ergodic pairs
    ergodic_mask = valid_values < threshold
    ergodic_pairs = []

    for idx, is_ergodic in enumerate(ergodic_mask):
        if is_ergodic:
            i, j = upper_triangle_indices[0][idx], upper_triangle_indices[1][idx]
            ergodic_pairs.append((i, j))

    stats = {
        "mean": np.mean(valid_values),
        "std": np.std(valid_values),
        "min": np.min(valid_values),
        "max": np.max(valid_values),
        "n_ergodic": len(ergodic_pairs),
        "ergodic_pairs": ergodic_pairs,
        "total_pairs": len(valid_values),
        "ergodic_fraction": len(ergodic_pairs) / len(valid_values) if len(valid_values) > 0 else 0,
    }

    return stats


def compare_with_external_de(
    E_computed: NDArray,
    E_external: NDArray,
    tolerance: float = 1e-15,
) -> Tuple[float, bool]:
    """
    Compare computed ergodicity matrix with external results.

    Args:
        E_computed: Computed ergodicity matrix
        E_external: External/reference ergodicity matrix
        tolerance: Tolerance for considering values equal

    Returns:
        Tuple of (mean_error, is_within_tolerance)
    """
    # Compute difference only for non-NaN values
    valid_mask = ~np.isnan(E_computed) & ~np.isnan(E_external)

    if not np.any(valid_mask):
        return np.nan, False

    diff = E_computed[valid_mask] - E_external[valid_mask]
    mean_error = np.mean(np.abs(diff))
    is_within_tolerance = np.all(np.abs(diff) < tolerance)

    return mean_error, is_within_tolerance


def run_full_de_analysis(
    FN_DATA: str,
    Y: NDArray,
    TAU: list,
    dm: int = 4,
    order: int = 3,
    WL: int = 2000,
    WS: int = 1000,
    plot: bool = True,
    save_plot: Optional[str] = "ergodicity_heatmap.png",
    save_windowed_de: bool = True,
    sampling_rate: Optional[float] = None,
    channel_names: Optional[list] = None,
) -> Tuple[NDArray, dict]:
    """
    Run complete dynamical ergodicity analysis on data.

    Args:
        Y: Input data matrix (samples x channels)
        TAU: List of two delay values [tau1, tau2]
        dm: Derivative method parameter (default: 4)
        order: Order of DDA (default: 3)
        WL: Window length (default: 2000)
        WS: Window shift (default: 1000)
        plot: Whether to create heatmap plot
        save_plot: Path to save plot
        save_windowed_de: If True, also compute and save the time-resolved
            (per-window) ergodicity to CSV instead of only the window-averaged E.
        sampling_rate: Sampling rate in Hz; if given, the windowed DE CSV gets a
            't_sec' column marking each window's start time.
        channel_names: Optional 10-20 channel labels (length n_channels). Used
            to label channels/pairs in the ST, CT, and DE CSV outputs.

    Returns:
        Tuple of (ergodicity matrix, statistics dictionary)
    """
    from dda_ct import compute_ct_multiple
    from dda_st import compute_st_multiple

    # Clean base name: strip directory and extension from FN_DATA
    # e.g. 'test_10/EM40_Sz18_scalp.mat' -> 'EM40_Sz18_scalp'
    base = os.path.splitext(os.path.basename(FN_DATA))[0]

    # Compute ST for all channels (need raw array, not the dict format)
    ST = compute_st_multiple(Y, TAU, dm, order, WL, WS, return_dict=False)

    # save ST results (per-channel)
    OUT_DIR = 'ST_test_10'
    os.makedirs(OUT_DIR, exist_ok=True)
    name = base
    np.save(os.path.join(OUT_DIR, f"{name}_ST.npy"), ST)
    save_st_csv(ST, os.path.join(OUT_DIR, f"{name}_ST.csv"), channel_names=channel_names)
    print(f"  saved {name}_ST.npy and {name}_ST.csv")


    # Compute CT for all channel pairs
    CT, channel_pairs = compute_ct_multiple(Y, TAU, dm, order, WL, WS)

    # save CT results (per-pair)
    OUT_DIR = 'CT_test_10'
    os.makedirs(OUT_DIR, exist_ok=True)
    name = base
    np.save(os.path.join(OUT_DIR, f"{name}_CT.npy"), CT)
    save_ct_csv(CT, channel_pairs, os.path.join(OUT_DIR, f"{name}_CT.csv"), channel_names=channel_names)
    print(f"  saved {name}_CT.npy and {name}_CT.csv")


    # Compute ergodicity matrix (window-averaged)
    E = compute_dynamical_ergodicity(ST, CT, channel_pairs)

    # Optionally save the time-resolved (per-window) ergodicity
    if save_windowed_de:
        OUT_DIR = 'DE_test_10'
        os.makedirs(OUT_DIR, exist_ok=True)
        name = base
        E_windowed = compute_dynamical_ergodicity_windowed(ST, CT, channel_pairs)
        np.save(os.path.join(OUT_DIR, f"{name}_DE_windowed.npy"), E_windowed)
        save_de_windowed_csv(
            E_windowed,
            channel_pairs,
            os.path.join(OUT_DIR, f"{name}_DE_windowed.csv"),
            channel_names=channel_names,
            WS=WS,
            sampling_rate=sampling_rate,
        )

    # Compute statistics
    stats = analyze_ergodicity_statistics(E)

    # Create plot if requested
    if plot:
        plot_ergodicity_heatmap(E, save_path=save_plot)

    return E, stats

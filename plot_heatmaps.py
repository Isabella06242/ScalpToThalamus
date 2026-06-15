"""
Time-resolved heatmaps for DDA results.

Each figure: horizontal axis = time (s), vertical axis = channel names (ST)
or pair names in 'O1-O2' format (CT, DE). Color encodes the metric
(DDA error for ST/CT, ergodicity for DE).
"""

import os
import glob

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROW_IN = 0.34  # inches of figure height per row (channel/pair)


def _row_fontsize(row_in=ROW_IN):
    """Largest y-label font (pt) that fits one row without overlapping.

    A row is row_in inches tall = row_in*72 points; use 80% of that so
    adjacent labels keep a small gap. Capped so a few-row plot (ST) isn't
    absurdly large.
    """
    return max(5, min(16, row_in * 72 * 0.8))


def _get_fs(mat_file):
    """Read sampling rate (Hz) from a FieldTrip .mat file."""
    with h5py.File(mat_file, "r") as f:
        return float(np.array(f["data/fsample"]).flatten()[0])


def _heatmap(matrix, row_labels, times, title, cbar_label, save_path,
             cmap="viridis"):
    """Render one time-resolved heatmap (rows x time)."""
    n_rows = matrix.shape[0]
    # Height scales with number of rows so labels stay legible
    fig, ax = plt.subplots(figsize=(14, max(3, ROW_IN * n_rows)))

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        extent=[times[0], times[-1], n_rows - 0.5, -0.5],
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=_row_fontsize())
    ax.set_xlabel("time (s)")
    ax.set_title(title)

    # Thin color band (small fraction + high aspect), very large tick numbers
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.012, aspect=50)
    cbar.set_label(cbar_label, fontsize=22)
    cbar.ax.tick_params(labelsize=28)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.close(fig)


def _pivot(df, row_key, value, time):
    """Pivot long df to (rows x windows) preserving first-appearance row order."""
    order = list(dict.fromkeys(df[row_key]))  # appearance order, not sorted
    wide = df.pivot_table(index=row_key, columns="window", values=value)
    wide = wide.reindex(order)
    return wide.values, order, wide.columns.values


def _fluctuation(matrix, row_labels, times, title, ylabel, save_path,
                 spacing_sd=4.0):
    """
    'Fluctuation' view as stacked traces (like the raw EEG montage): one trace
    per row (channel or pair), value over time, vertically offset so they don't
    overlap. Vertical axis = channel/pair names, horizontal axis = time (s).

    Args:
        spacing_sd: Vertical gap between traces, in units of the typical row
            std. Larger = more separation.
    """
    n_rows = matrix.shape[0]

    # constant offset between traces, scaled by the typical fluctuation size
    scale = np.nanmedian(np.nanstd(matrix, axis=1))
    if not np.isfinite(scale) or scale == 0:
        scale = np.nanmax(matrix) or 1.0
    offset = spacing_sd * scale
    offsets = np.arange(n_rows) * offset  # row 0 at bottom

    fig, ax = plt.subplots(figsize=(14, max(4, ROW_IN * n_rows)))
    for i in range(n_rows):
        # plot bottom-up so the first row sits at the top of the y-ticks
        ax.plot(times, matrix[i] + offsets[n_rows - 1 - i], color="red", lw=0.5)

    ax.set_yticks(offsets)
    ax.set_yticklabels(row_labels[::-1], fontsize=_row_fontsize())
    ax.set_ylim(-offset, offsets[-1] + offset)
    ax.set_xlim(0, times[-1])  # start the time axis exactly at 0, no left margin
    ax.set_xlabel("time (s)")
    ax.set_title(f"{title}  ({ylabel}, offset per row)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.close(fig)


def plot_st_heatmap(st_csv, fs, WS, value="error", save_path=None):
    """ST: rows = channels, x = time, color = `value` (default DDA error)."""
    df = pd.read_csv(st_csv)
    mat, rows, windows = _pivot(df, "channel", value, "window")
    times = windows * WS / fs
    _heatmap(mat, rows, times,
             title=f"ST {value} — {os.path.basename(st_csv)}",
             cbar_label=f"ST {value}", save_path=save_path)


def plot_ct_heatmap(ct_csv, fs, WS, value="error", save_path=None):
    """CT: rows = pairs (ch1-ch2), x = time, color = `value` (default error)."""
    df = pd.read_csv(ct_csv)
    df["pair_name"] = df["ch1"].astype(str) + "-" + df["ch2"].astype(str)
    mat, rows, windows = _pivot(df, "pair_name", value, "window")
    times = windows * WS / fs
    _heatmap(mat, rows, times,
             title=f"CT {value} — {os.path.basename(ct_csv)}",
             cbar_label=f"CT {value}", save_path=save_path)


def plot_de_heatmap(de_csv, fs, WS, save_path=None):
    """DE: rows = pairs (ch1-ch2), x = time, color = ergodicity."""
    df = pd.read_csv(de_csv)
    # DE csv already has a 'pair' column in 'O1-O2' format
    mat, rows, windows = _pivot(df, "pair", "ergodicity", "window")
    times = windows * WS / fs
    _heatmap(mat, rows, times,
             title=f"DE ergodicity — {os.path.basename(de_csv)}",
             cbar_label="ergodicity", save_path=save_path)


def plot_st_fluctuation(st_csv, fs, WS, value="error", save_path=None):
    """ST fluctuation: one line per channel, value over time."""
    df = pd.read_csv(st_csv)
    mat, rows, windows = _pivot(df, "channel", value, "window")
    times = windows * WS / fs
    _fluctuation(mat, rows, times,
                 title=f"ST {value} fluctuation — {os.path.basename(st_csv)}",
                 ylabel=f"ST {value}", save_path=save_path)


def plot_ct_fluctuation(ct_csv, fs, WS, value="error", save_path=None):
    """CT fluctuation: one line per pair, value over time."""
    df = pd.read_csv(ct_csv)
    df["pair_name"] = df["ch1"].astype(str) + "-" + df["ch2"].astype(str)
    mat, rows, windows = _pivot(df, "pair_name", value, "window")
    times = windows * WS / fs
    _fluctuation(mat, rows, times,
                 title=f"CT {value} fluctuation — {os.path.basename(ct_csv)}",
                 ylabel=f"CT {value}", save_path=save_path)


def plot_de_fluctuation(de_csv, fs, WS, save_path=None):
    """DE fluctuation: one line per pair, ergodicity over time."""
    df = pd.read_csv(de_csv)
    mat, rows, windows = _pivot(df, "pair", "ergodicity", "window")
    times = windows * WS / fs
    _fluctuation(mat, rows, times,
                 title=f"DE ergodicity fluctuation — {os.path.basename(de_csv)}",
                 ylabel="ergodicity", save_path=save_path)


if __name__ == "__main__":
    WS = 256  # must match the value used in DDA.py
    GRAPH_DIR = "test_10_graphs"
    os.makedirs(GRAPH_DIR, exist_ok=True)

    mat = sorted(glob.glob("test_10/*.mat"))[0]
    base = os.path.splitext(os.path.basename(mat))[0]
    fs = _get_fs(mat)

    st_csv = os.path.join("ST_test_10", f"{base}_ST.csv")
    ct_csv = os.path.join("CT_test_10", f"{base}_CT.csv")
    de_csv = os.path.join("DE_test_10", f"{base}_DE_windowed.csv")

    plot_st_heatmap(st_csv, fs, WS,
                    save_path=os.path.join(GRAPH_DIR, f"{base}_ST_heatmap.png"))
    plot_ct_heatmap(ct_csv, fs, WS,
                    save_path=os.path.join(GRAPH_DIR, f"{base}_CT_heatmap.png"))
    plot_de_heatmap(de_csv, fs, WS,
                    save_path=os.path.join(GRAPH_DIR, f"{base}_DE_heatmap.png"))

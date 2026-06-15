"""
Plot raw EEG and the temporal (per-window) Dynamical Ergodicity on a shared
time axis. Raw signal and DE live on opposite y-axes (twinx) because their
scales differ by orders of magnitude.
"""

import os
import glob

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_raw(mat_file):
    """Return (Y, channel_names, fs) from a FieldTrip .mat file."""
    with h5py.File(mat_file, "r") as f:
        trial_ref = f["data/trial"][0, 0]
        Y = np.array(f[trial_ref])  # (samples, channels)
        label_refs = f["data/label"][:].flatten()
        names = ["".join(chr(c) for c in np.array(f[r]).flatten()) for r in label_refs]
        fs = float(np.array(f["data/fsample"]).flatten()[0])
    return Y, names, fs


def plot_raw_and_de(
    mat_file,
    de_csv,
    raw_channel="FP1",
    pair=None,
    save_path=None,
):
    """
    Overlay one raw channel and the temporal DE on the same graph.

    Args:
        mat_file: Path to the seizure .mat file (raw EEG).
        de_csv: Path to the *_DE_windowed.csv produced by run_full_de_analysis.
        raw_channel: 10-20 name of the channel to show raw (default 'FP1').
        pair: DE pair label e.g. 'T3-T5'. If None, plot the mean DE over all
            pairs in each window (a global ergodicity summary).
        save_path: If given, save the figure there (PNG).
    """
    Y, names, fs = load_raw(mat_file)
    t_raw = np.arange(Y.shape[0]) / fs

    de = pd.read_csv(de_csv)
    # time axis for DE: use t_sec if present, else window index
    de_t_col = "t_sec" if "t_sec" in de.columns else "window"

    if pair is None:
        de_series = de.groupby(de_t_col)["ergodicity"].mean()
        de_label = "mean DE (all pairs)"
        de_x, de_y = de_series.index.values, de_series.values
    else:
        sub = de[de["pair"] == pair].sort_values(de_t_col)
        de_x, de_y = sub[de_t_col].values, sub["ergodicity"].values
        de_label = f"DE {pair}"

    ch_idx = names.index(raw_channel)

    fig, ax_raw = plt.subplots(figsize=(14, 5))

    # Raw EEG on the left axis (light, in the background)
    ax_raw.plot(t_raw, Y[:, ch_idx], color="0.6", lw=0.5, label=f"raw {raw_channel}")
    ax_raw.set_xlabel("time (s)")
    ax_raw.set_ylabel(f"raw {raw_channel} (a.u.)", color="0.4")
    ax_raw.tick_params(axis="y", labelcolor="0.4")

    # DE on the right axis (bold, in front)
    ax_de = ax_raw.twinx()
    ax_de.plot(de_x, de_y, color="crimson", lw=1.8, marker="o", ms=3, label=de_label)
    ax_de.set_ylabel(de_label, color="crimson")
    ax_de.tick_params(axis="y", labelcolor="crimson")

    fig.suptitle(f"{os.path.basename(mat_file)}  —  raw {raw_channel} vs {de_label}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.show()


def plot_raw_stacked(mat_file, save_path=None, spacing_sd=6.0):
    """
    Plot raw EEG, one trace per channel, stacked vertically (classic montage
    view). Vertical axis = channel names, horizontal axis = time (s).

    Args:
        mat_file: Path to the seizure .mat file.
        save_path: If given, save the figure there (PNG).
        spacing_sd: Vertical gap between channels, in units of each channel's
            std. Larger = more separation, less overlap.
    """
    Y, names, fs = load_raw(mat_file)
    t = np.arange(Y.shape[0]) / fs
    n_ch = Y.shape[1]

    # Offset each channel by a constant so traces don't overlap. Use the median
    # channel std as the scale so the spacing is consistent across channels.
    scale = np.nanmedian(np.nanstd(Y, axis=0))
    offset = spacing_sd * scale
    offsets = np.arange(n_ch) * offset  # channel 0 at bottom

    fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * n_ch)))
    for i in range(n_ch):
        # plot bottom-up so the first channel sits at the top of the y-ticks
        ax.plot(t, Y[:, i] + offsets[n_ch - 1 - i], color="black", lw=0.4)

    ax.set_yticks(offsets)
    ax.set_yticklabels(names[::-1])  # match the bottom-up offset order
    ax.set_ylim(-offset, offsets[-1] + offset)
    ax.set_xlabel("time (s)")
    ax.set_title(f"raw EEG per channel — {os.path.basename(mat_file)}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Default: first seizure file + its DE csv, channel FP1, mean DE over pairs
    GRAPH_DIR = "test_10_graphs"
    os.makedirs(GRAPH_DIR, exist_ok=True)

    mat = sorted(glob.glob("test_10/*.mat"))[0]
    base = os.path.splitext(os.path.basename(mat))[0]
    de_csv = os.path.join("DE_test_10", f"{base}_DE_windowed.csv")

    # raw EEG per channel, standalone (no DE)
    plot_raw_stacked(
        mat,
        save_path=os.path.join(GRAPH_DIR, f"{base}_raw_stacked.png"),
    )

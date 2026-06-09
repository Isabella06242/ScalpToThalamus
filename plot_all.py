"""
Batch plotting for every seizure in test_10.

For each .mat file produces, under test_10_graphs/<seizure>/:
  - raw stacked EEG (per channel)
  - ST  heatmap + fluctuation
  - CT  heatmap + fluctuation
  - DE  heatmap + fluctuation

Requires the per-seizure CSVs produced by DDA.py (ST_test_10, CT_test_10,
DE_test_10). Set matplotlib to a non-interactive backend so it runs headless.
"""

import os
import glob

import matplotlib
matplotlib.use("Agg")

from plot_de import plot_raw_stacked, load_raw
from plot_heatmaps import (
    plot_st_heatmap, plot_ct_heatmap, plot_de_heatmap,
    plot_st_fluctuation, plot_ct_fluctuation, plot_de_fluctuation,
)

DATA_DIR = "test_10"
GRAPH_DIR = "test_10_graphs"
WS = 256  # must match the window shift used in DDA.py


def plot_seizure(mat_file):
    base = os.path.splitext(os.path.basename(mat_file))[0]
    out_dir = os.path.join(GRAPH_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    st_csv = os.path.join("ST_test_10", f"{base}_ST.csv")
    ct_csv = os.path.join("CT_test_10", f"{base}_CT.csv")
    de_csv = os.path.join("DE_test_10", f"{base}_DE_windowed.csv")

    missing = [p for p in (st_csv, ct_csv, de_csv) if not os.path.exists(p)]
    if missing:
        print(f"  SKIP {base}: missing {missing}")
        return

    # sampling rate from the raw file (used to convert windows -> seconds)
    _, _, fs = load_raw(mat_file)

    # raw EEG per channel
    plot_raw_stacked(mat_file, save_path=os.path.join(out_dir, f"{base}_raw_stacked.png"))

    # ST
    plot_st_heatmap(st_csv, fs, WS, save_path=os.path.join(out_dir, f"{base}_ST_heatmap.png"))
    plot_st_fluctuation(st_csv, fs, WS, save_path=os.path.join(out_dir, f"{base}_ST_fluct.png"))

    # CT
    plot_ct_heatmap(ct_csv, fs, WS, save_path=os.path.join(out_dir, f"{base}_CT_heatmap.png"))
    plot_ct_fluctuation(ct_csv, fs, WS, save_path=os.path.join(out_dir, f"{base}_CT_fluct.png"))

    # DE
    plot_de_heatmap(de_csv, fs, WS, save_path=os.path.join(out_dir, f"{base}_DE_heatmap.png"))
    plot_de_fluctuation(de_csv, fs, WS, save_path=os.path.join(out_dir, f"{base}_DE_fluct.png"))


if __name__ == "__main__":
    os.makedirs(GRAPH_DIR, exist_ok=True)
    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))
    print(f"plotting {len(mat_files)} seizures")
    for mat_file in mat_files:
        print(f"processing {os.path.basename(mat_file)}")
        plot_seizure(mat_file)
    print("done")

"""
Shared data loading + feature extraction for thalamic-spread SVM classification.

Unit of classification = one EEG channel. Each channel's feature vector
summarizes its DDA coefficient time series (a1, a2, a3, RMSE across all sliding
windows) with simple distributional statistics. The label is the seizure
recording's known spread-to-thalamus class.

Binary task for now (weak spread excluded):
    MG136_sz2_35_34    -> 0  "no_spread"
    MG112b_Sz17_35_34  -> 1  "strong_spread"

Both train_svm.py and test_svm.py import load_split() from here so they operate
on the EXACT same stratified partition (same seed + same channel ordering).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# recording folder -> (label, label name)
DATASETS = {
    "MG136_sz2_35_34":   (0, "no_spread"),
    "MG112b_Sz17_35_34": (1, "strong_spread"),
}
CLASS_NAMES = ["no_spread", "strong_spread"]

COEFFS = ["a1", "a2", "a3", "RMSE"]
_STATS = ["mean", "std", "median", "min", "max"]
FEATURE_NAMES = [f"{c}_{s}" for c in COEFFS for s in _STATS]

N_CHANNELS = 276


def _channel_features(csv_path):
    """Summarize one channel's DDA time series into a fixed-length feature vector.

    Flat windows in the DDA output can be NaN (zero-variance segments), so all
    statistics are NaN-aware. A fully-NaN coefficient column yields NaN features
    here; build_feature_table() imputes those afterwards.
    """
    vals = pd.read_csv(csv_path, usecols=COEFFS)[COEFFS].to_numpy(dtype=float)
    feats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        for j in range(vals.shape[1]):
            col = vals[:, j]
            feats.extend([
                np.nanmean(col),
                np.nanstd(col),
                np.nanmedian(col),
                np.nanmin(col),
                np.nanmax(col),
            ])
    return np.asarray(feats, dtype=float)


def build_feature_table(root="."):
    """Return (X, y, ids) over every channel of both recordings."""
    root = Path(root)
    X, y, ids = [], [], []
    for folder, (label, _name) in DATASETS.items():
        d = root / folder
        for ch in range(1, N_CHANNELS + 1):
            f = d / f"dda_coefficients_timeseries_{ch}.csv"
            X.append(_channel_features(f))
            y.append(label)
            ids.append(f"{folder}:ch{ch}")
    X = np.vstack(X)
    y = np.asarray(y)
    ids = np.asarray(ids)

    # Impute any remaining NaN feature (e.g. an all-NaN channel) with its
    # column median so the SVM never sees NaN.
    col_med = np.nanmedian(X, axis=0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = col_med[cols]
    return X, y, ids


def load_split(root=".", train_frac=0.30, seed=42):
    """Stratified train/test split shared by the train and test scripts.

    train_frac of the channels (per class) go to training, the rest to testing.
    Identical inputs + seed => identical partition in both scripts.
    """
    X, y, ids = build_feature_table(root)
    return train_test_split(
        X, y, ids,
        train_size=train_frac,
        random_state=seed,
        stratify=y,
    )

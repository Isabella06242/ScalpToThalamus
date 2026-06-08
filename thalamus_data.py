"""
Shared data loading + feature extraction for thalamic-spread SVM classification.

Unit of classification = one EEG channel. Each channel's feature vector
summarizes its DDA coefficient time series (a1, a2, a3, RMSE across all sliding
windows) with simple distributional statistics. The label is the seizure
recording's known spread-to-thalamus class.

Binary task for now (weak spread excluded):
    MG136_sz2_35_34    -> 0  "no_spread"     (patient MG136)
    MG112b_Sz17_35_34  -> 1  "strong_spread" (patient MG112b)

The train/test split is at the PATIENT level: every channel (and every seizure)
belonging to a patient goes entirely to train OR test, never both. This avoids
leakage from the same patient appearing on both sides. The 30/70 fraction is
therefore applied to the number of *patients*, not channels. Both train_svm.py
and test_svm.py import load_split() so they operate on the EXACT same partition
(same seed + same ordering).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# recording folder -> metadata. "patient" is the grouping key for the split, so
# multiple seizures from one patient (e.g. MG90b_Sz5/Sz6/Sz7) must share the
# same patient string here.
DATASETS = {
    "MG136_sz2_35_34":   {"label": 0, "name": "no_spread",     "patient": "MG136"},
    "MG112b_Sz17_35_34": {"label": 1, "name": "strong_spread", "patient": "MG112b"},
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
    """Return (X, y, ids, groups) over every channel of every recording.

    groups[i] is the patient that channel i belongs to -- the grouping key used
    for the patient-level train/test split.
    """
    root = Path(root)
    X, y, ids, groups = [], [], [], []
    for folder, meta in DATASETS.items():
        d = root / folder
        for ch in range(1, N_CHANNELS + 1):
            f = d / f"dda_coefficients_timeseries_{ch}.csv"
            X.append(_channel_features(f))
            y.append(meta["label"])
            ids.append(f"{folder}:ch{ch}")
            groups.append(meta["patient"])
    X = np.vstack(X)
    y = np.asarray(y)
    ids = np.asarray(ids)
    groups = np.asarray(groups)

    # Impute any remaining NaN feature (e.g. an all-NaN channel) with its
    # column median so the SVM never sees NaN.
    col_med = np.nanmedian(X, axis=0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = col_med[cols]
    return X, y, ids, groups


def load_split(root=".", train_frac=0.30, seed=42):
    """Patient-level train/test split shared by the train and test scripts.

    Uses GroupShuffleSplit grouped by patient: ~train_frac of the *patients* go
    to training, the rest to testing, and no patient is split across both.
    Returns (X_tr, X_te, y_tr, y_te, id_tr, id_te, g_tr, g_te). Identical inputs
    + seed => identical partition in both scripts.
    """
    X, y, ids, groups = build_feature_table(root)
    gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        ids[train_idx], ids[test_idx],
        groups[train_idx], groups[test_idx],
    )

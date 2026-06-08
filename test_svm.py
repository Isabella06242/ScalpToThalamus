"""
Evaluate the trained thalamic-spread SVM on the held-out 70% of patients.

Reproduces the exact patient-level split used in training (same seed/frac stored
in the model bundle), predicts on the held-out patients' channels, and compares
to their true labels.

Metrics:
    Accuracy          - overall fraction correct
    Balanced accuracy - mean per-class recall (robust if classes are uneven)
    ROC-AUC           - ranking quality, threshold-independent
    Confusion matrix  - where errors fall
    Precision/Recall/F1 per class

Run (after train_svm.py):
    python test_svm.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from thalamus_data import load_split, CLASS_NAMES

MODEL_PATH = "thalamus_svm_model.joblib"
PRED_PATH = "test_predictions.csv"


def main():
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]

    # Reproduce the SAME partition used during training, then keep the test side.
    X_tr, X_te, y_tr, y_te, id_tr, id_te, g_tr, g_te = load_split(
        train_frac=bundle["train_frac"], seed=bundle["seed"]
    )
    n0, n1 = np.bincount(y_te, minlength=2)
    print(f"Test patients: {sorted(set(g_te))}")
    print(f"Test samples: {len(y_te)}  (no_spread={n0}, strong_spread={n1})")

    y_pred = clf.predict(X_te)
    y_score = clf.predict_proba(X_te)[:, 1]  # P(strong_spread)

    acc = accuracy_score(y_te, y_pred)
    bacc = balanced_accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_score)

    print(f"\nAccuracy:          {acc:.3f}")
    print(f"Balanced accuracy: {bacc:.3f}")
    print(f"ROC-AUC:           {auc:.3f}")

    print("\nConfusion matrix [rows=true, cols=pred]  (0=no_spread, 1=strong_spread):")
    print(confusion_matrix(y_te, y_pred))

    print("\nPer-class report:")
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))

    pd.DataFrame({
        "channel_id": id_te,
        "true": y_te,
        "pred": y_pred,
        "score_strong": y_score,
    }).to_csv(PRED_PATH, index=False)
    print(f"Saved per-channel predictions -> {PRED_PATH}")


if __name__ == "__main__":
    main()

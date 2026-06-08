"""
Train an SVM to classify a seizure EEG channel as no-spread vs strong-spread
to the thalamus, from its DDA coefficient features.

Uses 30% of the patients (patient-level split, all of a patient's channels on
one side) for training, then saves the fitted model + the split parameters so
test_svm.py evaluates on the matching held-out 70% of patients.

Run:
    python train_svm.py
"""
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from thalamus_data import load_split, FEATURE_NAMES, DATASETS

MODEL_PATH = "thalamus_svm_model.joblib"
TRAIN_FRAC = 0.30
SEED = 42


def main():
    X_tr, X_te, y_tr, y_te, id_tr, id_te, g_tr, g_te = load_split(
        train_frac=TRAIN_FRAC, seed=SEED
    )
    n0, n1 = np.bincount(y_tr, minlength=2)
    print(f"Train patients: {sorted(set(g_tr))}")
    print(f"Test  patients: {sorted(set(g_te))}")
    print(f"Training samples: {len(y_tr)}  (no_spread={n0}, strong_spread={n1})")
    print(f"Held-out test samples: {len(y_te)}")

    # StandardScaler is essential for an RBF SVM; class_weight balances classes.
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=SEED,
        )),
    ])
    clf.fit(X_tr, y_tr)

    print(f"Training accuracy: {clf.score(X_tr, y_tr):.3f}")

    joblib.dump(
        {
            "model": clf,
            "feature_names": FEATURE_NAMES,
            "train_frac": TRAIN_FRAC,
            "seed": SEED,
            "datasets": DATASETS,
        },
        MODEL_PATH,
    )
    print(f"Saved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()

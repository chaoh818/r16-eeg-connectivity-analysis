from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer


# ============================================================
# 1. Set project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

dataset_path = OUTPUT_DIR / "dataset_gu3.csv"

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Dataset path:", dataset_path)
print("Dataset exists?", dataset_path.exists())

if not dataset_path.exists():
    raise FileNotFoundError(f"Cannot find dataset file: {dataset_path}")


# ============================================================
# 2. Load full gu3 dataset
# ============================================================

dataset_gu3 = pd.read_csv(dataset_path)

print("\nFull dataset shape:", dataset_gu3.shape)
print("\nLanguage counts before filtering:")
print(dataset_gu3["lang"].value_counts(dropna=False))


# ============================================================
# 3. Filter to C vs S only
# ============================================================

dataset_cs = dataset_gu3[
    dataset_gu3["lang"].isin(["C", "S"])
].copy()

# Binary label:
# C = 1
# S = 0
dataset_cs["lang_binary_cs"] = dataset_cs["lang"].apply(
    lambda x: 1 if x == "C" else 0
)

print("\nC vs S dataset shape:", dataset_cs.shape)
print("\nLanguage counts after filtering:")
print(dataset_cs["lang"].value_counts(dropna=False))

print("\nEncoded label counts:")
print(dataset_cs["lang_binary_cs"].value_counts(dropna=False))


# ============================================================
# 4. Prepare X and y
# ============================================================

feature_cols = [
    col for col in dataset_cs.columns
    if col.startswith(("delta_", "theta_", "alpha_", "beta_", "highbeta_", "gamma_"))
]

if len(feature_cols) == 0:
    raise ValueError("No EEG feature columns found.")

X = dataset_cs[feature_cols].copy()
y = dataset_cs["lang_binary_cs"].astype(int)

print("\nNumber of EEG feature columns:", len(feature_cols))
print("X shape:", X.shape)


# ============================================================
# 5. Define metrics
# ============================================================

def specificity_score(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    Here:
    0 = S
    1 = C
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return np.nan

    return tn / (tn + fp)


scoring = {
    "accuracy": "accuracy",
    "auc": "roc_auc",
    "sensitivity": "recall",  # recall for positive class C
    "specificity": make_scorer(specificity_score),
}


# ============================================================
# 6. Define CV and models
# ============================================================

# C=30, S=21, total=51. 5-fold 比 10-fold 稳一点。
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42
    ),

    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=10000,
            random_state=42
        ))
    ])
}


# ============================================================
# 7. Run models
# ============================================================

results = []

for model_name, model in models.items():
    print(f"\nRunning {model_name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    row = {
        "task": "language_C_vs_S_gu3",
        "model": model_name,

        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_sd": scores["test_accuracy"].std(),

        "auc_mean": scores["test_auc"].mean(),
        "auc_sd": scores["test_auc"].std(),

        "sensitivity_mean": scores["test_sensitivity"].mean(),
        "sensitivity_sd": scores["test_sensitivity"].std(),

        "specificity_mean": scores["test_specificity"].mean(),
        "specificity_sd": scores["test_specificity"].std(),
    }

    results.append(row)

model_results_cs = pd.DataFrame(results)


# ============================================================
# 8. Save results and dataset
# ============================================================

cs_dataset_path = OUTPUT_DIR / "dataset_language_C_vs_S_gu3.csv"
cs_results_path = OUTPUT_DIR / "model_results_language_C_vs_S_gu3.csv"

dataset_cs.to_csv(cs_dataset_path, index=False)
model_results_cs.to_csv(cs_results_path, index=False)

print("\nModel results:")
print(model_results_cs)

print("\nSaved C vs S dataset to:")
print(cs_dataset_path)

print("\nSaved C vs S model results to:")
print(cs_results_path)
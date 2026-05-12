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

dataset_path = OUTPUT_DIR / "dataset_lang_gu3.csv"

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Dataset path:", dataset_path)
print("Dataset exists?", dataset_path.exists())

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Cannot find dataset file:\n{dataset_path}\n\n"
        "Please check whether dataset_lang_gu3.csv is saved in the outputs folder."
    )


# ============================================================
# 2. Load cleaned language classification dataset
# ============================================================

dataset_lang = pd.read_csv(dataset_path)

print("\nDataset loaded successfully.")
print("Dataset shape:", dataset_lang.shape)


# ============================================================
# 3. Prepare X and y
# ============================================================

feature_cols = [
    col for col in dataset_lang.columns
    if col.startswith(("delta_", "theta_", "alpha_", "beta_", "highbeta_", "gamma_"))
]

if len(feature_cols) == 0:
    raise ValueError("No EEG feature columns found. Please check feature column names.")

X = dataset_lang[feature_cols].copy()

# Binary label:
# C = 1
# Others = 0
y_label = dataset_lang["lang_binary"].copy()
y = (y_label == "C").astype(int)

print("\nNumber of EEG feature columns:", len(feature_cols))
print("X shape:", X.shape)

print("\nOriginal label counts:")
print(y_label.value_counts(dropna=False))

print("\nEncoded y counts:")
print(y.value_counts(dropna=False))


# ============================================================
# 4. Define metrics
# ============================================================

def specificity_score(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    Here:
    0 = Others
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
# 5. Define 10-fold CV and models
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
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
# 6. Run models
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
        "task": "lang_C_vs_Others_gu3",
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

model_results_gu3 = pd.DataFrame(results)


# ============================================================
# 7. Save results
# ============================================================

model_results_path = OUTPUT_DIR / "model_results_lang_gu3.csv"
model_results_gu3.to_csv(model_results_path, index=False)

print("\nModel results:")
print(model_results_gu3)

print("\nSaved model results to:")
print(model_results_path)
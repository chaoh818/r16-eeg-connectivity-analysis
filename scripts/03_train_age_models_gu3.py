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
    raise FileNotFoundError(
        f"Cannot find dataset file:\n{dataset_path}\n\n"
        "Please check whether dataset_gu3.csv is saved in the outputs folder."
    )


# ============================================================
# 2. Load full gu3 dataset
# ============================================================

dataset_gu3 = pd.read_csv(dataset_path)

print("\nDataset loaded successfully.")
print("Full dataset shape:", dataset_gu3.shape)

print("\nAge group counts before filtering:")
print(dataset_gu3["age_group"].value_counts(dropna=False))


# ============================================================
# 3. Filter to age classification dataset: 5-7 vs 8-12
# ============================================================

dataset_age = dataset_gu3[
    dataset_gu3["age_group"].isin(["5-7", "8-12"])
].copy()

# Binary label:
# 8-12 = 1
# 5-7 = 0
dataset_age["age_binary"] = dataset_age["age_group"].apply(
    lambda x: 1 if x == "8-12" else 0
)

print("\nAge dataset shape:", dataset_age.shape)

print("\nAge group counts after filtering:")
print(dataset_age["age_group"].value_counts(dropna=False))

print("\nEncoded age label counts:")
print(dataset_age["age_binary"].value_counts(dropna=False))


# ============================================================
# 4. Prepare X and y
# ============================================================

feature_cols = [
    col for col in dataset_age.columns
    if col.startswith(("delta_", "theta_", "alpha_", "beta_", "highbeta_", "gamma_"))
]

if len(feature_cols) == 0:
    raise ValueError("No EEG feature columns found. Please check feature column names.")

X = dataset_age[feature_cols].copy()
y = dataset_age["age_binary"].astype(int)

print("\nNumber of EEG feature columns:", len(feature_cols))
print("X shape:", X.shape)


# ============================================================
# 5. Define metrics
# ============================================================

def specificity_score(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    Here:
    0 = 5-7
    1 = 8-12
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return np.nan

    return tn / (tn + fp)


scoring = {
    "accuracy": "accuracy",
    "auc": "roc_auc",
    "sensitivity": "recall",  # recall for positive class = 8-12
    "specificity": make_scorer(specificity_score),
}


# ============================================================
# 6. Define CV and models
# ============================================================

# 样本比语言任务更少，所以这里用 5-fold 更稳。
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
        "task": "age_5-7_vs_8-12_gu3",
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

model_results_age_gu3 = pd.DataFrame(results)


# ============================================================
# 8. Save results and age dataset
# ============================================================

age_dataset_path = OUTPUT_DIR / "dataset_age_5-7_vs_8-12_gu3.csv"
age_results_path = OUTPUT_DIR / "model_results_age_5-7_vs_8-12_gu3.csv"

dataset_age.to_csv(age_dataset_path, index=False)
model_results_age_gu3.to_csv(age_results_path, index=False)

print("\nModel results:")
print(model_results_age_gu3)

print("\nSaved age dataset to:")
print(age_dataset_path)

print("\nSaved age model results to:")
print(age_results_path)
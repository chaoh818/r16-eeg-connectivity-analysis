from pathlib import Path
import warnings

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
# 1. Set paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"

dataset_path = OUTPUT_DIR / "dataset_gu3.csv"

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Dataset path:", dataset_path)
print("Dataset exists?", dataset_path.exists())

if not dataset_path.exists():
    raise FileNotFoundError(f"Cannot find dataset file: {dataset_path}")


# ============================================================
# 2. Load dataset
# ============================================================

dataset_gu3 = pd.read_csv(dataset_path)

print("\nFull dataset shape:", dataset_gu3.shape)

print("\nLanguage counts:")
print(dataset_gu3["lang"].value_counts(dropna=False))

print("\nAge group counts:")
print(dataset_gu3["age_group"].value_counts(dropna=False))


# ============================================================
# 3. Define feature groups by frequency band
# ============================================================

band_prefixes = {
    "delta": "delta_",
    "theta": "theta_",
    "alpha": "alpha_",
    "beta": "beta_",
    "highbeta": "highbeta_",
    "gamma": "gamma_",
}

band_feature_cols = {}

for band, prefix in band_prefixes.items():
    cols = [col for col in dataset_gu3.columns if col.startswith(prefix)]
    band_feature_cols[band] = cols
    print(f"{band}: {len(cols)} features")

all_band_cols = []
for cols in band_feature_cols.values():
    all_band_cols.extend(cols)

band_feature_cols["all"] = all_band_cols

print("all:", len(all_band_cols), "features")


# ============================================================
# 4. Define metrics
# ============================================================

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return np.nan

    return tn / (tn + fp)


scoring = {
    "accuracy": "accuracy",
    "auc": "roc_auc",
    "sensitivity": "recall",
    "specificity": make_scorer(specificity_score),
}


# ============================================================
# 5. Define model templates
# ============================================================

def get_models():
    return {
        "SVM_linear": Pipeline([
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
                max_iter=20000,
                random_state=42
            ))
        ])
    }


# ============================================================
# 6. Helper function: run one binary task across bands
# ============================================================

def run_bandwise_binary_task(
    dataset,
    task_name,
    label_col,
    positive_label,
    negative_label,
    n_splits=5
):
    """
    dataset: filtered dataset containing only two classes
    label_col: column containing original class labels
    positive_label: encoded as 1
    negative_label: encoded as 0
    """

    task_df = dataset[
        dataset[label_col].isin([positive_label, negative_label])
    ].copy()

    task_df["binary_y"] = task_df[label_col].apply(
        lambda x: 1 if x == positive_label else 0
    )

    print("\n" + "=" * 80)
    print("Task:", task_name)
    print("Positive class:", positive_label)
    print("Negative class:", negative_label)
    print("Dataset shape:", task_df.shape)
    print("Class counts:")
    print(task_df[label_col].value_counts(dropna=False))
    print("=" * 80)

    if task_df["binary_y"].nunique() != 2:
        raise ValueError(f"{task_name}: only one class found after filtering.")

    min_class_count = task_df["binary_y"].value_counts().min()

    if min_class_count < n_splits:
        n_splits = int(min_class_count)

    if n_splits < 2:
        raise ValueError(f"{task_name}: not enough samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    results = []

    for band_name, feature_cols in band_feature_cols.items():
        X = task_df[feature_cols].copy()
        y = task_df["binary_y"].astype(int)

        print(f"\nRunning band: {band_name} | features: {len(feature_cols)}")

        models = get_models()

        for model_name, model in models.items():
            print(f"  Model: {model_name}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                scores = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False
                )

            row = {
                "task": task_name,
                "band": band_name,
                "model": model_name,
                "n_samples": len(task_df),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "n_splits": n_splits,
                "n_features": len(feature_cols),

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

    return pd.DataFrame(results)


# ============================================================
# 7. Run tasks
# ============================================================

all_results = []

# Task 1: Language C vs S
language_cs_results = run_bandwise_binary_task(
    dataset=dataset_gu3,
    task_name="language_C_vs_S_gu3",
    label_col="lang",
    positive_label="C",
    negative_label="S",
    n_splits=5
)

all_results.append(language_cs_results)


# Task 2: Age 8-12 vs 5-7
age_results = run_bandwise_binary_task(
    dataset=dataset_gu3,
    task_name="age_8-12_vs_5-7_gu3",
    label_col="age_group",
    positive_label="8-12",
    negative_label="5-7",
    n_splits=5
)

all_results.append(age_results)


bandwise_results = pd.concat(all_results, ignore_index=True)


# ============================================================
# 8. Save full results
# ============================================================

results_path = OUTPUT_DIR / "bandwise_results_gu3.csv"
bandwise_results.to_csv(results_path, index=False)

print("\n" + "=" * 80)
print("Saved full band-wise results to:")
print(results_path)
print("=" * 80)


# ============================================================
# 9. Save best results summary
# ============================================================

best_by_task = (
    bandwise_results
    .sort_values(["task", "auc_mean"], ascending=[True, False])
    .groupby("task")
    .head(10)
    .reset_index(drop=True)
)

best_path = OUTPUT_DIR / "bandwise_best_results_gu3.csv"
best_by_task.to_csv(best_path, index=False)

print("\nTop results by task:")
print(best_by_task[[
    "task",
    "band",
    "model",
    "n_samples",
    "n_features",
    "accuracy_mean",
    "auc_mean",
    "sensitivity_mean",
    "specificity_mean"
]])

print("\nSaved best result summary to:")
print(best_path)
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
# 1. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

conditions = ["gu1", "gu2", "gu3"]

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)


# ============================================================
# 2. Metrics
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
# 3. Models
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
# 4. Feature bands
# ============================================================

band_prefixes = {
    "delta": "delta_",
    "theta": "theta_",
    "alpha": "alpha_",
    "beta": "beta_",
    "highbeta": "highbeta_",
    "gamma": "gamma_",
}


def get_band_feature_cols(df):
    band_feature_cols = {}

    for band, prefix in band_prefixes.items():
        cols = [col for col in df.columns if col.startswith(prefix)]
        band_feature_cols[band] = cols

    all_cols = []
    for cols in band_feature_cols.values():
        all_cols.extend(cols)

    band_feature_cols["all"] = all_cols

    return band_feature_cols


# ============================================================
# 5. Run one task
# ============================================================

def run_binary_task(
    df,
    condition,
    task_name,
    label_col,
    positive_label,
    negative_label,
    n_splits=5
):
    task_df = df[df[label_col].isin([positive_label, negative_label])].copy()

    task_df["binary_y"] = task_df[label_col].apply(
        lambda x: 1 if x == positive_label else 0
    )

    print("\n" + "=" * 90)
    print("Condition:", condition)
    print("Task:", task_name)
    print("Positive:", positive_label)
    print("Negative:", negative_label)
    print("Task dataset shape:", task_df.shape)
    print("Class counts:")
    print(task_df[label_col].value_counts(dropna=False))
    print("=" * 90)

    if task_df["binary_y"].nunique() != 2:
        print("Skipped: only one class found.")
        return pd.DataFrame()

    min_class_count = task_df["binary_y"].value_counts().min()
    effective_splits = min(n_splits, int(min_class_count))

    if effective_splits < 2:
        print("Skipped: not enough samples for CV.")
        return pd.DataFrame()

    cv = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=42
    )

    band_feature_cols = get_band_feature_cols(task_df)
    results = []

    for band_name, feature_cols in band_feature_cols.items():
        X = task_df[feature_cols].copy()
        y = task_df["binary_y"].astype(int)

        print(f"\nRunning condition={condition}, task={task_name}, band={band_name}, features={len(feature_cols)}")

        for model_name, model in get_models().items():
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

            results.append({
                "condition": condition,
                "task": task_name,
                "band": band_name,
                "model": model_name,
                "n_samples": len(task_df),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "n_splits": effective_splits,
                "n_features": len(feature_cols),

                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_sd": scores["test_accuracy"].std(),

                "auc_mean": scores["test_auc"].mean(),
                "auc_sd": scores["test_auc"].std(),

                "sensitivity_mean": scores["test_sensitivity"].mean(),
                "sensitivity_sd": scores["test_sensitivity"].std(),

                "specificity_mean": scores["test_specificity"].mean(),
                "specificity_sd": scores["test_specificity"].std(),
            })

    return pd.DataFrame(results)


# ============================================================
# 6. Run all conditions and tasks
# ============================================================

all_results = []

for condition in conditions:
    dataset_path = OUTPUT_DIR / f"dataset_{condition}.csv"

    print("\n" + "#" * 90)
    print("Loading:", dataset_path)
    print("#" * 90)

    if not dataset_path.exists():
        print("Missing dataset:", dataset_path)
        continue

    df = pd.read_csv(dataset_path)

    print("Dataset shape:", df.shape)
    print("Language counts:")
    print(df["lang"].value_counts(dropna=False))
    print("Age group counts:")
    print(df["age_group"].value_counts(dropna=False))

    # Task 1: Language C vs S
    result_lang_cs = run_binary_task(
        df=df,
        condition=condition,
        task_name="language_C_vs_S",
        label_col="lang",
        positive_label="C",
        negative_label="S",
        n_splits=5
    )
    all_results.append(result_lang_cs)

    # Task 2: Language C vs Others
    df_lang = df[df["lang"].notna()].copy()
    df_lang["lang_binary_task"] = df_lang["lang"].apply(
        lambda x: "C" if x == "C" else "Others"
    )

    result_lang_c_others = run_binary_task(
        df=df_lang,
        condition=condition,
        task_name="language_C_vs_Others",
        label_col="lang_binary_task",
        positive_label="C",
        negative_label="Others",
        n_splits=5
    )
    all_results.append(result_lang_c_others)

    # Task 3: Age 8-12 vs 5-7
    result_age = run_binary_task(
        df=df,
        condition=condition,
        task_name="age_8-12_vs_5-7",
        label_col="age_group",
        positive_label="8-12",
        negative_label="5-7",
        n_splits=5
    )
    all_results.append(result_age)


results_all = pd.concat(
    [r for r in all_results if not r.empty],
    ignore_index=True
)


# ============================================================
# 7. Save outputs
# ============================================================

full_path = OUTPUT_DIR / "bandwise_results_all_conditions.csv"
results_all.to_csv(full_path, index=False)

print("\n" + "=" * 90)
print("Saved full results to:")
print(full_path)
print("=" * 90)

best_summary = (
    results_all
    .sort_values(["task", "condition", "auc_mean"], ascending=[True, True, False])
    .groupby(["task", "condition"])
    .head(5)
    .reset_index(drop=True)
)

best_path = OUTPUT_DIR / "bandwise_best_results_all_conditions.csv"
best_summary.to_csv(best_path, index=False)

print("\nTop 5 results per task and condition:")
print(best_summary[[
    "condition",
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

print("\nSaved best summary to:")
print(best_path)


# Best overall per task
best_overall = (
    results_all
    .sort_values(["task", "auc_mean"], ascending=[True, False])
    .groupby("task")
    .head(10)
    .reset_index(drop=True)
)

best_overall_path = OUTPUT_DIR / "bandwise_best_overall_by_task.csv"
best_overall.to_csv(best_overall_path, index=False)

print("\nBest overall results by task:")
print(best_overall[[
    "condition",
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

print("\nSaved best overall summary to:")
print(best_overall_path)
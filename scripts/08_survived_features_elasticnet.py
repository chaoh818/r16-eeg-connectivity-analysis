from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"

RANDOM_STATE = 42
N_SPLITS = 5
SURVIVAL_THRESHOLD = 4  # survived if selected in >= 4/5 folds

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("N_SPLITS:", N_SPLITS)
print("SURVIVAL_THRESHOLD:", SURVIVAL_THRESHOLD)


# ============================================================
# 2. Feature helpers
# ============================================================

def get_feature_cols(df, band):
    if band == "all":
        prefixes = ("delta_", "theta_", "alpha_", "beta_", "highbeta_", "gamma_")
        return [col for col in df.columns if col.startswith(prefixes)]

    prefix = f"{band}_"
    return [col for col in df.columns if col.startswith(prefix)]


def parse_feature_name(feature_name):
    """
    Example:
    alpha_ACtL_FR -> band=alpha, ch1=ACtL, ch2=FR
    """
    parts = feature_name.split("_")

    if len(parts) < 3:
        return {
            "feature_band": None,
            "channel_1": None,
            "channel_2": None,
            "channel_pair": None,
        }

    feature_band = parts[0]
    channel_1 = parts[1]
    channel_2 = parts[2]
    channel_pair = f"{channel_1}-{channel_2}"

    return {
        "feature_band": feature_band,
        "channel_1": channel_1,
        "channel_2": channel_2,
        "channel_pair": channel_pair,
    }


# ============================================================
# 3. Prepare task data
# ============================================================

def prepare_binary_task(condition, task_name, label_col, positive_label, negative_label, band):
    dataset_path = OUTPUT_DIR / f"dataset_{condition}.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if task_name == "language_C_vs_Others":
        df = df[df["lang"].notna()].copy()
        df["lang_binary_task"] = df["lang"].apply(
            lambda x: "Mandarin" if x == "Mandarin" else "Others"
        )
        label_col = "lang_binary_task"

    task_df = df[df[label_col].isin([positive_label, negative_label])].copy()

    task_df["binary_y"] = task_df[label_col].apply(
        lambda x: 1 if x == positive_label else 0
    )

    feature_cols = get_feature_cols(task_df, band)

    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found for band: {band}")

    X = task_df[feature_cols].copy()
    y = task_df["binary_y"].astype(int).to_numpy()

    print("\n" + "=" * 90)
    print("Task:", task_name)
    print("Condition:", condition)
    print("Band:", band)
    print("Positive label:", positive_label)
    print("Negative label:", negative_label)
    print("Dataset shape:", task_df.shape)
    print("X shape:", X.shape)
    print("Class counts:")
    print(task_df[label_col].value_counts(dropna=False))
    print("=" * 90)

    return X, y, task_df, feature_cols


# ============================================================
# 4. ElasticNet survived features
# ============================================================

def run_elasticnet_survived_features(task_config):
    condition = task_config["condition"]
    task_name = task_config["task_name"]
    label_col = task_config["label_col"]
    positive_label = task_config["positive_label"]
    negative_label = task_config["negative_label"]
    band = task_config["band"]

    X, y, task_df, feature_cols = prepare_binary_task(
        condition=condition,
        task_name=task_name,
        label_col=label_col,
        positive_label=positive_label,
        negative_label=negative_label,
        band=band
    )

    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    n_splits = min(N_SPLITS, int(min_class_count))

    if n_splits < 2:
        raise ValueError("Not enough samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    coef_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\nTraining fold {fold_idx}/{n_splits}...")

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                class_weight="balanced",
                max_iter=20000,
                random_state=RANDOM_STATE + fold_idx
            ))
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        coefs = model.named_steps["model"].coef_[0]

        fold_df = pd.DataFrame({
            "task": task_name,
            "condition": condition,
            "band": band,
            "model": "ElasticNet",
            "fold": fold_idx,
            "feature": feature_cols,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            "selected": coefs != 0,
        })

        coef_records.append(fold_df)

        print("Selected features in this fold:", int((coefs != 0).sum()))

    coef_all = pd.concat(coef_records, ignore_index=True)

    # Save all fold coefficients
    safe_task_name = f"{task_name}_{condition}_{band}_ElasticNet"
    all_coef_path = OUTPUT_DIR / f"elasticnet_all_fold_coefficients_{safe_task_name}.csv"
    coef_all.to_csv(all_coef_path, index=False)

    # Summarise by feature
    summary = (
        coef_all
        .groupby(["task", "condition", "band", "model", "feature"], as_index=False)
        .agg(
            survival_count=("selected", "sum"),
            mean_coefficient=("coefficient", "mean"),
            median_coefficient=("coefficient", "median"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
    )

    summary["n_folds"] = n_splits
    summary["survival_rate"] = summary["survival_count"] / n_splits

    # Direction based on mean coefficient
    # Positive coefficient means stronger association with positive class
    summary["direction"] = np.where(
        summary["mean_coefficient"] > 0,
        f"towards_{positive_label}",
        np.where(
            summary["mean_coefficient"] < 0,
            f"towards_{negative_label}",
            "zero"
        )
    )

    parsed = summary["feature"].apply(parse_feature_name).apply(pd.Series)
    summary = pd.concat([summary, parsed], axis=1)

    summary = summary.sort_values(
        by=["survival_count", "mean_abs_coefficient"],
        ascending=[False, False]
    ).reset_index(drop=True)

    survived = summary[summary["survival_count"] >= SURVIVAL_THRESHOLD].copy()

    summary_path = OUTPUT_DIR / f"elasticnet_feature_summary_{safe_task_name}.csv"
    survived_path = OUTPUT_DIR / f"survived_features_{safe_task_name}.csv"

    summary.to_csv(summary_path, index=False)
    survived.to_csv(survived_path, index=False)

    print("\nSaved all fold coefficients to:")
    print(all_coef_path)

    print("\nSaved full feature summary to:")
    print(summary_path)

    print("\nSaved survived features to:")
    print(survived_path)

    print("\nTop 20 features:")
    print(summary[[
        "feature",
        "channel_pair",
        "survival_count",
        "survival_rate",
        "mean_coefficient",
        "mean_abs_coefficient",
        "direction"
    ]].head(20))

    print("\nSurvived features:")
    if survived.empty:
        print("No features survived the threshold.")
    else:
        print(survived[[
            "feature",
            "channel_pair",
            "survival_count",
            "survival_rate",
            "mean_coefficient",
            "mean_abs_coefficient",
            "direction"
        ]])

    return {
        "task": task_name,
        "condition": condition,
        "band": band,
        "model": "ElasticNet",
        "n_samples": len(y),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "n_features": X.shape[1],
        "n_folds": n_splits,
        "survival_threshold": SURVIVAL_THRESHOLD,
        "n_survived_features": survived.shape[0],
        "summary_path": str(summary_path),
        "survived_path": str(survived_path),
    }


# ============================================================
# 5. Task configs
# ============================================================

task_configs = [
    {
        "task_name": "language_C_vs_Others",
        "condition": "gu2",
        "band": "alpha",
        "label_col": "lang_binary_task",
        "positive_label": "Mandarin",
        "negative_label": "Others",
    },
    {
        "task_name": "age_8-12_vs_5-7",
        "condition": "gu2",
        "band": "alpha",
        "label_col": "age_group",
        "positive_label": "8-12",
        "negative_label": "5-7",
    },
]


# ============================================================
# 6. Run all tasks
# ============================================================

task_summaries = []

for config in task_configs:
    task_summary = run_elasticnet_survived_features(config)
    task_summaries.append(task_summary)

task_summary_df = pd.DataFrame(task_summaries)

task_summary_path = OUTPUT_DIR / "survived_features_task_summary.csv"
task_summary_df.to_csv(task_summary_path, index=False)

print("\n" + "=" * 90)
print("All survived-feature analyses completed.")
print("Saved task summary to:")
print(task_summary_path)
print("=" * 90)

print("\nTask summary:")
print(task_summary_df)
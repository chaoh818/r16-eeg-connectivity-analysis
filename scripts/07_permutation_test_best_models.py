from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

N_PERMUTATIONS = 1000
RANDOM_STATE = 42

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("N_PERMUTATIONS:", N_PERMUTATIONS)


# ============================================================
# 2. Model definitions
# ============================================================

def get_model(model_name):
    if model_name == "SVM_linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])

    elif model_name == "ElasticNet":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                class_weight="balanced",
                max_iter=20000,
                random_state=RANDOM_STATE
            ))
        ])

    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ============================================================
# 3. Feature selection by band
# ============================================================

def get_feature_cols(df, band):
    if band == "all":
        prefixes = ("delta_", "theta_", "alpha_", "beta_", "highbeta_", "gamma_")
        return [col for col in df.columns if col.startswith(prefixes)]

    prefix = f"{band}_"
    return [col for col in df.columns if col.startswith(prefix)]


# ============================================================
# 4. Prepare binary task
# ============================================================

def prepare_binary_task(condition, task_name, label_col, positive_label, negative_label, band):
    dataset_path = OUTPUT_DIR / f"dataset_{condition}.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # For C vs Others, construct a temporary label column
    if task_name == "language_C_vs_Others":
        df = df[df["lang"].notna()].copy()
        df["lang_binary_task"] = df["lang"].apply(
            lambda x: "C" if x == "C" else "Others"
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

    return X, y, task_df


# ============================================================
# 5. Permutation test
# ============================================================

def run_permutation_test(task_config):
    condition = task_config["condition"]
    task_name = task_config["task_name"]
    label_col = task_config["label_col"]
    positive_label = task_config["positive_label"]
    negative_label = task_config["negative_label"]
    band = task_config["band"]
    model_name = task_config["model"]

    X, y, task_df = prepare_binary_task(
        condition=condition,
        task_name=task_name,
        label_col=label_col,
        positive_label=positive_label,
        negative_label=negative_label,
        band=band
    )

    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    n_splits = min(5, int(min_class_count))

    if n_splits < 2:
        raise ValueError("Not enough samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model = get_model(model_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        true_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="roc_auc"
        )

    true_auc = true_scores.mean()

    print(f"\nTrue CV AUC: {true_auc:.4f}")
    print("Starting permutations...")

    rng = np.random.default_rng(RANDOM_STATE)
    perm_aucs = []

    for i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            perm_scores = cross_val_score(
                model,
                X,
                y_perm,
                cv=cv,
                scoring="roc_auc"
            )

        perm_auc = perm_scores.mean()
        perm_aucs.append(perm_auc)

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{N_PERMUTATIONS} permutations")

    perm_aucs = np.array(perm_aucs)

    # Add-one correction: avoids p=0 with finite permutations
    p_value = (np.sum(perm_aucs >= true_auc) + 1) / (N_PERMUTATIONS + 1)

    result = {
        "task": task_name,
        "condition": condition,
        "band": band,
        "model": model_name,
        "n_samples": len(y),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "n_splits": n_splits,
        "n_features": X.shape[1],
        "true_auc": true_auc,
        "true_auc_sd": true_scores.std(),
        "permutation_mean_auc": perm_aucs.mean(),
        "permutation_sd_auc": perm_aucs.std(),
        "permutation_min_auc": perm_aucs.min(),
        "permutation_max_auc": perm_aucs.max(),
        "p_value": p_value,
        "n_permutations": N_PERMUTATIONS,
    }

    # Save permutation distribution for this task
    safe_task_name = f"{task_name}_{condition}_{band}_{model_name}"
    perm_dist_path = OUTPUT_DIR / f"permutation_distribution_{safe_task_name}.csv"

    pd.DataFrame({
        "permutation_index": np.arange(1, N_PERMUTATIONS + 1),
        "permutation_auc": perm_aucs
    }).to_csv(perm_dist_path, index=False)

    print("\nPermutation result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("Saved permutation distribution to:")
    print(perm_dist_path)

    return result


# ============================================================
# 6. Best model configs
# ============================================================

best_model_configs = [
    {
        "task_name": "age_8-12_vs_5-7",
        "condition": "gu2",
        "band": "alpha",
        "model": "ElasticNet",
        "label_col": "age_group",
        "positive_label": "8-12",
        "negative_label": "5-7",
    },
    {
        "task_name": "language_C_vs_Others",
        "condition": "gu2",
        "band": "alpha",
        "model": "ElasticNet",
        "label_col": "lang_binary_task",
        "positive_label": "C",
        "negative_label": "Others",
    },
    {
        "task_name": "language_C_vs_S",
        "condition": "gu3",
        "band": "delta",
        "model": "SVM_linear",
        "label_col": "lang",
        "positive_label": "C",
        "negative_label": "S",
    },
]


# ============================================================
# 7. Run all permutation tests
# ============================================================

all_results = []

for config in best_model_configs:
    result = run_permutation_test(config)
    all_results.append(result)

permutation_results = pd.DataFrame(all_results)

results_path = OUTPUT_DIR / "permutation_results_best_models.csv"
permutation_results.to_csv(results_path, index=False)

print("\n" + "=" * 90)
print("All permutation tests completed.")
print("Saved summary to:")
print(results_path)
print("=" * 90)

print("\nSummary:")
print(permutation_results[[
    "task",
    "condition",
    "band",
    "model",
    "true_auc",
    "permutation_mean_auc",
    "permutation_sd_auc",
    "p_value",
    "n_permutations"
]])
#!/usr/bin/env python3
"""
06_compare_bands_all_conditions_multiclass.py
---------------------------------------------
Compares multiclass and binary classification performance for EEG connectivity features
using Nested Cross-Validation (Outer 5-Fold, Inner 3-Fold Grid Search) across:
- Three conditions: gu1, gu2, gu3
- Seven band settings: delta, theta, alpha, beta, highbeta, gamma, all
- Three models: Linear/RBF SVM, Random Forest, Elastic Net
- Five tasks: Age Group, Lookup Language, Lookup Age Group, 4-class Child Interaction, 8-class Child Interaction

Prevents data scaling leakage via sklearn Pipeline and tunes model hyperparameters on training folds.
Saves results to the 'outputs/' directory.
"""


import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ============================================================
# 1. Set paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR = OUTPUT_DIR
base_path = Path(DATA_DIR).expanduser()

CONDITIONS = ["gu1", "gu2", "gu3"]
BANDS = ["delta", "theta", "alpha", "beta", "highbeta", "gamma", "all"]

TASKS = {
    "Age_Group": "age_group",
    "Lang2_Group": "lang_group",
    "Lang4_Group": "lang",
    "Age_Child_Lang2": "interaction_child_lang2"
}

def get_features_for_band(df, band):
    all_bands = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
    if band == "all":
        return [col for col in df.columns if any(col.startswith(b + "_") for b in all_bands)]
    else:
        return [col for col in df.columns if col.startswith(band + "_")]

def prepare_task_data(df, task_key, target_col):
    """Prepares and filters dataset for the given task key and target column."""
    df_task = df.copy()
    
    # Target column mapping and fallback detection
    if target_col not in df_task.columns:
        if task_key == "Age_Group" and "age_group" in df_task.columns:
            target_col = "age_group"
        elif task_key == "Lang2_Group" and "lang_group" in df_task.columns:
            target_col = "lang_group"
        elif task_key == "Lang4_Group" and "lang" in df_task.columns:
            target_col = "lang"
        elif task_key in ["Age_Child_Lang2", "Age_Child_Lang4"]:
            if "age_group" in df_task.columns and "lang" in df_task.columns:
                # Filter population to 5-7 and 8-12 age brackets
                df_task = df_task[df_task["age_group"].isin(["5-7", "8-12"])]
                if task_key == "Age_Child_Lang2":
                    df_task["interaction_child_lang2"] = df_task["age_group"] + "_" + df_task["lang_group"].astype(str)
                    target_col = "interaction_child_lang2"
                else:
                    df_task["interaction_child_lang4"] = df_task["age_group"] + "_" + df_task["lang"].astype(str)
                    target_col = "interaction_child_lang4"
            else:
                return None, None, None
        else:
            return None, None, None
            
    df_task = df_task.dropna(subset=[target_col])
    y = df_task[target_col].values
    classes = sorted(list(np.unique(y)))
    
    if len(classes) < 2:
        return None, None, None
        
    return df_task, y, classes

def build_param_grids():
    """Defines pipelines and hyperparameter search grids for inner cross-validation."""
    pipelines = {
        "Elastic Net": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", max_iter=20000, random_state=42))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(class_weight="balanced", probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
        ])
    }
    
    param_grids = {
        "Elastic Net": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.2, 0.5, 0.8]
        },
        "SVM": {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto"]
        },
        "Random Forest": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__max_features": ["sqrt", "log2"]
        }
    }
    return pipelines, param_grids

def main():
    print("=== Step 2: Multi-Task Model Comparison with Nested CV & Grid Search ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pipelines, param_grids = build_param_grids()
    results = []
    
    for condition in CONDITIONS:
        filename = f"dataset_{condition}.csv"
        file_path = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping condition {condition}.")
            continue
            
        print(f"\nProcessing Condition: {condition} (from {file_path})")
        df_raw = pd.read_csv(file_path).drop_duplicates(subset=["participant_id"])
        
        for task_key, target_col in TASKS.items():
            df_task, y, classes = prepare_task_data(df_raw, task_key, target_col)
            if df_task is None:
                print(f"  [Skipping Task: {task_key}] Target column '{target_col}' not available or insufficient classes.")
                continue
                
            n_classes = len(classes)
            is_binary = (n_classes == 2)
            scoring_metric = "roc_auc" if is_binary else "roc_auc_ovr"
            
            print(f"\n  Evaluating Task: {task_key} ({n_classes} classes: {classes})")
            
            for band in BANDS:
                feature_cols = get_features_for_band(df_task, band)
                if not feature_cols:
                    print(f"    Warning: No features found for band '{band}'. Skipping.")
                    continue
                    
                X = df_task[feature_cols].values
                
                # Dynamic fold count based on class distribution
                class_counts = pd.Series(y).value_counts()
                min_samples = class_counts.min()
                
                if min_samples < 2:
                    print(f"    Skipping {task_key} on band {band}: a class has fewer than 2 samples ({min_samples}).")
                    continue
                    
                outer_splits = min(5, min_samples)
                inner_splits = min(3, min_samples - 1) if min_samples > 1 else 2
                
                if outer_splits < 2 or inner_splits < 2:
                    continue
                    
                outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
                
                for model_name, pipeline in pipelines.items():
                    param_grid = param_grids[model_name]
                    
                    outer_accs, outer_aucs = [], []
                    outer_sens, outer_specs = [], []
                    best_params_per_fold = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
                        grid_search = GridSearchCV(
                            estimator=pipeline,
                            param_grid=param_grid,
                            scoring=scoring_metric,
                            cv=inner_cv,
                            n_jobs=-1
                        )
                        
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        best_params_per_fold.append(str(grid_search.best_params_))
                        
                        y_pred = best_model.predict(X_test)
                        y_proba = best_model.predict_proba(X_test)
                        
                        outer_accs.append(accuracy_score(y_test, y_pred))
                        
                        if is_binary:
                            auc = roc_auc_score(y_test, y_proba[:, 1], labels=classes)
                        else:
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro', labels=classes)
                        outer_aucs.append(auc)
                        
                        # Confusion matrix for Sensitivity and Specificity
                        cm = confusion_matrix(y_test, y_pred, labels=classes)
                        sens_list, spec_list = [], []
                        for i in range(n_classes):
                            tp = cm[i, i]
                            fn = sum(cm[i, :]) - tp
                            fp = sum(cm[:, i]) - tp
                            tn = sum(sum(cm)) - tp - fn - fp
                            
                            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                            
                            sens_list.append(sens)
                            spec_list.append(spec)
                            
                        outer_sens.append(np.mean(sens_list))
                        outer_specs.append(np.mean(spec_list))
                        
                    results.append({
                        "Task": task_key,
                        "Condition": condition,
                        "Band": band,
                        "Model": model_name,
                        "AUC_mean": np.mean(outer_aucs),
                        "AUC_std": np.std(outer_aucs),
                        "Accuracy_mean": np.mean(outer_accs),
                        "Accuracy_std": np.std(outer_accs),
                        "Sensitivity_mean": np.mean(outer_sens),
                        "Sensitivity_std": np.std(outer_sens),
                        "Specificity_mean": np.mean(outer_specs),
                        "Specificity_std": np.std(outer_specs),
                        "Optimal_Params_Fold_List": "; ".join(best_params_per_fold)
                    })
                    
                    print(f"    - Band: {band:10} | Model: {model_name:13} | Nested AUC = {np.mean(outer_aucs):.4f} | Acc = {np.mean(outer_accs):.4f}")

    # Save raw band-wise results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "multiclass_results_all_conditions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved full results to: '{output_path}'")
    
    if len(results_df) > 0:
        best_overall = results_df.sort_values(by="AUC_mean", ascending=False).groupby(["Task"]).head(5)
        best_overall_path = os.path.join(OUTPUT_DIR, "multiclass_best_overall_by_task.csv")
        best_overall.to_csv(best_overall_path, index=False)
        print(f"✓ Saved top overall models by task to: '{best_overall_path}'")
        
        best_by_cond = results_df.sort_values("AUC_mean", ascending=False).groupby(["Condition", "Task"]).first().reset_index()
        best_by_cond_path = os.path.join(OUTPUT_DIR, "multiclass_best_results_all_conditions.csv")
        best_by_cond.to_csv(best_by_cond_path, index=False)
        print(f"✓ Saved best model per condition and task to: '{best_by_cond_path}'")

if __name__ == "__main__":
    main()
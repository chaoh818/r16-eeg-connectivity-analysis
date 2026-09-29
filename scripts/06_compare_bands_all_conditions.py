#!/usr/bin/env python3
"""
06_compare_bands_all_conditions.py
---------------------------------------------
Compares multiclass and binary classification performance for EEG datasets
combining Power Spectral Density (PSD) and Functional Connectivity (FC) features.

Evaluates across:
- Three stimulus conditions: gu1, gu2, gu3
- Seven frequency bands: delta, theta, alpha, beta, highbeta, gamma, all
- Three feature representations: FC (Coherence), PSD (Power), PSD_FC (Combined)
- Multiple classification tasks: eg. Age_Group, Age_Child_Language
- Three classifiers: Linear SVM, Random Forest, Elastic Net

Saves results to the 'outputs/' directory.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import argparse

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
ALL_BANDS = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
FEATURE_MODES = ["FC", "PSD", "PSD_FC"]

TASKS = {
    "Age_Group": "age_group",
    "Language": "lang",
    "Language_Tonal": "lang_group",
    "Age_Child_Language_Tonal": "interaction_child_lang2",
    "Age_Child_Language": "interaction_child_lang4"
}

time_start_arg = 0
time_end_arg = 1000

def classify_column_type(col):
    """
    Classifies a dataset column into 'PSD' (single channel spectral power)
    or 'FC' (pairwise channel connectivity / coherence).
    """
    col_str = str(col).lower()
    
    # Exclude metadata columns
    meta_keywords = ["participant", "file", "condition", "age", "lang", "lookup", "task", "interaction"]
    if any(m in col_str for m in meta_keywords):
        return "META"
        
    # Check explicitly labeled PSD columns
    if "psd" in col_str or "power" in col_str or "pow" in col_str:
        return "PSD"
        
    # Check band prefixes
    matching_band = None
    for band in ALL_BANDS:
        if col_str.startswith(band + "_") or col_str.startswith("psd_" + band):
            matching_band = band
            break
            
    if not matching_band:
        return "OTHER"
        
    # Split column tokens to distinguish single-channel (PSD) vs pair (FC)
    # e.g., 'delta_ACtL_ACrL' -> FC (3 parts), 'delta_ACtL' -> PSD (2 parts)
    parts = [p for p in col_str.split("_") if p != "psd" and p != "power"]
    
    # If it has 3+ parts or pairwise delimiter, it's Functional Connectivity (FC)
    if len(parts) >= 3 or "-" in col_str:
        return "FC"
    elif len(parts) == 2:
        return "PSD"
    else:
        return "FC" # Default connectivity feature

def get_features_by_band_and_mode(df, band, mode):
    """
    Extracts column names matching a specific frequency band and feature mode (FC, PSD, or PSD_FC).
    """
    target_bands = ALL_BANDS if band == "all" else [band]
    
    selected_cols = []
    for col in df.columns:
        col_type = classify_column_type(col)
        if col_type == "META":
            continue
            
        col_str = str(col).lower()
        # Check if column belongs to requested band(s)
        belongs_to_band = any(col_str.startswith(b + "_") or f"_{b}_" in col_str or col_str.startswith("psd_" + b) for b in target_bands)
        
        if not belongs_to_band:
            continue
            
        if mode == "PSD_FC":
            if col_type in ["PSD", "FC", "OTHER"]:
                selected_cols.append(col)
        elif mode == "FC":
            if col_type == "FC":
                selected_cols.append(col)
        elif mode == "PSD":
            if col_type == "PSD":
                selected_cols.append(col)
                
    # Fallback: if PSD alone is requested but columns aren't explicitly named PSD,
    # fallback to all band columns to ensure non-empty matrix
    if mode == "PSD" and not selected_cols:
        for col in df.columns:
            col_type = classify_column_type(col)
            if col_type != "META" and any(str(col).lower().startswith(b + "_") for b in target_bands):
                selected_cols.append(col)
                
    return selected_cols

def prepare_task_data(df, task_key, target_col):
    """
    Prepares dataset and target array for a specific analysis task.
    """
    df_task = df.copy()
    
    if target_col not in df_task.columns:
        if task_key == "Age_Group" and "age_group" in df_task.columns:
            target_col = "age_group"
        elif task_key == "Language_Tonal" and "lang_group" in df_task.columns:
            target_col = "lang_group"
        elif task_key == "Language" and "lang" in df_task.columns:
            target_col = "lang"
        elif task_key in ["Age_Child_Language", "Age_Child_Language_Tonal"]:
            if "age_group" in df_task.columns and "lang" in df_task.columns:
                df_task = df_task[df_task["age_group"].isin(["5-7", "8-12"])]
                if task_key == "Age_Child_Language_Tonal":
                    def map_lang2(l):
                        s = str(l).upper()
                        return "C" if any(k in s for k in ["C", "ZH", "MANDARIN"]) else "NC"
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

def main():
    print("=== Step 2: Multi-Task Model Comparison (Combining PSD and FC Features) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define machine learning classifiers
    models = {
        "Linear SVM": SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42),
        "Elastic Net": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, class_weight="balanced", max_iter=20000, random_state=42)
    }
    
    results = []
    
    for condition in CONDITIONS:
        filename = f"dataset_{condition}_{time_start_arg}_{time_end_arg}.csv"
        file_path = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(file_path):
            file_path = os.path.join("/workspace/scratch", filename)
            
        if not os.path.exists(file_path):
            print(f"Warning: File {filename} not found. Skipping condition {condition}.")
            continue
            
        print(f"\nProcessing Condition: {condition} (from {file_path})")
        df_raw = pd.read_csv(file_path)
        if "participant_id" in df_raw.columns:
            df_raw = df_raw.drop_duplicates(subset=["participant_id"])
            
        for task_key, target_col in TASKS.items():
            df_task, y, classes = prepare_task_data(df_raw, task_key, target_col)
            if df_task is None:
                continue
                
            n_classes = len(classes)
            is_binary = (n_classes == 2)
            
            print(f"\n  Evaluating Task: {task_key} ({n_classes} classes: {classes})")
            
            for mode in FEATURE_MODES:
                for band in ALL_BANDS:
                    feature_cols = get_features_by_band_and_mode(df_task, band, mode)
                    if not feature_cols:
                        continue
                        
                    X = df_task[feature_cols].values
                    n_features = X.shape[1]
                    
                    # Adaptive stratified cross-validation
                    class_counts = pd.Series(y).value_counts()
                    min_samples = class_counts.min()
                    n_splits = min(5, min_samples)
                    
                    if n_splits < 2:
                        continue
                        
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                    for model_name, model in models.items():
                        fold_accs, fold_aucs = [], []
                        fold_sens, fold_specs = [], []
                        
                        for train_idx, test_idx in skf.split(X, y):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            
                            # Standardize numeric features for SVM and Elastic Net
                            if model_name in ["Linear SVM", "Elastic Net"]:
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                            else:
                                X_train_scaled = X_train
                                X_test_scaled = X_test
                                
                            model.fit(X_train_scaled, y_train)
                            
                            y_pred = model.predict(X_test_scaled)
                            y_proba = model.predict_proba(X_test_scaled)
                            
                            fold_accs.append(accuracy_score(y_test, y_pred))
                            
                            # ROC AUC evaluation (binary vs OvR multiclass)
                            if is_binary:
                                auc = roc_auc_score(y_test, y_proba[:, 1], labels=classes)
                            else:
                                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro', labels=classes)
                            fold_aucs.append(auc)
                            
                            # Macro-averaged Sensitivity and Specificity
                            cm = confusion_matrix(y_test, y_pred, labels=classes)
                            s_list, sp_list = [], []
                            for i in range(n_classes):
                                tp = cm[i, i]
                                fn = sum(cm[i, :]) - tp
                                fp = sum(cm[:, i]) - tp
                                tn = sum(sum(cm)) - tp - fn - fp
                                
                                s_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                                sp_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                                
                            fold_sens.append(np.mean(s_list))
                            fold_specs.append(np.mean(sp_list))
                            
                        results.append({
                            "Task": task_key,
                            "Condition": condition,
                            "Feature_Mode": mode,
                            "Band": band,
                            "Model": model_name,
                            "N_Features": n_features,
                            "AUC_mean": np.mean(fold_aucs),
                            "AUC_std": np.std(fold_aucs),
                            "Accuracy_mean": np.mean(fold_accs),
                            "Accuracy_std": np.std(fold_accs),
                            "Sensitivity_mean": np.mean(fold_sens),
                            "Sensitivity_std": np.std(fold_sens),
                            "Specificity_mean": np.mean(fold_specs),
                            "Specificity_std": np.std(fold_specs)
                        })
                        
                        print(f"    - Mode: {mode:6} | Band: {band:8} | Model: {model_name:13} | N_Feat: {n_features:3} | AUC = {np.mean(fold_aucs):.4f} | Acc = {np.mean(fold_accs):.4f}")

    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, f"multiclass_results_all_conditions_{time_start_arg}_{time_end_arg}.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved full results to: '{output_path}'")
    
    if len(results_df) > 0:
        best_overall = results_df.sort_values(by="AUC_mean", ascending=False).groupby(["Task", "Condition"]).first().reset_index()
        best_overall_path = os.path.join(OUTPUT_DIR, f"multiclass_best_results_all_conditions_{time_start_arg}_{time_end_arg}.csv")
        best_overall.to_csv(best_overall_path, index=False)
        print(f"✓ Saved top model per task-condition to: '{best_overall_path}'")

if __name__ == "__main__":
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="Process EEG time windows.")

    # 2. Define the arguments and force them to be integers
    # Using -500 and 1000 as default fallbacks based on your standard epoch
    parser.add_argument("--start_time", type=int, default=-500, help="Start time in milliseconds")
    parser.add_argument("--end_time", type=int, default=1000, help="End time in milliseconds")

    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    # 4. Access the variables (they are already converted to integers)
    time_start_arg = args.start_time
    time_end_arg = args.end_time
    main()
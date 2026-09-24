#!/usr/bin/env python3
"""
06_compare_bands_all_conditions_multiclass.py
---------------------------------------------
Compares classification performance across multiple tasks including cross-group
interactions:
- Tasks:
  0) Age_Group (4-class: 5-7, 8-12, Teen, Adult)
  1) Age_Child_Lang2 (4-class interaction: 5-7_C, 5-7_NC, 8-12_C, 8-12_NC for child age subset [5-7, 8-12])
  2) Age_Child_Lang4 (8-class interaction: 5-7_C/E/S/A vs 8-12_C/E/S/A for child age subset [5-7, 8-12])
  3) Age_Edu_Lang2 (4-class interaction: 5-7_C, 5-7_NC, 8-12_C, 8-12_NC for child age subset [5-7, 8-12])
  4) Age_Edu_Lang4 (8-class interaction: 5-7_C/E/S/A vs 8-12_C/E/S/A for child age subset [5-7, 8-12])
- Three conditions: gu1, gu2, gu3
- Seven band settings: delta, theta, alpha, beta, highbeta, gamma, all
- Three models: Linear SVM, Random Forest, Elastic Net

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

def get_features_for_band(df, band):
    all_bands = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
    if band == "all":
        return [col for col in df.columns if any(col.startswith(b + "_") for b in all_bands)]
    else:
        return [col for col in df.columns if col.startswith(band + "_")]

def prepare_task_data(df, task_name):
    """
    Extracts feature matrix X, target labels y, and class names for a given task.
    Supports single targets and cross-group child-subset interaction targets.
    """
    # Identify available column names
    age_col = 'age_group'
    age_edu_col = 'age_group_edu'
    lang2_col = 'lang_group' 
    lang4_col = 'lang' 
    
    df_clean = df.copy()
    
    if task_name == "Age_Group":
        if not age_col or age_col not in df_clean.columns:
            return None, None, None
        df_clean = df_clean.dropna(subset=[age_col])
        y = df_clean[age_col].astype(str).values
        
    elif task_name == "Age_Child_Lang2":
        # Child subset (5-7 and 8-12) x 2 Language categories (e.g. C vs NC / Tonal vs NonTonal) -> 4 Classes
        if not age_col or not lang2_col or age_col not in df_clean.columns or lang2_col not in df_clean.columns:
            return None, None, None
        df_clean = df_clean[df_clean[age_col].astype(str).isin(['5-7', '8-12'])].dropna(subset=[age_col, lang2_col])
        if len(df_clean) == 0:
            return None, None, None
        y = (df_clean[age_col].astype(str) + "_" + df_clean[lang2_col].astype(str)).values
        
    elif task_name == "Age_Child_Lang4":
        # Child subset (5-7 and 8-12) x 4 Language categories (C, E, S, A) -> 8 Classes
        if not age_col or not lang4_col or age_col not in df_clean.columns or lang4_col not in df_clean.columns:
            return None, None, None
        df_clean = df_clean[df_clean[age_col].astype(str).isin(['5-7', '8-12'])].dropna(subset=[age_col, lang4_col])
        if len(df_clean) == 0:
            return None, None, None
        y = (df_clean[age_col].astype(str) + "_" + df_clean[lang4_col].astype(str)).values
        
    elif task_name == "Age_Edu_Lang2":
        # Education Child subset (Elementary and Secondary) x 2 Language categories (e.g. C vs NC / Tonal vs NonTonal) -> 4 Classes
        if not age_edu_col or not lang2_col or age_edu_col not in df_clean.columns or lang2_col not in df_clean.columns:
            return None, None, None
        df_clean = df_clean[df_clean[age_edu_col].astype(str).isin(['Elementary', 'Secondary'])].dropna(subset=[age_edu_col, lang2_col])
        if len(df_clean) == 0:
            return None, None, None
        y = (df_clean[age_edu_col].astype(str) + "_" + df_clean[lang2_col].astype(str)).values
        
    elif task_name == "Age_Edu_Lang4":
        # Education Child subset (Elementary and Secondary) x 4 Language categories (C, E, S, A) -> 8 Classes
        if not age_edu_col or not lang4_col or age_edu_col not in df_clean.columns or lang4_col not in df_clean.columns:
            return None, None, None
        df_clean = df_clean[df_clean[age_edu_col].astype(str).isin(['Elementary', 'Secondary'])].dropna(subset=[age_edu_col, lang4_col])
        if len(df_clean) == 0:
            return None, None, None
        y = (df_clean[age_edu_col].astype(str) + "_" + df_clean[lang4_col].astype(str)).values
        
    else:
        return None, None, None
        
    classes = sorted(list(np.unique(y)))
    return df_clean, y, classes

def main():
    print("=== Step 2: Multiclass and Cross-Group Model Comparison Across Conditions and Bands ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tasks = ["Age_Group", "Age_Child_Lang2", "Age_Child_Lang4"]
    
    models = {
        "Linear SVM": SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42),
        "Elastic Net": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, class_weight="balanced", max_iter=20000, random_state=42)
    }
    
    results = []
    
    for condition in CONDITIONS:
        filename = f"dataset_{condition}.csv"
        file_path = os.path.join(INPUT_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping condition {condition}.")
            continue
            
        print(f"\nProcessing Condition: {condition} (from {file_path})")
        df_raw = pd.read_csv(file_path)
        df_raw = df_raw.drop_duplicates(subset=["participant_id"])
        
        for task_name in tasks:
            df_task, y, classes = prepare_task_data(df_raw, task_name)
            
            if df_task is None or y is None or len(np.unique(y)) < 2:
                print(f"  [Skipping Task: {task_name}] Target or required columns not present/valid.")
                continue
                
            n_classes = len(classes)
            print(f"\n  Evaluating Task: {task_name} | Target Classes ({n_classes}): {classes} | Samples: {len(df_task)}")
            
            for band in BANDS:
                feature_cols = get_features_for_band(df_task, band)
                if not feature_cols:
                    continue
                    
                X = df_task[feature_cols].values
                
                for model_name, model in models.items():
                    # 5-Fold Stratified CV
                    # Use min splits if a class has fewer samples
                    class_counts = pd.Series(y).value_counts()
                    min_samples = class_counts.min()
                    n_splits = min(5, min_samples)
                    
                    if n_splits < 2:
                        print(f"    [Warning] Insufficient class samples for CV in {task_name}. Skipping.")
                        continue
                        
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                    fold_accs, fold_aucs = [], []
                    fold_sensitivities, fold_specificities = [], []
                    
                    for train_idx, test_idx in skf.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
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
                        
                        if n_classes == 2:
                            # Handle binary case
                            pos_label = classes[1]
                            pos_idx = list(model.classes_).index(pos_label)
                            auc = roc_auc_score(y_test == pos_label, y_proba[:, pos_idx])
                        else:
                            # Multiclass OvR macro AUC
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro', labels=classes)
                            
                        fold_aucs.append(auc)
                        
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
                            
                        fold_sensitivities.append(np.mean(sens_list))
                        fold_specificities.append(np.mean(spec_list))
                    
                    results.append({
                        "Task": task_name,
                        "Condition": condition,
                        "Band": band,
                        "Model": model_name,
                        "N_Classes": n_classes,
                        "AUC_mean": np.mean(fold_aucs),
                        "AUC_std": np.std(fold_aucs),
                        "Accuracy_mean": np.mean(fold_accs),
                        "Accuracy_std": np.std(fold_accs),
                        "Sensitivity_mean": np.mean(fold_sensitivities),
                        "Sensitivity_std": np.std(fold_sensitivities),
                        "Specificity_mean": np.mean(fold_specificities),
                        "Specificity_std": np.std(fold_specificities)
                    })
                    
                    print(f"    - Band: {band:10} | Model: {model_name:13} | AUC = {np.mean(fold_aucs):.4f} | Acc = {np.mean(fold_accs):.4f}")
                    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "multiclass_results_all_conditions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved full results to: '{output_path}'")
    
    if len(results_df) > 0:
        best_overall = results_df.sort_values(by="AUC_mean", ascending=False)
        best_overall_path = os.path.join(OUTPUT_DIR, "multiclass_best_overall_by_task.csv")
        best_overall.head(10).to_csv(best_overall_path, index=False)
        print(f"✓ Saved top 10 overall models to: '{best_overall_path}'")
        
        best_by_cond = results_df.sort_values("AUC_mean", ascending=False).groupby(["Task", "Condition"]).first().reset_index()
        best_by_cond_path = os.path.join(OUTPUT_DIR, "multiclass_best_results_all_conditions.csv")
        best_by_cond.to_csv(best_by_cond_path, index=False)
        print(f"✓ Saved best models per task-condition to: '{best_by_cond_path}'")

if __name__ == "__main__":
    main()

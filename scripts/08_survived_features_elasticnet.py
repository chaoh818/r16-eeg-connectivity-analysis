#!/usr/bin/env python3
"""
08_survived_features_elasticnet.py
----------------------------------
Identifies stable brain features (Power Spectral Density and Functional Connectivity
coherence channel-pairs) selected by Multiclass Elastic Net across cross-validation folds.

Dynamic Target Resolution:
- Age_Group: 'age_group'
- Language_Tonal: 'lang_tonal' (C vs. NC)
- Language: 'lang' (A, C, E, S)
- Age_Child_Language_Tonal: 4-class child interaction ('5-7_C', '5-7_NC', '8-12_C', '8-12_NC')
- Age_Child_Language: 6-class child interaction ('5-7_C', '5-7_E', '5-7_S', '8-12_C', '8-12_E', '8-12_S')

Window-Aware Dataset Resolution:
Matches files in formats like 'dataset_gu3_400_800.csv', 'dataset_gu2_0_300.csv', 'dataset_gu1_300_600.csv'.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR = OUTPUT_DIR
base_path = Path(DATA_DIR).expanduser()

def resolve_dataset_path(input_dir, condition, window_str):
    """Locates dataset CSV files formatted as 'dataset_{condition}_{start}_{end}.csv'."""
    window_clean = str(window_str).replace("-", "_").replace("ms", "").replace(" ", "").strip()
    window_raw = str(window_str).strip()
    
    candidates = [
        f"dataset_{condition}_{window_clean}.csv",
        f"dataset_{condition}_{window_raw}.csv",
        f"dataset_{condition}_{window_raw.replace('ms', '')}.csv",
        f"dataset_{condition}.csv"
    ]
    
    search_dirs = [input_dir, ".", "/workspace/scratch", "/workspace/knowledge", "/workspace/artifacts"]
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for cand in candidates:
            p = os.path.join(d, cand)
            if os.path.exists(p):
                return p
    return None

def resolve_target_and_data(df, task_name):
    """Dynamically creates target vector 'y' for each active task target."""
    df_task = df.copy().drop_duplicates(subset=["participant_id"])
    
    # 1. Age_Group
    if task_name == "Age_Group":
        if "age_group" in df_task.columns:
            df_task = df_task.dropna(subset=["age_group"])
            return df_task, df_task["age_group"].values
            
    # 2. Language_Tonal
    elif task_name == "Language_Tonal":
        if "lang_tonal" in df_task.columns:
            df_task = df_task.dropna(subset=["lang_tonal"])
            return df_task, df_task["lang_tonal"].values
        elif "lang" in df_task.columns:
            df_task = df_task.dropna(subset=["lang"])
            df_task["lang_tonal"] = df_task["lang"].apply(lambda x: "C" if str(x).upper() == "C" else "NC")
            return df_task, df_task["lang_tonal"].values
            
    # 3. Language
    elif task_name == "Language":
        if "lang" in df_task.columns:
            df_task = df_task.dropna(subset=["lang"])
            return df_task, df_task["lang"].values
            
    # 4. Age_Child_Language_Tonal
    elif task_name == "Age_Child_Language_Tonal":
        if "age_group" in df_task.columns and ("lang" in df_task.columns or "lang_tonal" in df_task.columns):
            df_task = df_task[df_task["age_group"].isin(["5-7", "8-12"])]
            if "lang_tonal" not in df_task.columns:
                df_task["lang_tonal"] = df_task["lang"].apply(lambda x: "C" if str(x).upper() == "C" else "NC")
            df_task["target"] = df_task["age_group"].astype(str) + "_" + df_task["lang_tonal"].astype(str)
            df_task = df_task.dropna(subset=["target"])
            return df_task, df_task["target"].values
            
    # 5. Age_Child_Language
    elif task_name == "Age_Child_Language":
        if "age_group" in df_task.columns and "lang" in df_task.columns:
            df_task = df_task[df_task["age_group"].isin(["5-7", "8-12"])]
            df_task["target"] = df_task["age_group"].astype(str) + "_" + df_task["lang"].astype(str)
            df_task = df_task.dropna(subset=["target"])
            return df_task, df_task["target"].values
            
    # Fallbacks for raw columns
    for col in [task_name, task_name.lower(), "target", "interaction_child_lang2", "interaction_child_lang4"]:
        if col in df_task.columns:
            df_task = df_task.dropna(subset=[col])
            return df_task, df_task[col].values
            
    return None, None

def get_features_for_band_and_mode(df, band, mode):
    """Filters feature columns based on frequency band and feature mode (PSD vs. FC)."""
    meta_cols = ["participant_id", "filename", "condition", "lang", "age_group", "file_name", 
                 "LookupLang", "LookupAgeGroup", "lang_tonal", "target", "interaction_child_lang2", 
                 "interaction_child_lang4", "window", "start_time", "end_time"]
    candidate_cols = [c for c in df.columns if c not in meta_cols]
    
    # Filter by band
    if str(band).lower() != "all":
        band_cols = [c for c in candidate_cols if c.startswith(f"{band}_") or f"_{band}_" in c or c.endswith(f"_{band}") or f"psd_{band}" in c.lower()]
    else:
        band_cols = candidate_cols
        
    if not band_cols:
        band_cols = candidate_cols
        
    mode_str = str(mode).upper() if mode else "ALL"
    
    if mode_str == "PSD":
        psd_cols = [c for c in band_cols if "psd" in c.lower() or len(c.split("_")) <= 2]
        return psd_cols if psd_cols else band_cols
    elif mode_str == "FC":
        fc_cols = [c for c in band_cols if "psd" not in c.lower() and len(c.split("_")) >= 3]
        return fc_cols if fc_cols else band_cols
    else:  # PSD_FC or ALL
        return band_cols

def extract_channel_descriptor(col_name):
    """Extracts channel or channel-pair label from feature name."""
    parts = col_name.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}-{parts[2]}"
    elif len(parts) == 2:
        return parts[1]
    return col_name

def main():
    print("=== Step 4: Multi-Task Elastic Net Survived-Feature Analysis ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    best_results_path = os.path.join(OUTPUT_DIR, "multiclass_best_results_all_conditions.csv")
    if not os.path.exists(best_results_path):
        best_results_path = os.path.join(OUTPUT_DIR, "multiclass_best_overall_by_task.csv")
        
    if os.path.exists(best_results_path):
        best_df = pd.read_csv(best_results_path)
        print(f"Loaded peak configurations from: {best_results_path}")
    else:
        print("Warning: No precomputed best results file found. Using default peak baseline configurations.")
        default_configs = [
            {"Task": "Age_Group", "Condition": "gu2", "Window": "0-800ms", "Band": "beta", "Feature_Mode": "PSD", "Model": "Elastic Net"},
            {"Task": "Language_Tonal", "Condition": "gu2", "Window": "400-800ms", "Band": "delta", "Feature_Mode": "FC", "Model": "Elastic Net"},
            {"Task": "Language", "Condition": "gu2", "Window": "0-300ms", "Band": "all", "Feature_Mode": "PSD_FC", "Model": "Elastic Net"},
            {"Task": "Age_Child_Language_Tonal", "Condition": "gu1", "Window": "300-600ms", "Band": "delta", "Feature_Mode": "PSD", "Model": "Elastic Net"},
            {"Task": "Age_Child_Language", "Condition": "gu3", "Window": "300-600ms", "Band": "delta", "Feature_Mode": "PSD", "Model": "Elastic Net"}
        ]
        best_df = pd.DataFrame(default_configs)
        
    all_tasks_summaries = []
    
    for idx, row in best_df.iterrows():
        task = row.get("Task", "Age_Group")
        condition = row.get("Condition", "gu1")
        window_str = row.get("Window", "0-800ms")
        band = row.get("Band", "delta")
        feature_mode = row.get("Feature_Mode", row.get("Mode", "PSD_FC"))
        
        file_path = resolve_dataset_path(INPUT_DIR, condition, window_str)
        if not file_path or not os.path.exists(file_path):
            print(f"Warning: Dataset for Condition={condition}, Window={window_str} not found. Skipping Task={task}.")
            continue
            
        df_raw = pd.read_csv(file_path)
        df_task, y = resolve_target_and_data(df_raw, task)
        
        if df_task is None or y is None or len(np.unique(y)) < 2:
            print(f"Warning: Insufficient class labels for Task='{task}' in {file_path}. Skipping.")
            continue
            
        classes = sorted(list(np.unique(y)))
        n_classes = len(classes)
        
        feature_cols = get_features_for_band_and_mode(df_task, band, feature_mode)
        if not feature_cols:
            print(f"Skipping Task: {task} | Band: {band} | Mode: {feature_mode} (No feature columns)")
            continue
            
        X = df_task[feature_cols].values
        n_features = len(feature_cols)
        
        print(f"\nAnalyzing Survived Features: Task={task:25} | Cond={condition} | Window={window_str:10} | Band={band:6} | Mode={feature_mode}")
        print(f"  - Target Classes ({n_classes}): {classes} | Total Input Features={n_features}")
        
        class_counts = pd.Series(y).value_counts()
        min_samples = class_counts.min()
        n_splits = min(5, min_samples)
        
        if n_splits < 2:
            print(f"  - Skipping: min samples per class ({min_samples}) < 2 for cross-validation.")
            continue
            
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Configure Elastic Net Classifier
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=20000,
            random_state=42
        )
        
        # Matrix shape: (n_splits, n_classes if n_classes > 2 else 1, n_features)
        if n_classes > 2:
            fold_coefficients = np.zeros((n_splits, n_classes, n_features))
        else:
            fold_coefficients = np.zeros((n_splits, 1, n_features))
            
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model.fit(X_train_scaled, y_train)
            
            if n_classes > 2:
                fold_coefficients[fold_idx, :, :] = model.coef_
            else:
                fold_coefficients[fold_idx, 0, :] = model.coef_[0]
                
        # Evaluate feature survival per class
        if n_classes > 2:
            target_eval_classes = list(enumerate(classes))
        else:
            target_eval_classes = [(0, f"{classes[1]}_vs_{classes[0]}")]
            
        for c_idx, cls_label in target_eval_classes:
            cls_coefs = fold_coefficients[:, c_idx, :]  # Shape: (n_splits, n_features)
            
            selection_mask = (cls_coefs != 0)
            survival_counts = selection_mask.sum(axis=0)
            survival_rate = survival_counts / float(n_splits)
            
            summary_df = pd.DataFrame({
                "Task": task,
                "feature": feature_cols,
                "descriptor": [extract_channel_descriptor(col) for col in feature_cols],
                "condition": condition,
                "window": window_str,
                "band": band,
                "feature_mode": feature_mode,
                "target_class": cls_label,
                "survival_count": survival_counts,
                "survival_rate": survival_rate,
                "mean_coefficient": cls_coefs.mean(axis=0),
                "mean_abs_coefficient": np.abs(cls_coefs).mean(axis=0),
                "direction": np.where(cls_coefs.mean(axis=0) > 0, f"towards_{cls_label}", "towards_others")
            })
            
            # Filter survived features (survival rate >= 80%, i.e. >= 4 out of 5 folds)
            threshold_count = max(1, int(round(0.8 * n_splits)))
            survived_df = summary_df[summary_df["survival_count"] >= threshold_count].copy()
            survived_df = survived_df.sort_values(by="mean_abs_coefficient", ascending=False)
            
            print(f"  ✓ Target Class '{cls_label}': {len(survived_df)} survived features (>= {threshold_count}/{n_splits} folds)")
            
            cls_safe_name = str(cls_label).replace("/", "_").replace(" ", "_")
            individual_path = os.path.join(OUTPUT_DIR, f"multiclass_survived_features_{task}_{condition}_{cls_safe_name}.csv")
            survived_df.to_csv(individual_path, index=False)
            
            all_tasks_summaries.append(survived_df)
            
    if all_tasks_summaries:
        task_summary_df = pd.concat(all_tasks_summaries, ignore_index=True)
        task_summary_path = os.path.join(OUTPUT_DIR, "multiclass_survived_features_task_summary.csv")
        task_summary_df.to_csv(task_summary_path, index=False)
        print(f"\n✓ Saved consolidated task summary ({len(task_summary_df)} rows) to: '{task_summary_path}'")
    else:
        print("\nNo survived features extracted across specified models.")

if __name__ == "__main__":
    main()
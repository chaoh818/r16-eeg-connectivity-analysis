#!/usr/bin/env python3
"""
10_make_visualizations.py
-------------------------
Generates publication-quality visualizations summarizing results from the
multiclass and binary EEG connectivity pipeline.

Generated Figures (saved to outputs/figures/):
- 00_best_results_table.png: Visual summary table displaying ALL conditions per task, highlighting peak condition.
- 01_best_model_auc_summary.png: Grouped bar chart comparing ALL conditions per task, highlighting peak condition with stars.
- 02_bandwise_auc_trajectory.png: Coherence band performance curves for all tasks.
- 04a_survived_features_age_group.png: Top stable connectivity pathways for Age_Group.
- 04b_survived_features_language_tonal.png: Top stable connectivity pathways for Language_Tonal.
- 04c_survived_features_age_child_language_tonal.png: Top stable connectivity pathways for Age_Child_Language_Tonal.
- 05_time_window_scheme_comparison.png: Time window scheme comparison across tasks.
- 06_feature_mode_synergy.png: Feature mode synergy (PSD vs FC vs PSD_FC).
- 07_stimulus_condition_selectivity.png: Acoustic stimulus selectivity (gu1 vs gu2 vs gu3).
- 08_task_granularity_tradeoffs.png: Task granularity trade-offs on a single Y-axis with 5 task ticks.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR = OUTPUT_DIR
base_path = Path(DATA_DIR).expanduser()

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 2. Visual Palette Theme
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Calibri', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'legend.fontsize': 9
})

PALETTE = ["#2F5496", "#4472C4", "#8FAADC", "#A6A6A6", "#ED7D31"]

def find_input_file(filename):
    """Finds input CSV across potential execution directories."""
    candidates = [
        os.path.join(INPUT_DIR, filename),
        os.path.join(OUTPUT_DIR, filename),
        os.path.join("/workspace/scratch", filename),
        os.path.join("/workspace/artifacts", filename),
        os.path.join("outputs", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def make_best_results_table_image(best_results_csv):
    """Renders a visual summary table showing ALL conditions per task, highlighting the best condition per task."""
    print("Generating Figure 00: Best Results Table (All conditions, highlighting best)...")
    csv_path = find_input_file("multiclass_best_results_all_conditions.csv") if not os.path.exists(best_results_csv) else best_results_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: best results CSV not found. Skipping Figure 00.")
        return
    
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
        
    # Standardize column selection
    available_cols = list(df.columns)
    col_map = {
        "Task": "Task",
        "Condition": "Cond",
        "Window": "Window",
        "Band": "Band",
        "Feature_Mode": "Mode",
        "Mode": "Mode",
        "Model": "Model",
        "AUC_mean": "Mean AUC",
        "Accuracy_mean": "Mean Acc",
        "Sensitivity_mean": "Sens",
        "Specificity_mean": "Spec"
    }
    
    selected_cols = [c for c in ["Task", "Condition", "Window", "Band", "Feature_Mode", "Mode", "Model", "AUC_mean", "Accuracy_mean", "Sensitivity_mean", "Specificity_mean"] if c in available_cols]
    
    df_disp = df[selected_cols].copy()
    
    # Identify best condition per Task (max AUC_mean)
    best_idx_per_task = set()
    if "Task" in df.columns and "AUC_mean" in df.columns:
        best_idx_per_task = set(df.groupby("Task")["AUC_mean"].idxmax())
        
    # Format and round columns
    for col in df_disp.columns:
        if "AUC" in col or "Acc" in col or "Sens" in col or "Spec" in col or "mean" in col:
            df_disp[col] = df_disp[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
            
    # Add star indicator to best row's Task/Cond
    df_render = df_disp.copy()
    df_render = df_render.rename(columns=col_map)
    
    # Render matplotlib table
    n_rows = len(df_render)
    fig_height = max(4.0, 0.45 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_height), dpi=200)
    ax.axis('off')
    
    table = ax.table(
        cellText=df_render.values,
        colLabels=df_render.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.35)
    
    # Apply styling & highlights
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2F5496')
        else:
            data_row_idx = row_idx - 1
            is_best_row = (df.index[data_row_idx] in best_idx_per_task)
            
            if is_best_row:
                # Highlight winning condition row
                cell.set_facecolor('#D9E1F2')  # Soft blue highlight
                cell.set_text_props(weight='bold', color='#002060')
            else:
                cell.set_facecolor('#F9FAFB' if data_row_idx % 2 == 1 else '#FFFFFF')
                cell.set_text_props(color='#333333')
                
            if col_idx == 0:
                cell.set_text_props(weight='bold')
                
    plt.title("EEG Coherence Pipeline - Classification Results Across All Conditions\n(★ Highlighted = Peak Condition per Task)", fontsize=12, pad=18, weight='bold', color='#1F497D')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "00_best_results_table.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    if os.path.exists("/workspace/out"):
        plt.savefig(os.path.join("/workspace/out", "00_best_results_table.png"), bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  ✓ Saved Figure 00 to: '{out_path}'")


def make_best_model_auc_summary(best_results_csv):
    """Generates a grouped bar chart showing ALL conditions per task, highlighting the best condition per task."""
    print("Generating Figure 01: Best Model AUC Summary (All conditions, highlighting best)...")
    csv_path = find_input_file("multiclass_best_results_all_conditions.csv") if not os.path.exists(best_results_csv) else best_results_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: best results CSV not found. Skipping Figure 01.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
        
    plt.figure(figsize=(9.5, 5.0), dpi=150)
    ax = sns.barplot(
        data=df,
        x="Task",
        y="AUC_mean",
        hue="Condition",
        palette=PALETTE[:3],
        edgecolor=".2"
    )
    
    # Identify max AUC per Task to highlight
    best_per_task = df.groupby("Task")["AUC_mean"].max().to_dict()
    
    # Label height on bars and highlight peak condition with star ★
    for container in ax.containers:
        for idx, bar in enumerate(container):
            height = bar.get_height()
            if np.isnan(height) or height == 0:
                continue
                
            task_name = ax.get_xticklabels()[idx].get_text()
            max_auc_for_task = best_per_task.get(task_name, None)
            
            is_peak = (max_auc_for_task is not None and abs(height - max_auc_for_task) < 1e-4)
            
            if is_peak:
                ax.annotate(
                    f"★ {height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8.5,
                    weight='bold',
                    color='#002060'
                )
                bar.set_edgecolor('#002060')
                bar.set_linewidth(1.8)
            else:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=7.5,
                    color='#555553'
                )
        
    plt.axhline(0.50, color='red', linestyle='--', linewidth=1, label="Chance Level (Binary 0.50)")
    plt.axhline(0.25, color='gray', linestyle=':', linewidth=1, label="Chance Level (4-Class 0.25)")
    
    plt.ylim(0, 0.90)
    plt.ylabel("Cross-Validated Mean Macro AUC", weight='bold')
    plt.xlabel("Target Variables", weight='bold')
    plt.title("Predictive Performance (AUC) Across All Stimulus Conditions\n(★ Indicates Peak Condition per Task)", fontsize=12, pad=15, weight='bold', color='#1F497D')
    plt.legend(title="Stimulus Condition", loc="lower left", frameon=True)
    plt.xticks(rotation=12, ha='right')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "01_best_model_auc_summary.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    if os.path.exists("/workspace/out"):
        plt.savefig(os.path.join("/workspace/out", "01_best_model_auc_summary.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 01 to: '{out_path}'")


def make_bandwise_auc_trajectory(results_all_csv):
    """Plots line charts showing which frequency bands drive accuracy."""
    print("Generating Figure 02: Band-Wise AUC Trajectories...")
    csv_path = find_input_file("multiclass_results_all_conditions.csv") if not os.path.exists(results_all_csv) else results_all_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: full band-wise results CSV not found. Skipping Figure 02.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
        
    df_best = df.groupby(["Task", "Condition", "Band"])["AUC_mean"].max().reset_index()
    
    g = sns.FacetGrid(
        df_best,
        col="Task",
        hue="Condition",
        palette=PALETTE[:3],
        height=3.8,
        aspect=1.1,
        sharey=False,
        col_wrap=3
    )
    
    def _plot_lines(x, y, **kwargs):
        data = kwargs.pop('data')
        color_val = kwargs.pop('color', None)
        sns.lineplot(data=data, x=x, y=y, marker='o', markersize=6, linewidth=1.5, color=color_val, **kwargs)
        
    g.map_dataframe(_plot_lines, x="Band", y="AUC_mean")
    
    for ax in g.axes.flat:
        ax.axhline(0.50, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.set_xlabel("Physiological Frequency Band", weight='bold')
        ax.set_ylabel("Peak Mean AUC", weight='bold')
        
    g.add_legend(title="Stimulus")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Frequency-Band Coherence Profiles across EEG Auditory Tasks", fontsize=13, weight='bold', color='#1F497D')
    
    out_path = os.path.join(FIGURES_DIR, "02_bandwise_auc_trajectory.png")
    g.savefig(out_path, bbox_inches='tight', dpi=150)
    if os.path.exists("/workspace/out"):
        g.savefig(os.path.join("/workspace/out", "02_bandwise_auc_trajectory.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 02 to: '{out_path}'")


def _plot_single_task_survived_features(df, task_name, figure_id, title_suffix, filename):
    """Helper to draw top selected channel-pair pathways for a single specific task."""
    if len(df) == 0:
        return
        
    df_task = df[df["Task"].astype(str).str.contains(task_name, case=False, na=False)].copy() if "Task" in df.columns else df.copy()
    if len(df_task) == 0:
        print(f"  Warning: No survived features found for task '{task_name}'. Skipping {figure_id}.")
        return
        
    df_top = df_task.sort_values(by="mean_abs_coefficient", ascending=False).head(12)
    
    feature_labels = []
    for _, row in df_top.iterrows():
        cond_str = str(row.get('condition', row.get('Condition', 'gu')))
        band_str = str(row.get('band', row.get('Band', 'all'))).upper()
        f_str = str(row.get('channel_pair', row.get('feature', row.get('feature_name', 'Feature'))))
        tgt_str = str(row.get('target_class', 'Target'))
        feature_labels.append(f"({cond_str}) {band_str} {f_str}\n→ Target: {tgt_str}")
        
    df_top["Feature_Label"] = feature_labels
    
    plt.figure(figsize=(8.5, 4.5), dpi=150)
    colors = ["#4472C4" if x > 0 else "#ED7D31" for x in df_top["mean_coefficient"]]
    
    ax = sns.barplot(
        data=df_top,
        x="mean_coefficient",
        y="Feature_Label",
        palette=colors,
        hue="Feature_Label",
        legend=False,
        edgecolor=".2"
    )
    
    plt.axvline(0, color='gray', linestyle='-', linewidth=1)
    plt.xlabel("Mean Elastic Net Coefficient (L1 Regularization)", weight='bold')
    plt.ylabel("")
    plt.title(f"Top Stable Connectivity Pathways: {title_suffix}\n(Selected in ≥ 4/5 Cross-Validation Folds)", fontsize=11, pad=12, weight='bold', color='#1F497D')
    
    plt.figtext(
        0.5, -0.04, 
        "Blue = Positive predictor of target class | Orange = Predicts alternative classes",
        ha="center", fontsize=8, style="italic", bbox=dict(facecolor='#F9FAFB', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.4')
    )
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    if os.path.exists("/workspace/out"):
        plt.savefig(os.path.join("/workspace/out", filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved {figure_id} to: '{out_path}'")


def make_survived_features_age_group(survived_task_summary_csv):
    """Draws top selected features for Age_Group."""
    print("Generating Figure 04a: Survived Features for Age_Group...")
    csv_path = find_input_file("multiclass_survived_features_task_summary.csv") if not os.path.exists(survived_task_summary_csv) else survived_task_summary_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: survived features summary CSV not found. Skipping 04a.")
        return
    df = pd.read_csv(csv_path)
    _plot_single_task_survived_features(df, "Age_Group", "Figure 04a", "Lifespan Age Maturation (Age_Group)", "04a_survived_features_age_group.png")


def make_survived_features_language_tonal(survived_task_summary_csv):
    """Draws top selected features for Language_Tonal."""
    print("Generating Figure 04b: Survived Features for Language_Tonal...")
    csv_path = find_input_file("multiclass_survived_features_task_summary.csv") if not os.path.exists(survived_task_summary_csv) else survived_task_summary_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: survived features summary CSV not found. Skipping 04b.")
        return
    df = pd.read_csv(csv_path)
    _plot_single_task_survived_features(df, "Language_Tonal", "Figure 04b", "Binary Tonal Contrast (Language_Tonal)", "04b_survived_features_language_tonal.png")


def make_survived_features_age_child_language_tonal(survived_task_summary_csv):
    """Draws top selected features for Age_Child_Language_Tonal."""
    print("Generating Figure 04c: Survived Features for Age_Child_Language_Tonal...")
    csv_path = find_input_file("multiclass_survived_features_task_summary.csv") if not os.path.exists(survived_task_summary_csv) else survived_task_summary_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: survived features summary CSV not found. Skipping 04c.")
        return
    df = pd.read_csv(csv_path)
    _plot_single_task_survived_features(df, "Age_Child_Language_Tonal", "Figure 04c", "Child Age x Tonal Interaction (Age_Child_Language_Tonal)", "04c_survived_features_age_child_language_tonal.png")


def make_time_window_scheme_comparison(results_all_csv):
    """Compares AUC performance across time window schemes."""
    print("Generating Figure 05: Time Window Scheme Comparison...")
    csv_path = find_input_file("multiclass_results_all_conditions.csv") if not os.path.exists(results_all_csv) else results_all_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: full results CSV not found. Skipping Figure 05.")
        return
        
    df = pd.read_csv(csv_path)
    if "Window" not in df.columns or len(df) == 0:
        return
        
    df_win = df.groupby(["Task", "Window"])["AUC_mean"].max().reset_index()
    
    plt.figure(figsize=(8.5, 4.5), dpi=150)
    ax = sns.barplot(data=df_win, x="Task", y="AUC_mean", hue="Window", palette="Blues_r", edgecolor=".2")
    
    plt.axhline(0.50, color='red', linestyle='--', linewidth=0.9, label="Binary Chance (0.50)")
    plt.ylim(0, 0.85)
    plt.ylabel("Peak Cross-Validated Macro AUC", weight='bold')
    plt.xlabel("Classification Task", weight='bold')
    plt.title("Predictive Capacity Comparison Across Time Epoch Windows", fontsize=11, pad=12, weight='bold', color='#1F497D')
    plt.legend(title="Time Window", loc="lower left")
    plt.xticks(rotation=12, ha='right')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "05_time_window_scheme_comparison.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    if os.path.exists("/workspace/out"):
        plt.savefig(os.path.join("/workspace/out", "05_time_window_scheme_comparison.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 05 to: '{out_path}'")


def make_feature_mode_synergy(results_all_csv):
    """Compares FC vs PSD vs PSD_FC feature extraction modes."""
    print("Generating Figure 06: Feature Mode Synergy...")
    csv_path = find_input_file("multiclass_results_all_conditions.csv") if not os.path.exists(results_all_csv) else results_all_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: full results CSV not found. Skipping Figure 06.")
        return
        
    df = pd.read_csv(csv_path)
    mode_col = "Feature_Mode" if "Feature_Mode" in df.columns else ("Mode" if "Mode" in df.columns else None)
    if not mode_col or len(df) == 0:
        return
        
    df_mode = df.groupby(["Task", mode_col])["AUC_mean"].max().reset_index()
    
    plt.figure(figsize=(8.5, 4.5), dpi=150)
    ax = sns.barplot(data=df_mode, x="Task", y="AUC_mean", hue=mode_col, palette="Set2", edgecolor=".2")
    
    plt.axhline(0.50, color='red', linestyle='--', linewidth=0.9)
    plt.ylim(0, 0.85)
    plt.ylabel("Peak Cross-Validated Macro AUC", weight='bold')
    plt.xlabel("Classification Task", weight='bold')
    plt.title("Feature Synergy: Power Spectral Density (PSD) vs Functional Connectivity (FC)", fontsize=11, pad=12, weight='bold', color='#1F497D')
    plt.legend(title="Feature Mode", loc="lower left")
    plt.xticks(rotation=12, ha='right')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "06_feature_mode_synergy.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    if os.path.exists("/workspace/out"):
        plt.savefig(os.path.join("/workspace/out", "06_feature_mode_synergy.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 06 to: '{out_path}'")


def make_stimulus_condition_selectivity(results_all_csv):
    """Compares gu1 vs gu2 vs gu3 stimulus selectivity across tasks."""
    print("Generating Figure 07: Acoustic Stimulus Selectivity...")
    csv_path = find_input_file("multiclass_results_all_conditions.csv") if not os.path.exists(results_all_csv) else results_all_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: full results CSV not found. Skipping Figure 07.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
        
    df_cond = df.groupby(["Task", "Condition"])["AUC_mean"].max().reset_index()
    
    plt.figure(figsize=(8.5, 4.5), dpi=150)
    ax = sns.barplot(data=df_cond, x="Task", y="AUC_mean", hue="Condition", palette=PALETTE[:3], edgecolor=".2")
    
    plt.axhline(0.50, color='red', linestyle='--', linewidth=0.9)
    plt.ylim(0, 0.85)
    plt.ylabel("Peak Cross-Validated Macro AUC", weight='bold')
    plt.xlabel("Classification Task", weight='bold')
    plt.title("Acoustic Stimulus Selectivity Across Tasks (gu1 vs gu2 vs gu3)", fontsize=11, pad=12, weight='bold', color='#1F497D')
    plt.legend(title="Stimulus Condition", loc="lower left")
    plt.xticks(rotation=12, ha='right')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "07_stimulus_condition_selectivity.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    if os.path.exists("/workspace/out"):
        plt.savefig(os.path.join("/workspace/out", "07_stimulus_condition_selectivity.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 07 to: '{out_path}'")


def make_task_granularity_tradeoffs(best_results_csv):
    """Plots task granularity vs model AUC and chance baseline on a single Y-axis with 5 x-ticks."""
    print("Generating Figure 08: Task Granularity Trade-offs (Single Y-Axis, 5 Task Ticks)...")
    csv_path = find_input_file("multiclass_best_results_all_conditions.csv") if not os.path.exists(best_results_csv) else best_results_csv
    if not csv_path or not os.path.exists(csv_path):
        print("  Warning: best results CSV not found. Skipping Figure 08.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
        
    df_peak = df.groupby("Task")["AUC_mean"].max().reset_index()
    
    task_order = [
        ("Language_Tonal", "Language Tonal\n(2 Classes)", 2, 0.500),
        ("Age_Group", "Age Group\n(4 Classes)", 4, 0.250),
        ("Language", "Language\n(4 Classes)", 4, 0.250),
        ("Age_Child_Language_Tonal", "Child Tonal Interaction\n(4 Classes)", 4, 0.250),
        ("Age_Child_Language", "Child Interaction\n(6 Classes)", 6, 0.167)
    ]
    
    tasks_df = pd.DataFrame(task_order, columns=["Task", "Display_Name", "N_Classes", "Chance_Baseline"])
    df_merged = pd.merge(tasks_df, df_peak, on="Task", how="left").fillna(0.500)
    
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    
    x_indices = np.arange(len(df_merged))
    
    ax.plot(x_indices, df_merged["AUC_mean"], marker='o', markersize=8, linewidth=2.2, color='#2F5496', label="Peak Model Macro AUC")
    ax.plot(x_indices, df_merged["Chance_Baseline"], marker='s', markersize=6, linewidth=1.8, linestyle='--', color='#ED7D31', label="Theoretical Chance Baseline")
    
    for i, row in df_merged.iterrows():
        ax.annotate(f"{row['AUC_mean']:.3f}", (i, row['AUC_mean']), textcoords="offset points", xytext=(0, 8), ha='center', weight='bold', color='#2F5496', fontsize=9)
        ax.annotate(f"{row['Chance_Baseline']:.3f}", (i, row['Chance_Baseline']), textcoords="offset points", xytext=(0, -14), ha='center', color='#ED7D31', fontsize=8)
        
    ax.set_ylim(0.10, 0.85)
    ax.set_ylabel("Macro OvR AUC / Accuracy Baseline", weight='bold', color='#1F497D')
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(df_merged["Display_Name"], fontsize=9, weight='bold')
    
    ax.set_title("Model Predictive Power vs. Chance Baseline Across Task Granularities", fontsize=11, pad=14, weight='bold', color='#1F497D')
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "08_task_granularity_tradeoffs.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    if os.path.exists("/workspace/out"):
        plt.savefig(os.path.join("/workspace/out", "08_task_granularity_tradeoffs.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 08 to: '{out_path}'")


def main():
    print("=== Step 6: Generating Full Visualization Suite (Script 10) ===")
    
    best_results_csv = os.path.join(INPUT_DIR, "multiclass_best_results_all_conditions.csv")
    results_all_csv = os.path.join(INPUT_DIR, "multiclass_results_all_conditions.csv")
    survived_task_summary_csv = os.path.join(INPUT_DIR, "multiclass_survived_features_task_summary.csv")
    
    make_best_results_table_image(best_results_csv)
    make_best_model_auc_summary(best_results_csv)
    make_bandwise_auc_trajectory(results_all_csv)
    make_survived_features_age_group(survived_task_summary_csv)
    make_survived_features_language_tonal(survived_task_summary_csv)
    make_survived_features_age_child_language_tonal(survived_task_summary_csv)
    make_time_window_scheme_comparison(results_all_csv)
    make_feature_mode_synergy(results_all_csv)
    make_stimulus_condition_selectivity(results_all_csv)
    make_task_granularity_tradeoffs(best_results_csv)
    
    print(f"\n✓ Successfully generated all updated pipeline figures under: '{FIGURES_DIR}/'")

if __name__ == "__main__":
    main()
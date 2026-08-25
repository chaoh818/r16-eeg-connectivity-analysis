from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"

summary_path = OUTPUT_DIR / "R16_EEG_analysis_summary.xlsx"

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("Summary Excel:", summary_path)


# ============================================================
# 2. Helper functions
# ============================================================

def read_csv_if_exists(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        print(f"Loaded: {filename}")
        return pd.read_csv(path)
    else:
        print(f"Missing: {filename}")
        return pd.DataFrame()


def clean_task_name(task):
    mapping = {
        "age_8-12_vs_5-7": "Age: 8–12 vs 5–7",
        "language_C_vs_Others": "Language: C vs Others",
        "language_C_vs_S": "Language: C vs S",
    }
    return mapping.get(task, task)


def clean_model_name(model):
    mapping = {
        "SVM_linear": "Linear SVM",
        "ElasticNet": "Elastic Net",
        "RandomForest": "Random Forest",
    }
    return mapping.get(model, model)


def clean_for_excel(df):
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).replace("nan", "")
    return df


def write_sheet(writer, df, sheet_name):
    df = clean_for_excel(df)

    if df.empty:
        df = pd.DataFrame({"Message": [f"No data found for {sheet_name}."]})

    df.to_excel(writer, sheet_name=sheet_name, index=False)


def add_clean_columns(df):
    if df.empty:
        return df

    df = df.copy()

    if "task" in df.columns:
        df["task_label"] = df["task"].apply(clean_task_name)

    if "model" in df.columns:
        df["model_label"] = df["model"].apply(clean_model_name)

    return df


# ============================================================
# 3. Load result files
# ============================================================

bandwise_all = read_csv_if_exists("bandwise_results_all_conditions.csv")
best_by_condition = read_csv_if_exists("bandwise_best_results_all_conditions.csv")
best_overall = read_csv_if_exists("bandwise_best_overall_by_task.csv")
permutation_results = read_csv_if_exists("permutation_results_best_models.csv")
survived_summary = read_csv_if_exists("survived_features_task_summary.csv")

survived_language = read_csv_if_exists(
    "survived_features_language_C_vs_Others_gu2_alpha_ElasticNet.csv"
)

survived_age = read_csv_if_exists(
    "survived_features_age_8-12_vs_5-7_gu2_alpha_ElasticNet.csv"
)

bandwise_all = add_clean_columns(bandwise_all)
best_by_condition = add_clean_columns(best_by_condition)
best_overall = add_clean_columns(best_overall)
permutation_results = add_clean_columns(permutation_results)
survived_summary = add_clean_columns(survived_summary)
survived_language = add_clean_columns(survived_language)
survived_age = add_clean_columns(survived_age)


# ============================================================
# 4. Dataset counts
# ============================================================

dataset_counts = []

for condition in ["gu1", "gu2", "gu3"]:
    dataset_file = OUTPUT_DIR / f"dataset_{condition}.csv"

    if dataset_file.exists():
        df = pd.read_csv(dataset_file)

        dataset_counts.append({
            "condition": condition,
            "n_total_rows": len(df),

            "n_with_lang": df["lang"].notna().sum() if "lang" in df.columns else None,
            "n_missing_lang_or_metadata": df["lang"].isna().sum() if "lang" in df.columns else None,

            "n_C": (df["lang"] == "C").sum() if "lang" in df.columns else None,
            "n_S": (df["lang"] == "S").sum() if "lang" in df.columns else None,
            "n_E": (df["lang"] == "E").sum() if "lang" in df.columns else None,
            "n_F": (df["lang"] == "F").sum() if "lang" in df.columns else None,
            "n_K": (df["lang"] == "K").sum() if "lang" in df.columns else None,

            "n_age_5_7": (df["age_group"] == "5-7").sum() if "age_group" in df.columns else None,
            "n_age_8_12": (df["age_group"] == "8-12").sum() if "age_group" in df.columns else None,
            "n_teens": (df["age_group"] == "Teens").sum() if "age_group" in df.columns else None,
            "n_age_other": (df["age_group"] == "Other").sum() if "age_group" in df.columns else None,
            "n_age_missing": df["age_group"].isna().sum() if "age_group" in df.columns else None,
        })

dataset_counts = pd.DataFrame(dataset_counts)


# ============================================================
# 5. Executive summary
# ============================================================

executive_rows = []

if not best_overall.empty:
    for task in best_overall["task"].unique():
        top = (
            best_overall[best_overall["task"] == task]
            .sort_values("auc_mean", ascending=False)
            .head(1)
        )

        if not top.empty:
            row = top.iloc[0]
            executive_rows.append({
                "result_type": "Best CV model",
                "task": clean_task_name(row.get("task", "")),
                "condition": row.get("condition", ""),
                "band": row.get("band", ""),
                "model": clean_model_name(row.get("model", "")),
                "n_samples": row.get("n_samples", ""),
                "n_features": row.get("n_features", ""),
                "accuracy_mean": row.get("accuracy_mean", ""),
                "auc_mean": row.get("auc_mean", ""),
                "sensitivity_mean": row.get("sensitivity_mean", ""),
                "specificity_mean": row.get("specificity_mean", ""),
                "p_value": "",
                "interpretation": "Best cross-validated AUC found in band-wise condition comparison."
            })

if not permutation_results.empty:
    for _, row in permutation_results.iterrows():
        executive_rows.append({
            "result_type": "Permutation test",
            "task": clean_task_name(row.get("task", "")),
            "condition": row.get("condition", ""),
            "band": row.get("band", ""),
            "model": clean_model_name(row.get("model", "")),
            "n_samples": row.get("n_samples", ""),
            "n_features": row.get("n_features", ""),
            "accuracy_mean": "",
            "auc_mean": row.get("true_auc", ""),
            "sensitivity_mean": "",
            "specificity_mean": "",
            "p_value": row.get("p_value", ""),
            "interpretation": "Observed AUC compared against 1000 random-label permutations."
        })

executive_summary = pd.DataFrame(executive_rows)


# ============================================================
# 6. Key findings
# ============================================================

key_findings = pd.DataFrame([
    {
        "finding_id": 1,
        "finding": "All-band combined features were weaker than selected frequency-band features.",
        "evidence": "Band-wise models showed stronger AUC values than the all-band baseline, especially for gu2 alpha-band models.",
        "interpretation": "Discriminative information appears frequency-specific rather than evenly distributed across all bands."
    },
    {
        "finding_id": 2,
        "finding": "gu2 alpha-band coherence produced the strongest C vs Others language result.",
        "evidence": "Language C vs Others: gu2 alpha Elastic Net achieved AUC ≈ 0.771 with permutation p ≈ 0.001.",
        "interpretation": "Alpha-band coherence under gu2 may carry useful information for distinguishing C participants from non-C participants."
    },
    {
        "finding_id": 3,
        "finding": "gu2 alpha-band coherence also produced the strongest age classification result.",
        "evidence": "Age 8–12 vs 5–7: gu2 alpha Elastic Net achieved AUC ≈ 0.754 with permutation p ≈ 0.006.",
        "interpretation": "The same condition-band combination appears informative for developmental age-group differences."
    },
    {
        "finding_id": 4,
        "finding": "The cleaner C vs S language task was best captured by gu3 delta-band features using a linear SVM.",
        "evidence": "Language C vs S: gu3 delta linear SVM achieved AUC ≈ 0.740 with permutation p ≈ 0.012.",
        "interpretation": "Different language contrasts may rely on different stimulus conditions and frequency bands."
    },
    {
        "finding_id": 5,
        "finding": "Elastic Net identified stable survived features for the two gu2 alpha tasks.",
        "evidence": "Language C vs Others had 16 survived features; Age 8–12 vs 5–7 had 14 survived features using a ≥4/5 fold threshold.",
        "interpretation": "These channel-pair features were repeatedly selected across cross-validation folds and may be useful for interpretation."
    },
])


# ============================================================
# 7. Best compact table
# ============================================================

best_compact = pd.DataFrame([
    {
        "task": "Age: 8–12 vs 5–7",
        "condition": "gu2",
        "band": "alpha",
        "model": "Elastic Net",
        "AUC": 0.7536,
        "permutation_p": 0.00599,
        "comment": "Best age classification result."
    },
    {
        "task": "Language: C vs Others",
        "condition": "gu2",
        "band": "alpha",
        "model": "Elastic Net",
        "AUC": 0.7714,
        "permutation_p": 0.000999,
        "comment": "Best overall language result."
    },
    {
        "task": "Language: C vs S",
        "condition": "gu3",
        "band": "delta",
        "model": "Linear SVM",
        "AUC": 0.7400,
        "permutation_p": 0.01199,
        "comment": "Best cleaner pairwise language result."
    },
])


# ============================================================
# 8. Write Excel workbook
# ============================================================

with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
    write_sheet(writer, key_findings, "00_Key_Findings")
    write_sheet(writer, best_compact, "01_Best_Results")
    write_sheet(writer, executive_summary, "02_Executive_Summary")
    write_sheet(writer, dataset_counts, "03_Dataset_Counts")
    write_sheet(writer, best_overall, "04_Best_Overall")
    write_sheet(writer, best_by_condition, "05_Best_By_Condition")
    write_sheet(writer, permutation_results, "06_Permutation_Tests")
    write_sheet(writer, survived_summary, "07_Survived_Summary")
    write_sheet(writer, survived_language, "08_Survived_Language")
    write_sheet(writer, survived_age, "09_Survived_Age")
    write_sheet(writer, bandwise_all, "10_All_Bandwise_Results")


# ============================================================
# 9. Format workbook
# ============================================================

wb = load_workbook(summary_path)

header_fill = PatternFill("solid", fgColor="1F4E78")
header_font = Font(color="FFFFFF", bold=True)
thin_border = Border(
    left=Side(style="thin", color="D9E2F3"),
    right=Side(style="thin", color="D9E2F3"),
    top=Side(style="thin", color="D9E2F3"),
    bottom=Side(style="thin", color="D9E2F3"),
)

for ws in wb.worksheets:
    ws.freeze_panes = "A2"

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = thin_border
            cell.alignment = Alignment(vertical="top", wrap_text=True)

            if isinstance(cell.value, float):
                cell.number_format = "0.0000"

    ws.auto_filter.ref = ws.dimensions

    for col_idx, col_cells in enumerate(ws.columns, start=1):
        max_len = 0
        col_letter = get_column_letter(col_idx)

        for cell in col_cells:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(value))

        width = min(max(max_len + 2, 10), 45)
        ws.column_dimensions[col_letter].width = width

    ws.row_dimensions[1].height = 28


# Highlight p-values
for sheet_name in ["01_Best_Results", "02_Executive_Summary", "06_Permutation_Tests"]:
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        p_cols = []

        for idx, cell in enumerate(ws[1], start=1):
            if cell.value in ["p_value", "permutation_p"]:
                p_cols.append(idx)

        for p_col in p_cols:
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=p_col)
                try:
                    p = float(cell.value)
                    if p < 0.05:
                        cell.fill = PatternFill("solid", fgColor="C6EFCE")
                        cell.font = Font(color="006100", bold=True)
                    else:
                        cell.fill = PatternFill("solid", fgColor="FFC7CE")
                        cell.font = Font(color="9C0006", bold=True)
                except Exception:
                    pass


# Highlight AUC values
for sheet_name in [
    "01_Best_Results",
    "02_Executive_Summary",
    "04_Best_Overall",
    "05_Best_By_Condition",
    "10_All_Bandwise_Results"
]:
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        auc_cols = []
        for idx, cell in enumerate(ws[1], start=1):
            if cell.value in ["AUC", "auc_mean", "true_auc"]:
                auc_cols.append(idx)

        for auc_col in auc_cols:
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=auc_col)
                try:
                    auc = float(cell.value)
                    if auc >= 0.75:
                        cell.fill = PatternFill("solid", fgColor="C6EFCE")
                    elif auc >= 0.70:
                        cell.fill = PatternFill("solid", fgColor="FFEB9C")
                    elif auc < 0.60:
                        cell.fill = PatternFill("solid", fgColor="FFC7CE")
                except Exception:
                    pass


wb.save(summary_path)

print("\nSummary workbook created:")
print(summary_path)
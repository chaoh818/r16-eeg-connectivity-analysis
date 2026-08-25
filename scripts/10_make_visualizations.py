from pathlib import Path
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# Delete old figures and recreate folder
if FIG_DIR.exists():
    shutil.rmtree(FIG_DIR)

FIG_DIR.mkdir(exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("FIG_DIR:", FIG_DIR)


# ============================================================
# 2. Helper functions
# ============================================================

def read_csv_safe(path):
    if not path.exists():
        print(f"Missing file: {path}")
        return pd.DataFrame()
    print(f"Loaded: {path.name}")
    return pd.read_csv(path)


def save_current_figure(filename):
    save_path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", save_path)


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


# ============================================================
# 3. Load data
# ============================================================

best_overall = read_csv_safe(OUTPUT_DIR / "bandwise_best_overall_by_task.csv")
bandwise_all = read_csv_safe(OUTPUT_DIR / "bandwise_results_all_conditions.csv")
permutation_results = read_csv_safe(OUTPUT_DIR / "permutation_results_best_models.csv")

survived_language = read_csv_safe(
    OUTPUT_DIR / "survived_features_language_C_vs_Others_gu2_alpha_ElasticNet.csv"
)

survived_age = read_csv_safe(
    OUTPUT_DIR / "survived_features_age_8-12_vs_5-7_gu2_alpha_ElasticNet.csv"
)


# ============================================================
# 4. Figure 1: Final best results table
# ============================================================

best_results_table = pd.DataFrame([
    {
        "Task": "Age: 8–12 vs 5–7",
        "Best setting": "gu2 | alpha | Elastic Net",
        "AUC": 0.7536,
        "Permutation p": 0.0060,
    },
    {
        "Task": "Language: C vs Others",
        "Best setting": "gu2 | alpha | Elastic Net",
        "AUC": 0.7714,
        "Permutation p": 0.0010,
    },
    {
        "Task": "Language: C vs S",
        "Best setting": "gu3 | delta | Linear SVM",
        "AUC": 0.7400,
        "Permutation p": 0.0120,
    },
])

fig, ax = plt.subplots(figsize=(11, 2.4))
ax.axis("off")

table_data = best_results_table.copy()
table_data["AUC"] = table_data["AUC"].map(lambda x: f"{x:.3f}")
table_data["Permutation p"] = table_data["Permutation p"].map(lambda x: f"{x:.4f}")

table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#D9EAF7")
    else:
        cell.set_facecolor("#FFFFFF")

ax.set_title("Best Model Results Summary", fontsize=14, pad=12)
save_current_figure("00_best_results_table.png")


# ============================================================
# 5. Figure 2: Best model AUC summary
# ============================================================

plt.figure(figsize=(10, 6))

labels = [
    "Age: 8–12 vs 5–7\ngu2 | alpha | Elastic Net",
    "Language: C vs Others\ngu2 | alpha | Elastic Net",
    "Language: C vs S\ngu3 | delta | Linear SVM",
]
auc_values = [0.7536, 0.7714, 0.7400]

plt.bar(range(len(labels)), auc_values)
plt.xticks(range(len(labels)), labels, rotation=12, ha="right")
plt.ylabel("AUC")
plt.ylim(0, 1)
plt.title("Best Model AUC Summary")

for i, v in enumerate(auc_values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

save_current_figure("01_best_model_auc_summary.png")


# ============================================================
# 6. Band-wise grouped bar charts
# ============================================================

def plot_bandwise_task(task_name, model_name, filename, title):
    if bandwise_all.empty:
        return

    task_df = bandwise_all[
        (bandwise_all["task"] == task_name) &
        (bandwise_all["model"] == model_name)
    ].copy()

    if task_df.empty:
        print(f"No data for task={task_name}, model={model_name}")
        return

    pivot_df = task_df.pivot_table(
        index="band",
        columns="condition",
        values="auc_mean",
        aggfunc="max"
    )

    desired_band_order = ["delta", "theta", "alpha", "beta", "highbeta", "gamma", "all"]
    pivot_df = pivot_df.reindex([b for b in desired_band_order if b in pivot_df.index])

    x = np.arange(len(pivot_df.index))
    width = 0.25

    plt.figure(figsize=(11, 6))

    conditions = [c for c in ["gu1", "gu2", "gu3"] if c in pivot_df.columns]

    for i, condition in enumerate(conditions):
        values = pivot_df[condition].values

        plt.bar(
            x + (i - (len(conditions) - 1) / 2) * width,
            values,
            width=width,
            label=condition
        )

    plt.xticks(x, pivot_df.index)
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.title(title)
    plt.legend(title="Condition")

    save_current_figure(filename)


plot_bandwise_task(
    task_name="language_C_vs_Others",
    model_name="ElasticNet",
    filename="02_bandwise_language_C_vs_Others_auc_ElasticNet.png",
    title="Band-wise AUC: Language C vs Others (Elastic Net)"
)

plot_bandwise_task(
    task_name="age_8-12_vs_5-7",
    model_name="ElasticNet",
    filename="03_bandwise_age_8-12_vs_5-7_auc_ElasticNet.png",
    title="Band-wise AUC: Age 8–12 vs 5–7 (Elastic Net)"
)

# Important: C vs S best result is Linear SVM, not ElasticNet
# plot_bandwise_task(
#     task_name="language_C_vs_S",
#     model_name="SVM_linear",
#     filename="04_bandwise_language_C_vs_S_auc_LinearSVM.png",
#     title="Band-wise AUC: Language C vs S (Linear SVM)"
# )


# ============================================================
# 7. Permutation histogram plots
# ============================================================

def plot_permutation_histogram(task, condition, band, model, clean_title):
    if permutation_results.empty:
        return

    safe_name = f"{task}_{condition}_{band}_{model}"
    dist_path = OUTPUT_DIR / f"permutation_distribution_{safe_name}.csv"

    if not dist_path.exists():
        print("Missing permutation distribution:", dist_path)
        return

    dist_df = pd.read_csv(dist_path)

    row = permutation_results[
        (permutation_results["task"] == task) &
        (permutation_results["condition"] == condition) &
        (permutation_results["band"] == band) &
        (permutation_results["model"] == model)
    ]

    if row.empty:
        print("Missing summary row for permutation plot:", safe_name)
        return

    true_auc = row.iloc[0]["true_auc"]
    p_value = row.iloc[0]["p_value"]

    plt.figure(figsize=(9, 5.5))
    plt.hist(dist_df["permutation_auc"], bins=30)
    plt.axvline(true_auc, linestyle="--", linewidth=2)

    plt.xlabel("Permutation AUC")
    plt.ylabel("Frequency")
    plt.title(clean_title)

    plt.text(
        0.98, 0.95,
        f"True AUC = {true_auc:.3f}\np = {p_value:.4f}",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    filename = f"05_permutation_{safe_name}.png"
    save_current_figure(filename)


plot_permutation_histogram(
    task="language_C_vs_Others",
    condition="gu2",
    band="alpha",
    model="ElasticNet",
    clean_title="Permutation Test: Language C vs Others\ngu2 | alpha | Elastic Net"
)

plot_permutation_histogram(
    task="age_8-12_vs_5-7",
    condition="gu2",
    band="alpha",
    model="ElasticNet",
    clean_title="Permutation Test: Age 8–12 vs 5–7\ngu2 | alpha | Elastic Net"
)

# plot_permutation_histogram(
#     task="language_C_vs_S",
#     condition="gu3",
#     band="delta",
#     model="SVM_linear",
#     clean_title="Permutation Test: Language C vs S\ngu3 | delta | Linear SVM"
# )


# ============================================================
# 8. Survived features: absolute importance
# ============================================================

def plot_survived_abs(df, title, filename, top_n=10):
    if df.empty:
        print("No survived features to plot:", filename)
        return

    plot_df = df.sort_values(
        ["survival_count", "mean_abs_coefficient"],
        ascending=[False, False]
    ).head(top_n).copy()

    labels = plot_df["channel_pair"].astype(str).tolist()
    values = plot_df["mean_abs_coefficient"].values

    plt.figure(figsize=(9, 6))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Mean Absolute Coefficient")
    plt.title(title)
    plt.gca().invert_yaxis()

    for i, v in enumerate(values):
        plt.text(v, i, f" {v:.3f}", va="center")

    save_current_figure(filename)


plot_survived_abs(
    df=survived_language,
    title="Top Survived Features by Importance\nLanguage C vs Others | gu2 alpha | Elastic Net",
    filename="06_survived_language_top10_abs.png",
    top_n=10
)

plot_survived_abs(
    df=survived_age,
    title="Top Survived Features by Importance\nAge 8–12 vs 5–7 | gu2 alpha | Elastic Net",
    filename="07_survived_age_top10_abs.png",
    top_n=10
)


# ============================================================
# 9. Survived features: coefficient direction
# ============================================================

def plot_survived_direction(df, title, filename, positive_label, negative_label, top_n=14):
    if df.empty:
        print("No survived features to plot:", filename)
        return

    plot_df = df.sort_values(
        ["survival_count", "mean_abs_coefficient"],
        ascending=[False, False]
    ).head(top_n).copy()

    # Sort by signed coefficient for a more readable direction plot
    plot_df = plot_df.sort_values("mean_coefficient", ascending=True)

    labels = plot_df["channel_pair"].astype(str).tolist()
    values = plot_df["mean_coefficient"].values

    plt.figure(figsize=(10, 7))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels)
    plt.axvline(0, linewidth=1)
    plt.xlabel(f"Mean Coefficient\npositive → {positive_label}, negative → {negative_label}")
    plt.title(title)

    for i, v in enumerate(values):
        if v >= 0:
            plt.text(v, i, f" {v:.3f}", va="center")
        else:
            plt.text(v, i, f"{v:.3f} ", va="center", ha="right")

    save_current_figure(filename)


plot_survived_direction(
    df=survived_language,
    title="Survived Feature Direction\nLanguage C vs Others | gu2 alpha | Elastic Net",
    filename="08_survived_language_direction.png",
    positive_label="C",
    negative_label="Others",
    top_n=16
)

plot_survived_direction(
    df=survived_age,
    title="Survived Feature Direction\nAge 8–12 vs 5–7 | gu2 alpha | Elastic Net",
    filename="09_survived_age_direction.png",
    positive_label="8–12",
    negative_label="5–7",
    top_n=14
)


print("\nAll visualizations completed.")
print("Figures saved in:", FIG_DIR)
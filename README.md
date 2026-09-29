# R16 EEG Connectivity & Spectral Power Analysis Pipeline (v2.0)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pipeline: Validated](https://img.shields.io/badge/Pipeline-Validated-green.svg)](#validated-peak-results)

An end-to-end, reproducible machine learning and statistical pipeline for analyzing **auditory-evoked EEG functional connectivity** (coherence) and **power spectral density (PSD)**. This framework evaluates neurodevelopmental age maturation and linguistic background (tonal vs. non-tonal language processing) across multiple acoustic stimulus conditions and post-stimulus temporal epoch windows.

---

## 🌟 Key Pipeline Features

- **Multimodal Feature Integration**: Combines long-range inter-hemispheric **Functional Connectivity (`FC`)** coherence with local **Power Spectral Density (`PSD`)** oscillatory power.
- **Dynamic Time-Window Optimization**: Evaluates sensory gating (`0–300ms`), mid-latency phonological integration (`300–600ms`), and late cognitive decay (`400–800ms`) epoch windows.
- **Acoustic Stimulus Selectivity**: Compares neural responses across three distinct auditory conditions (`gu1`, `gu2`, `gu3`).
- **Rigorous Cross-Validation**: Uses Stratified 5-Fold Cross-Validation with inner-fold feature scaling (`StandardScaler`) to prevent data leakage.
- **Statistical Significance**: Computes empirical $p$-values via 1,000-run shuffled label permutation testing.
- **L1 Survived Feature Extraction**: Identifies stable, non-zero channel-pair and spectral pathways selected by Elastic Net logistic regression across $\ge 80\%$ ($\ge 4/5$) of CV folds.
- **Publication-Ready Outputs**: Automatically compiles structured Excel workbooks (`R16_EEG_analysis_summary.xlsx`) and high-resolution 300 DPI figures.

---

## 🎯 Classification Tasks & Validated Peak Performance

| Classification Task Target | Target Type | Classes | Chance Baseline | Peak Model AUC | Model Accuracy | Gain Above Chance | Peak Configuration (Condition \| Window \| Band \| Mode \| Model) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Age_Group`** | Lifespan Maturation | 4 | 25.0% | **0.724** | **44.0%** | **+19.0 pp** | `gu2` \| `0–800ms` \| Beta \| PSD \| Elastic Net |
| **`Language_Tonal`** | Binary Language Contrast | 2 | 50.0% | **0.701** | **68.0%** | **+18.0 pp** | `gu2` \| `400–800ms` \| Delta \| FC \| Random Forest |
| **`Age_Child_Language`** | Child Multiclass Interaction | 6 | 16.7% | **0.651** | **23.7%** | **+7.0 pp** | `gu3` \| `300–600ms` \| Delta \| PSD \| Random Forest |
| **`Language`** | 4-Class Language Cohorts | 4 | 25.0% | **0.649** | **43.3%** | **+18.3 pp** | `gu2` \| `0–300ms` \| Full Spectrum \| FC / PSD_FC \| Linear SVM |
| **`Age_Child_Language_Tonal`** | Child Tonal Interaction | 4 | 25.0% | **0.644** | **36.4%** | **+11.4 pp** | `gu1` \| `300–600ms` \| Delta \| PSD \| Random Forest |

---

## 🔄 End-to-End Pipeline Workflow

The repository is organized into a modular execution chain:

```
[Script 05 / 06] Cross-Validated Model Evaluation across Bands, Conditions, & Time Windows
                 (Outputs: dataset_{cond}_{start}_{end}.csv, multiclass_results_all_conditions_{start}_{end}.csv)
       │
       ▼
[Script 065] Time-Window Optimization & Best Model Selection
             (Groups by [Task, Condition], picks max AUC across windows/modes; outputs multiclass_best_results_all_conditions.csv)
       │
       ▼
 ┌─────┴─────────────────────────────────────────┐
 │                                               │
 ▼                                               ▼
[Script 07] Permutation Testing             [Script 08] Survived Feature Extraction
(1,000 label shuffles -> p-values)          (Elastic Net fold selection >= 4/5)
 │                                               │
 └───────────────────────┬───────────────────────┘
                         │
                         ▼
            [Script 09] Master Excel Compiler
            (Generates R16_EEG_analysis_summary.xlsx with 11 sheets)
                         │
                         ▼
            [Script 10] Publication Visualization Suite
            (Generates 9 high-res PNG charts under outputs/figures/)
```

### Script Architecture & Responsibilities

1. **`05_build_task_datasets.py` & `06_compare_bands_all_conditions_multiclass.py`**:
   - Takes `--start_time` and `--end_time` CLI arguments (e.g., `--start_time 0 --end_time 300`).
   - Extracts target variables (`Age_Group`, `Language_Tonal`, `Language`, `Age_Child_Language_Tonal`, `Age_Child_Language`).
   - Saves window-tagged datasets (`dataset_gu3_300_600.csv`) and performance metrics (`multiclass_results_all_conditions_300_600.csv`).

2. **`065_select_best_windows.py`**:
   - Bridge script that scans all window-specific performance CSVs.
   - Groups results by `[Task, Condition]` and extracts the global peak configuration by max `AUC_mean`.
   - Saves `multiclass_best_results_all_conditions.csv` carrying the winning `Window` attribute.

3. **`07_permutation_test_multiclass.py`**:
   - Resolves window-specific dataset paths (`dataset_{cond}_{start}_{end}.csv`).
   - Runs 1,000 iterations of shuffled-label cross-validation to construct empirical null distributions and calculate $p$-values ($p < 0.05$).

4. **`08_survived_features_multiclass.py`**:
   - Dynamically constructs task target vectors (`Age_Group`, `Language`, `Age_Child_Language`).
   - Fits SAGA Elastic Net models per fold and isolates stable "survived features" selected in $\ge 4/5$ folds.
   - Outputs `multiclass_survived_features_task_summary.csv`.

5. **`09_compile_results_summary.py`**:
   - Compiles all metrics, demographics, permutation results, and feature summaries into `R16_EEG_analysis_summary.xlsx`.
   - Formatted across 11 professional sheets with custom navy styling, zebra striping, and auto-adjusted column widths.

6. **`10_make_visualizations.py`**:
   - Generates 9 publication-grade figures saved to `outputs/figures/`:
     - **`00_best_results_table.png`**: High-resolution table displaying all condition results and highlighting the peak condition per task.
     - **`01_best_model_auc_summary.png`**: Grouped bar chart comparing `gu1`, `gu2`, and `gu3` with star callouts on peak configurations.
     - **`02_bandwise_auc_trajectory.png`**: Coherence trajectories across physiological frequency bands (Delta to Gamma).
     - **`04a_survived_features_age_group.png`**: Top stable L1 Elastic Net features for lifespan age maturation.
     - **`04b_survived_features_language_tonal.png`**: Top stable L1 features for binary tonal language processing.
     - **`04c_survived_features_age_child_language_tonal.png`**: Top stable L1 features for child tonal interaction.
     - **`05_time_window_scheme_comparison.png`**: Performance across sensory gating (`0–300ms`), mid-latency (`300–600ms`), and late (`400–800ms`) epoch windows.
     - **`06_feature_mode_synergy.png`**: Comparative gains across `FC`, `PSD`, and multimodal `PSD_FC` feature sets.
     - **`07_stimulus_condition_selectivity.png`**: Acoustic condition selectivity across task targets.
     - **`08_task_granularity_tradeoffs.png`**: Single Y-axis plot contrasting model AUC against 0.500 chance baseline across 5 task granularities.

---

## ⚡ Quick Start & Execution Guide

### Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/chaoh818/r16-eeg-connectivity-analysis.git
cd r16-eeg-connectivity-analysis

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# 1. Run time-window CV evaluations across target windows
python scripts/06_compare_bands_all_conditions_multiclass.py --start_time 0 --end_time 300
python scripts/06_compare_bands_all_conditions_multiclass.py --start_time 300 --end_time 600
python scripts/06_compare_bands_all_conditions_multiclass.py --start_time 400 --end_time 800
python scripts/06_compare_bands_all_conditions_multiclass.py --start_time 0 --end_time 800

# 2. Optimize and select optimal time windows per [Task, Condition]
python scripts/065_select_best_windows.py

# 3. Perform 1,000-run permutation significance testing
python scripts/07_permutation_test_multiclass.py 1000

# 4. Extract L1 Elastic Net survived connectivity/power features
python scripts/08_survived_features_multiclass.py

# 5. Compile Excel summary workbook
python scripts/09_compile_results_summary.py

# 6. Generate full figure visualization suite
python scripts/10_make_visualizations.py
```

---

## 📂 Repository Layout

```
r16-eeg-connectivity-analysis/
├── data/                          # Raw and processed EEG .conn and PSD files
├── scripts/
│   ├── 05_build_task_datasets.py
│   ├── 06_compare_bands_all_conditions_multiclass.py
│   ├── 065_select_best_windows.py
│   ├── 07_permutation_test_multiclass.py
│   ├── 08_survived_features_multiclass.py
│   ├── 09_compile_results_summary.py
│   └── 10_make_visualizations.py
├── outputs/
│   ├── dataset_gu*.csv             # Window-tagged dataset files
│   ├── multiclass_*.csv            # Full and best cross-validated result metrics
│   ├── figures/                    # Generated 300 DPI publication charts (00–08)
│   └── R16_EEG_analysis_summary.xlsx  # Master 11-sheet Excel analysis workbook
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📄 Citation & Attribution

[Tentative] If you use this pipeline or dataset analysis in your research, please cite:

```bibtex
@article{r16_eeg_connectivity_2026,
  title={Multimodal EEG Coherence and Spectral Power Dynamics in Auditory-Evoked Maturation and Language Processing},
  author={R16 EEG Research Group},
  year={2026}
}
```

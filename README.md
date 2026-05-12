# r16-eeg-connectivity-analysis
EEG coherence connectivity analysis pipeline for R16 .conn files.
====================================

1. QUICK SUMMARY
----------------

This project contains the EEG connectivity analysis pipeline for the R16 BESA .conn files.

The pipeline does the following:

- Reads EEG coherence .conn files from three stimulus conditions: gu1, gu2, and gu3.
- Extracts band-averaged coherence features from six frequency bands:
  delta, theta, alpha, beta, high beta, and gamma.
- Converts each participant-condition .conn file into channel-pair connectivity features.
- Merges EEG features with participant metadata from R16 Participant information.xlsx.
- Runs binary classification tasks for language group and age group.
- Compares Linear SVM, Random Forest, and Elastic Net models.
- Evaluates models using stratified 5-fold cross-validation.
- Runs 1000-permutation tests for the best models.
- Runs Elastic Net survived-feature analysis for the two best Elastic Net models.
- Generates a final Excel summary workbook and figures.

Final main outputs:

- outputs/R16_EEG_analysis_summary.xlsx
- outputs/figures/
- outputs/bandwise_results_all_conditions.csv
- outputs/permutation_results_best_models.csv
- outputs/survived_features_task_summary.csv


2. WHAT WAS ANALYSED
--------------------

2.1 Stimulus Conditions

The analysis used three stimulus conditions:

Condition    Description
---------    -----------
gu1          Stimulus condition 1
gu2          Stimulus condition 2
gu3          Stimulus condition 3

Each condition folder contains BESA .conn files. EEG coherence features were extracted separately for each condition.


2.2 Frequency Bands

For each participant and condition, coherence features were extracted from these frequency bands:

Band         Frequency range
----         ----------------
delta        2-4 Hz
theta        4-8 Hz
alpha        8-12 Hz
beta         12-25 Hz
high beta    25-30 Hz
gamma        30-40 Hz

An additional all-band setting was also tested by combining all six frequency bands.


2.3 Classification Tasks

Three binary classification tasks were performed:

Task                      Positive class    Negative class    Purpose
----                      --------------    --------------    -------
Language C vs Others      C                 Others            Test whether EEG coherence features distinguish the C language group from all non-C groups.
Language C vs S           C                 S                 Test a cleaner pairwise language comparison between C and S groups.
Age 8-12 vs 5-7           8-12              5-7               Test whether EEG coherence features distinguish older children from younger children.


2.4 Models Compared

Three machine learning models were compared:

Model            Notes
-----            -----
Linear SVM       Linear support vector machine with balanced class weights.
Random Forest    500-tree random forest with balanced class weights.
Elastic Net      Logistic regression with Elastic Net regularisation.

The main model comparison used stratified 5-fold cross-validation.

The main evaluation metrics were:

- AUC
- Accuracy
- Sensitivity
- Specificity


3. MAIN RESULTS
---------------

The best results were:

Task                    Best setting                    AUC      Permutation p-value
----                    ------------                    ---      -------------------
Language C vs Others    gu2 alpha Elastic Net            0.771    0.001
Age 8-12 vs 5-7         gu2 alpha Elastic Net            0.754    0.006
Language C vs S         gu3 delta Linear SVM             0.740    0.012

Main interpretation:

- The strongest overall result was Language C vs Others using gu2 alpha-band coherence features with Elastic Net.
- The best age classification result was also obtained using gu2 alpha-band coherence features with Elastic Net.
- The cleaner C vs S language comparison showed a different pattern: gu3 delta-band features with Linear SVM performed best.
- The results suggest that useful classification information is condition-specific and frequency-band-specific.
- Combining all bands together was generally less effective than using selected frequency bands.


4. PROJECT STRUCTURE
--------------------

Expected folder structure:

R16_EEG_analysis/
  data/
    gu1/
    gu2/
    gu3/
  meta/
    R16 Participant information.xlsx
  outputs/
  scripts/
  01_check_conn_file.ipynb
  02_train_language_models_gu3.py
  02b_train_language_C_vs_S_gu3.py
  03_train_age_models_gu3.py
  04_compare_bands_gu3.py
  05_extract_all_conditions.py
  06_compare_bands_all_conditions.py
  07_permutation_test_best_models.py
  08_survived_features_elasticnet.py
  09_compile_results_summary.py
  10_make_visualizations.py
  README.txt

Main output folder:

outputs/

Figure output folder:

outputs/figures/


5. INPUT DATA
-------------

5.1 EEG Connectivity Files

The raw EEG connectivity files are BESA .conn files stored under:

data/gu1/
data/gu2/
data/gu3/

Each .conn file contains coherence decomposition data for one participant under one stimulus condition.

Each file includes the following structure:

NumberTimeSamples = 31
NumberFrequencies = 39
NumberChannels = 15
FreqStartInHz = 2
FreqIntervalInHz = 1

This corresponds to:

31 time samples x 39 frequency bins x 15 channels x 15 channels

The frequency range is 2-40 Hz.

The 15 channels are parsed directly from the .conn file header.


5.2 Metadata File

Participant metadata are stored in:

meta/R16 Participant information.xlsx

The metadata file contains participant-level information, including:

- Participant ID
- Language group
- Date of birth
- Date of test
- Language background
- Music background
- TONI scores
- PPVT scores
- TVIP scores


6. PIPELINE
-----------

The pipeline is organised into six main steps.


6.1 Step 1: Feature Extraction and Metadata Merge

Script:

python 05_extract_all_conditions.py

Purpose:

This script reads all .conn files from gu1, gu2, and gu3, extracts EEG coherence features, and merges them with participant metadata.

For each .conn file, the script:

1. Reads the file header.
2. Extracts time sample, frequency, channel, and channel-label information.
3. Extracts the numerical coherence values.
4. Reshapes the data into:

   time x frequency x channel x channel

5. Averages coherence values across time.
6. Computes band-averaged coherence matrices.
7. Extracts upper-triangular channel-pair features.
8. Merges EEG features with participant metadata.

Feature dimension:

- There are 15 EEG channels.
- For each band, the number of unique channel-pair features is:

  15 x 14 / 2 = 105

- Across six frequency bands, each participant-condition file has:

  105 x 6 = 630 EEG coherence features

Main outputs:

outputs/X_conn_gu1.csv
outputs/X_conn_gu2.csv
outputs/X_conn_gu3.csv

outputs/dataset_gu1.csv
outputs/dataset_gu2.csv
outputs/dataset_gu3.csv

outputs/dataset_lang_gu1.csv
outputs/dataset_lang_gu2.csv
outputs/dataset_lang_gu3.csv

outputs/dataset_age_5-7_vs_8-12_gu1.csv
outputs/dataset_age_5-7_vs_8-12_gu2.csv
outputs/dataset_age_5-7_vs_8-12_gu3.csv


6.2 Step 2: Model Comparison Across Conditions and Bands

Script:

python 06_compare_bands_all_conditions.py

Purpose:

This script compares classification performance across:

- Three conditions: gu1, gu2, gu3
- Seven band settings: delta, theta, alpha, beta, high beta, gamma, all
- Three models: Linear SVM, Random Forest, Elastic Net
- Three tasks: Language C vs Others, Language C vs S, Age 8-12 vs 5-7

Models:

1. Linear SVM
   - kernel = linear
   - class_weight = balanced
   - probability = True
   - StandardScaler used before the model

2. Random Forest
   - n_estimators = 500
   - max_features = sqrt
   - class_weight = balanced

3. Elastic Net logistic regression
   - penalty = elasticnet
   - solver = saga
   - l1_ratio = 0.5
   - C = 1.0
   - class_weight = balanced
   - max_iter = 20000
   - StandardScaler used before the model

Evaluation:

- Stratified 5-fold cross-validation
- Accuracy
- AUC
- Sensitivity
- Specificity

AUC was used as the main model-selection metric.

Main outputs:

outputs/bandwise_results_all_conditions.csv
outputs/bandwise_best_results_all_conditions.csv
outputs/bandwise_best_overall_by_task.csv


6.3 Step 3: Permutation Testing

Script:

python 07_permutation_test_best_models.py

Purpose:

This script tests whether the best model AUC values are significantly higher than random-label performance.

Method:

1. Compute the true cross-validated AUC using the original labels.
2. Randomly shuffle the labels.
3. Recompute cross-validated AUC with shuffled labels.
4. Repeat the process 1000 times.
5. Build a null distribution of permutation AUC values.
6. Calculate a p-value.

P-value formula with add-one correction:

p = (number of permutation AUCs >= true AUC + 1) / (number of permutations + 1)

Best models tested:

Task                    Condition    Band     Model
----                    ---------    ----     -----
Age 8-12 vs 5-7         gu2          alpha    Elastic Net
Language C vs Others    gu2          alpha    Elastic Net
Language C vs S         gu3          delta    Linear SVM

Main outputs:

outputs/permutation_results_best_models.csv
outputs/permutation_distribution_age_8-12_vs_5-7_gu2_alpha_ElasticNet.csv
outputs/permutation_distribution_language_C_vs_Others_gu2_alpha_ElasticNet.csv
outputs/permutation_distribution_language_C_vs_S_gu3_delta_SVM_linear.csv


6.4 Step 4: Elastic Net Survived-Feature Analysis

Script:

python 08_survived_features_elasticnet.py

Purpose:

This script identifies stable channel-pair features selected by Elastic Net across cross-validation folds.

This analysis was applied to the two best Elastic Net models:

Task                    Condition    Band     Model
----                    ---------    ----     -----
Language C vs Others    gu2          alpha    Elastic Net
Age 8-12 vs 5-7         gu2          alpha    Elastic Net

Method:

1. Run 5-fold cross-validation.
2. Train Elastic Net on each fold.
3. Extract the coefficient for each feature in each fold.
4. Mark a feature as selected if its coefficient is non-zero.
5. Count how often each feature is selected across the 5 folds.
6. Define survived features as features selected in at least 4 out of 5 folds.

Survival threshold:

survival_count >= 4

Output fields include:

- feature
- channel_pair
- survival_count
- survival_rate
- mean_coefficient
- mean_abs_coefficient
- direction

Direction interpretation:

For Language C vs Others:

- positive coefficient = towards C
- negative coefficient = towards Others

For Age 8-12 vs 5-7:

- positive coefficient = towards 8-12
- negative coefficient = towards 5-7

Main outputs:

outputs/elasticnet_all_fold_coefficients_language_C_vs_Others_gu2_alpha_ElasticNet.csv
outputs/elasticnet_feature_summary_language_C_vs_Others_gu2_alpha_ElasticNet.csv
outputs/survived_features_language_C_vs_Others_gu2_alpha_ElasticNet.csv

outputs/elasticnet_all_fold_coefficients_age_8-12_vs_5-7_gu2_alpha_ElasticNet.csv
outputs/elasticnet_feature_summary_age_8-12_vs_5-7_gu2_alpha_ElasticNet.csv
outputs/survived_features_age_8-12_vs_5-7_gu2_alpha_ElasticNet.csv

outputs/survived_features_task_summary.csv


6.5 Step 5: Compile Excel Summary

Script:

python 09_compile_results_summary.py

Purpose:

This script compiles the main outputs into a formatted Excel workbook.

Main output:

outputs/R16_EEG_analysis_summary.xlsx

Workbook sheets:

Sheet name                  Content
----------                  -------
00_Key_Findings             Main findings and interpretations
01_Best_Results             Final best-result table
02_Executive_Summary        Compact model and permutation summary
03_Dataset_Counts           Dataset and metadata counts
04_Best_Overall             Best models by task
05_Best_By_Condition        Best results by task and condition
06_Permutation_Tests        1000-permutation test results
07_Survived_Summary         Survived-feature counts
08_Survived_Language        Language survived features
09_Survived_Age             Age survived features
10_All_Bandwise_Results     Full band-wise model results

Most useful sheets for quick review:

- 00_Key_Findings
- 01_Best_Results
- 03_Dataset_Counts
- 06_Permutation_Tests
- 08_Survived_Language
- 09_Survived_Age


6.6 Step 6: Generate Figures

Script:

python 10_make_visualizations.py

Purpose:

This script creates the final figure folder and generates visual summaries.

Main output folder:

outputs/figures/

Generated figures:

00_best_results_table.png
01_best_model_auc_summary.png
02_bandwise_language_C_vs_Others_auc_ElasticNet.png
03_bandwise_age_8-12_vs_5-7_auc_ElasticNet.png
04_bandwise_language_C_vs_S_auc_LinearSVM.png
05_permutation_language_C_vs_Others_gu2_alpha_ElasticNet.png
05_permutation_age_8-12_vs_5-7_gu2_alpha_ElasticNet.png
05_permutation_language_C_vs_S_gu3_delta_SVM_linear.png
06_survived_language_top10_abs.png
07_survived_age_top10_abs.png
08_survived_language_direction.png
09_survived_age_direction.png

Most useful figures for presentation:

- 00_best_results_table.png
- 01_best_model_auc_summary.png
- 02_bandwise_language_C_vs_Others_auc_ElasticNet.png
- 03_bandwise_age_8-12_vs_5-7_auc_ElasticNet.png
- 04_bandwise_language_C_vs_S_auc_LinearSVM.png
- 05_permutation_language_C_vs_Others_gu2_alpha_ElasticNet.png
- 05_permutation_age_8-12_vs_5-7_gu2_alpha_ElasticNet.png
- 05_permutation_language_C_vs_S_gu3_delta_SVM_linear.png
- 08_survived_language_direction.png
- 09_survived_age_direction.png


7. HOW TO RUN THE FULL PIPELINE
-------------------------------

Run the scripts in this order:

Step 1: Extract features and merge metadata

python 05_extract_all_conditions.py

Step 2: Compare conditions, bands, tasks, and models

python 06_compare_bands_all_conditions.py

Step 3: Run permutation tests

python 07_permutation_test_best_models.py

Step 4: Run Elastic Net survived-feature analysis

python 08_survived_features_elasticnet.py

Step 5: Compile Excel summary

python 09_compile_results_summary.py

Step 6: Generate figures

python 10_make_visualizations.py

The order is important because later scripts depend on output files generated by earlier scripts.


8. DETAILED RESULTS
-------------------

8.1 Best Model Results

Task                    Best condition    Best band    Best model     AUC      Permutation p-value
----                    --------------    ---------    ----------     ---      -------------------
Language C vs Others    gu2               alpha        Elastic Net    0.771    0.001
Age 8-12 vs 5-7         gu2               alpha        Elastic Net    0.754    0.006
Language C vs S         gu3               delta        Linear SVM     0.740    0.012


8.2 Language C vs Others

Best setting:

- Condition: gu2
- Band: alpha
- Model: Elastic Net
- AUC: approximately 0.771
- Permutation p-value: approximately 0.001

Interpretation:

This was the strongest overall result. It suggests that gu2 alpha-band coherence features contain useful information for distinguishing C participants from non-C participants.


8.3 Age 8-12 vs 5-7

Best setting:

- Condition: gu2
- Band: alpha
- Model: Elastic Net
- AUC: approximately 0.754
- Permutation p-value: approximately 0.006

Interpretation:

This suggests that gu2 alpha-band coherence features also contain useful information for distinguishing older children from younger children.


8.4 Language C vs S

Best setting:

- Condition: gu3
- Band: delta
- Model: Linear SVM
- AUC: approximately 0.740
- Permutation p-value: approximately 0.012

Interpretation:

The cleaner C vs S language contrast showed a different best condition-band combination compared with C vs Others. The best model used gu3 delta-band features with Linear SVM.


8.5 Survived Features

Elastic Net survived-feature analysis produced:

Task                    Condition    Band     Model          Survived features
----                    ---------    ----     -----          -----------------
Language C vs Others    gu2          alpha    Elastic Net    16
Age 8-12 vs 5-7         gu2          alpha    Elastic Net    14

Interpretation:

These features are channel-pair coherence values that were repeatedly selected by Elastic Net across cross-validation folds.


9. KEY FINDINGS
---------------

1. Band-wise features were more informative than all-band combined features.

   The strongest results were found in specific frequency bands rather than by combining all frequency bands into one large feature set.

2. gu2 alpha-band coherence was the most important setting for two tasks.

   Both Language C vs Others and Age 8-12 vs 5-7 achieved their best performance with gu2 alpha-band features using Elastic Net.

3. C vs S showed a different pattern.

   The best C vs S result used gu3 delta-band features with Linear SVM.

4. The best models were significantly above random-label performance.

   All three selected best models passed 1000-permutation testing with p-values below 0.05.

5. Elastic Net identified stable survived features.

   The two gu2 alpha Elastic Net models produced stable survived channel-pair features across 5-fold cross-validation.


10. LIMITATIONS
---------------

The current analysis should be interpreted as preliminary because:

1. The sample size is relatively small.
2. Some participants could not be matched to complete metadata.
3. The models were evaluated using cross-validation rather than an independent held-out test set.
4. Hyperparameters were fixed rather than optimised using nested cross-validation.
5. Survived-feature analysis was based on 5 folds with a 4/5 survival threshold.
6. The results show classification signal, but they do not establish causal neurophysiological mechanisms.


11. POSSIBLE FUTURE IMPROVEMENTS
--------------------------------

Future improvements could include:

1. Completing missing participant metadata.
2. Running repeated stratified cross-validation.
3. Using nested cross-validation for hyperparameter tuning.
4. Testing PCA or feature-reduction approaches.
5. Running a more stable repeated survived-feature analysis.
6. Visualising survived channel-pair features as brain-network graphs.
7. Adding an independent validation set if more data become available.


12. SHORT SUMMARY
-----------------

This project extracted EEG coherence features from BESA .conn files across three stimulus conditions and six frequency bands. Machine learning models were trained to classify language and age groups using these features.

The strongest results were:

Language C vs Others:
- gu2 alpha Elastic Net
- AUC approximately 0.771
- permutation p approximately 0.001

Age 8-12 vs 5-7:
- gu2 alpha Elastic Net
- AUC approximately 0.754
- permutation p approximately 0.006

Language C vs S:
- gu3 delta Linear SVM
- AUC approximately 0.740
- permutation p approximately 0.012

Overall, the results suggest that EEG coherence classification performance is frequency-band-specific and condition-specific, with gu2 alpha-band coherence showing the strongest signal for both the Language C vs Others task and the Age 8-12 vs 5-7 task.

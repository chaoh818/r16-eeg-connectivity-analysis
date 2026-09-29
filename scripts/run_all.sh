#!/bin/bash

# Define the exact time windows as pairs of "start end"
time_windows=(
    "0 800"
    "0 400"
    "400 800"
    "0 300"
    "300 600"
    "600 900"
)

# Loop through each pair in the array
for window in "${time_windows[@]}"; do
    # Extract the start and end values into individual variables
    read -r start_time end_time <<< "$window"
    
    echo "------------------------------------------------"
    echo "Starting iteration for epoch: ${start_time}ms to ${end_time}ms"
    
    # Execute your target script. 
    /home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysis/scripts/05_extract_all_conditions.py --start_time "$start_time" --end_time "$end_time" 
    /home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysis/scripts/06_compare_bands_all_conditions.py --start_time "$start_time" --end_time "$end_time" 
done

echo "------------------------------------------------"
echo "All 6 iterations completed."

/home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysisscripts/065_select_best_window.py  
/home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysisscripts/07_permutation_test_best_models.py  
/home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysisscripts/08_survived_features_elasticnet.py  
/home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysisscripts/09_compile_results_summary.py
/home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysisscripts/10_make_visualizations.py

echo "------------------------------------------------"
echo "Post processing and visualization completed."
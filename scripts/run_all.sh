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
    /home/chaopear/dev/r16-eeg-connectivity-analysis/r16venv/bin/python /home/chaopear/dev/r16-eeg-connectivity-analysis/scripts/006_compare_bands_all_conditions_multiclass.py
    mv /home/chaopear/dev/r16-eeg-connectivity-analysis/outputs/multiclass_best_results_all_conditions.csv /home/chaopear/dev/r16-eeg-connectivity-analysis/outputs/multiclass_best_results_all_conditions_${start_time}_${end_time}.csv
done

echo "------------------------------------------------"
echo "All 6 iterations completed."
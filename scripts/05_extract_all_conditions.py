from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# 1. Set paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "scripts":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
base_path = Path(DATA_DIR).expanduser()

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

data_channels = ["ACtL", "ACrL", "ACaL", "ACtR", "ACrR", "ACaR"]
n_ch_used = len(data_channels)


freq_bands = {
        "delta": (2, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 25),
        "highbeta": (25, 30),
        "gamma": (30, 50),
    }

# ============================================================
# 2. Helper functions for .conn files
# ============================================================

def parse_conn_header(text):
    header_patterns = {
        "NumberTimeSamples": r"NumberTimeSamples=(\d+)",
        "NumberFrequencies": r"NumberFrequencies=(\d+)",
        "FreqStartInHz": r"FreqStartInHz=([0-9.]+)",
        "FreqIntervalInHz": r"FreqIntervalInHz=([0-9.]+)",
        "NumberChannels": r"NumberChannels=(\d+)",
        "TimeStartInMS": r"TimeStartInMS=(-?[0-9.]+)",
        "IntervalInMS": r"IntervalInMS=([0-9.]+)",                
    }

    header = {}

    for key, pattern in header_patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            header[key] = float(value) if "." in value else int(value)
        else:
            header[key] = None

    lines = text.splitlines()
    channels = None

    for i, line in enumerate(lines):
        if "NumberChannels=" in line:
            if i + 1 < len(lines):
                channels = lines[i + 1].strip().split()
            break

    header["channels"] = channels
    return header


def extract_numeric_data_after_channels(text):
    lines = text.splitlines()
    data_start_idx = None

    for i, line in enumerate(lines):
        if "NumberChannels=" in line:
            data_start_idx = i + 2
            break

    if data_start_idx is None:
        raise ValueError("Could not find NumberChannels line.")

    data_text = "\n".join(lines[data_start_idx:])
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", data_text)
    values = np.array(numbers, dtype=float)
    return values

def extract_participant_id(filename):
    return filename.split("_")[0].upper()


def make_canonical_id(pid):
    if pd.isna(pid):
        return np.nan

    pid = str(pid).strip()
    pid_upper = pid.upper()

    # R16A02g05 -> R16A02
    # R16C05y13speech -> R16C05
    # R16E08y05 -> R16E08
    match = re.match(r"^(R16[A-Z]+\d{2})", pid_upper)
    if match:
        return match.group(1)

    # R1629
    match = re.match(r"^(R16\d+)", pid_upper)
    if match:
        return match.group(1)

    return pid_upper


def extract_upper_triangle_features(band_matrices, channels):
    feature_dict = {}

    n_ch = len(channels)
    triu_idx = np.triu_indices(n_ch, k=1)

    for band_name, matrix in band_matrices.items():
        for i, j in zip(*triu_idx):
            ch1 = channels[i]
            ch2 = channels[j]
            feature_name = f"{band_name}_{ch1}_{ch2}"
            feature_dict[feature_name] = matrix[i, j]

    return feature_dict


def read_conn_file_to_features(conn_path, condition_name):
    text = conn_path.read_text(encoding="utf-8", errors="ignore")

    header = parse_conn_header(text)
    values = extract_numeric_data_after_channels(text)

    n_time = header["NumberTimeSamples"] # 31
    n_freq = header["NumberFrequencies"] # 49
    n_ch = header["NumberChannels"] # 15
    time_start = header["TimeStartInMS"] # -500
    interval = header["IntervalInMS"] # 50

    time_start_used = 0
    time_end_used = 800

    n_time_start_used = int((time_start_used - time_start) / interval)
    n_time_end_used = int((time_end_used - time_start) / interval)

    expected_4d = n_time * n_freq * n_ch * n_ch
    expected_3d = n_freq * n_ch * n_ch
    if len(values) == expected_4d:
        data_4d = values.reshape(n_time, n_freq, n_ch, n_ch)
    elif len(values) == expected_3d:
        data_4d = values.reshape(1, n_freq, n_ch, n_ch)
    else:
        raise ValueError(
            f"{conn_path.name}: value count mismatch. "
            f"Got {len(values)}, expected {expected_4d} or {expected_3d}"
        )

    # truncate time frame from 31 (-500 - 1000ms) to 17 (0 - 800ms), and channels to first 6 x 6
    data_4d_trunc = data_4d[n_time_start_used:n_time_end_used+1, :, :n_ch_used, :n_ch_used]
    # Average across time sam`p`les
    data_freq_ch_ch = data_4d_trunc.mean(axis=0)

    # Frequency list
    freqs = (
        header["FreqStartInHz"]
        + np.arange(header["NumberFrequencies"]) * header["FreqIntervalInHz"]
    )    
    band_matrices = {}
    for band_name, (low, high) in freq_bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        band_matrices[band_name] = data_freq_ch_ch[idx].mean(axis=0)
    features = extract_upper_triangle_features(band_matrices, header["channels"][:n_ch_used])

    participant_id = extract_participant_id(conn_path.name)
    row = {
        "participant_id": participant_id,
        "filename": conn_path.name,
        "condition": condition_name,
    }

    row.update(features)
    
    # alternatively, keep the time dimension to explore best window
    # for t_idx in range(n_time_end_used+1-n_time_start_used):
    #     ms = t_idx * 50  # Index 10 is 0ms (stimulus onset) [8]        
    #     coherence_at_t = data_4d_trunc[t_idx, :, :, :]
    #     band_matrices = {}
    #     for band_name, (low, high) in freq_bands.items():
    #         idx = np.where((freqs >= low) & (freqs <= high))[0]
    #         band_matrices[band_name] = coherence_at_t[idx].mean(axis=0)
    #     features = extract_upper_triangle_features(band_matrices, header["channels"][:n_ch_used])
    #     suffix_features = {f"{key}_{ms}ms":value for key, value in features.items()}
    #     row.update(suffix_features)

    return row


# ============================================================
# 3. Prepare metadata from conn file paths
# ============================================================

def extract_group_info(directory_name):
    """
    Parses the directory name and returns a tuple of (language, age_group).
    """
    dir_lower = directory_name.lower()
    
    if 'eng' in dir_lower or 'ees' in dir_lower:
        language = 'English'
    elif 'man' in dir_lower or 'mms' in dir_lower:
        language = 'Mandarin'
    else:
        language = 'Unknown'
        
    if '5_7' in dir_lower:
        age_group = '5-7'
    elif '8_11' in dir_lower:
        age_group = '8-12'
    elif 'teen' in dir_lower:
        age_group = 'Teen'
    elif 'adult' in dir_lower:
        age_group = 'Adult'
    else:
        age_group = 'Unknown'
        
    return language, age_group

def read_meta_data_from_conn_file(conn_path):
    relative_path = conn_path.relative_to(base_path)
    parts = relative_path.parts
    
    # pattern is: <Group_Dir> / Coherence / Complex Demodulation / <Stimulus_Dir> / <Filename>
    # eg. ('2026-08-18_eng_children_5_7yrs_connect', 'Coherence', 'Complex Demodulation', 'gu1', 'R16C99b8.conn')
    group_dir = parts[0]
    stimulus_dir = parts[3]
    filename = parts[4]

    # print(f"Directory: {group_dir}")
    # print(f"Stimulus:  {stimulus_dir}")
    # print(f"Filename:  {filename}")
    # print("-" * 40)
    
    lang, age_group = extract_group_info(group_dir)
    
    row = {
        "participant_id": extract_participant_id(conn_path.name),
        "lang": lang,
        "age_group": age_group,
        "file_name": filename
    }

    return row

rows = []
failed_files = []

search_pattern = "*/Coherence/Complex Demodulation/*/*.conn"
conn_files = list(base_path.glob(search_pattern))
for conn_path in conn_files:
    try:
        row = read_meta_data_from_conn_file(conn_path)
        rows.append(row)
    except Exception as e:
        failed_files.append((conn_path.name, str(e)))

X_meta_all = pd.DataFrame(rows)
X_meta = X_meta_all.drop_duplicates(
    subset=["participant_id"],
    keep="first"
).copy()

X_meta.to_csv(OUTPUT_DIR / "metadata_unique_all_conditions.csv", index=False)

print("\nMetadata prepared.")
print("Unique metadata shape:", X_meta.shape)

# ============================================================
# 4. Extract and merge each condition
# ============================================================

conditions = ["gu1", "gu2", "gu3"]

for condition in conditions:
    print("\n" + "=" * 80)
    print("Processing condition:", condition)

    rows = []
    failed_files = []

    search_pattern = f"*/Coherence/Complex Demodulation/{condition}/*.conn"
    conn_files = list(base_path.glob(search_pattern))
    print(f"Number of valid {condition} .conn files:", len(conn_files))

    for conn_path in conn_files:
        try:
            row = read_conn_file_to_features(conn_path, condition)
            rows.append(row)
        except Exception as e:
            failed_files.append((conn_path.name, str(e)))

    X_conn = pd.DataFrame(rows)

    print("Successfully processed:", len(rows))
    print("Failed files:", len(failed_files))
    print("X_conn shape:", X_conn.shape)

    if failed_files:
        print("\nFailed files:")
        for name, err in failed_files:
            print(name, "->", err)

    # Save feature table
    x_path = OUTPUT_DIR / f"X_conn_{condition}.csv"
    X_conn.to_csv(x_path, index=False)

    # Merge metadata
    dataset = X_conn.merge(
        X_meta,
        on="participant_id",
        how="left",
        suffixes=("_conn", "_meta")
    )

    dataset_path = OUTPUT_DIR / f"dataset_{condition}.csv"
    dataset.to_csv(dataset_path, index=False)

    # Language dataset: valid lang only
    dataset_lang = dataset[
        dataset["lang"].notna()
        & ~dataset["lang"].astype(str).str.lower().isin(["nan", "none", ""])
    ].copy()

    dataset_lang_path = OUTPUT_DIR / f"dataset_lang_{condition}.csv"
    dataset_lang.to_csv(dataset_lang_path, index=False)

    dataset_age = dataset[
        # dataset["age_group"].isin(["5-7", "8-12", "Teen", "Adult"])
        dataset["age_group"].isin(["5-7", "8-12"])
    ].copy()

    dataset_age_path = OUTPUT_DIR / f"dataset_age_{condition}.csv"
    dataset_age.to_csv(dataset_age_path, index=False)

    missing_meta = dataset[dataset["lang"].isna()]

    print("Saved:", x_path)
    print("Saved:", dataset_path)
    print("Saved:", dataset_lang_path)
    print("Saved:", dataset_age_path)

    print("\nMerged dataset shape:", dataset.shape)
    print("Rows without lang metadata:", missing_meta.shape[0])

    print("\nLanguage counts:")
    print(dataset_lang["lang"].value_counts(dropna=False))

    print("\nAge group counts:")
    print(dataset["age_group"].value_counts(dropna=False))

print("\nAll conditions processed.")
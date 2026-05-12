from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# 1. Set paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
META_PATH = BASE_DIR / "meta" / "R16 Participant information.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("META_PATH:", META_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)


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
    return filename.split("_")[0]


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

    n_time = header["NumberTimeSamples"]
    n_freq = header["NumberFrequencies"]
    n_ch = header["NumberChannels"]

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

    # Average across time samples
    data_freq_ch_ch = data_4d.mean(axis=0)

    # Frequency list
    freqs = (
        header["FreqStartInHz"]
        + np.arange(header["NumberFrequencies"]) * header["FreqIntervalInHz"]
    )

    bands = {
        "delta": (2, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 25),
        "highbeta": (25, 30),
        "gamma": (30, 40),
    }

    band_matrices = {}

    for band_name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        band_matrices[band_name] = data_freq_ch_ch[idx].mean(axis=0)

    features = extract_upper_triangle_features(band_matrices, header["channels"])

    participant_id = extract_participant_id(conn_path.name)

    row = {
        "participant_id": participant_id,
        "canonical_id": make_canonical_id(participant_id),
        "filename": conn_path.name,
        "condition": condition_name,
    }
    row.update(features)

    return row


# ============================================================
# 3. Prepare metadata
# ============================================================

meta_raw = pd.read_excel(META_PATH)

metadata_clean = meta_raw.copy()

metadata_clean = metadata_clean.rename(columns={
    "Participant ID": "participant_id",
    "DOE/date of test": "date_of_test",
    "Language background": "language_background",
    "Music background": "music_background",
})

metadata_clean = metadata_clean[
    [
        "participant_id",
        "lang",
        "DOB",
        "date_of_test",
        "language_background",
        "music_background",
        "Toni raw",
        "Toni SS",
        "PPVT raw",
        "PPVT ss",
        "TVIP raw",
        "TVIP ss",
    ]
].copy()

metadata_clean["participant_id"] = metadata_clean["participant_id"].astype(str).str.strip()
metadata_clean["lang"] = metadata_clean["lang"].astype(str).str.strip()
metadata_clean["lang"] = metadata_clean["lang"].replace(["nan", "NaN", "None", ""], np.nan)

metadata_clean["DOB"] = pd.to_datetime(metadata_clean["DOB"], errors="coerce")
metadata_clean["date_of_test"] = pd.to_datetime(metadata_clean["date_of_test"], errors="coerce")

metadata_clean["age"] = (
    (metadata_clean["date_of_test"] - metadata_clean["DOB"]).dt.days / 365.25
)

def make_age_group(age):
    if pd.isna(age):
        return np.nan
    elif 5 <= age <= 7:
        return "5-7"
    elif 8 <= age <= 12:
        return "8-12"
    elif 13 <= age <= 19:
        return "Teens"
    else:
        return "Other"

metadata_clean["age_group"] = metadata_clean["age"].apply(make_age_group)

metadata_clean["lang_binary"] = metadata_clean["lang"].apply(
    lambda x: "C" if x == "C" else ("Others" if pd.notna(x) else np.nan)
)

metadata_clean["exact_id"] = metadata_clean["participant_id"].astype(str).str.strip().str.upper()
metadata_clean["canonical_id"] = metadata_clean["participant_id"].apply(make_canonical_id)
metadata_clean["is_base_id"] = metadata_clean["exact_id"] == metadata_clean["canonical_id"]

# Resolve duplicated canonical IDs by preferring base IDs
metadata_sorted = metadata_clean.sort_values(
    by=["canonical_id", "is_base_id"],
    ascending=[True, False]
)

metadata_unique = metadata_sorted.drop_duplicates(
    subset=["canonical_id"],
    keep="first"
).copy()

metadata_clean.to_csv(OUTPUT_DIR / "metadata_clean_all_conditions.csv", index=False)
metadata_unique.to_csv(OUTPUT_DIR / "metadata_unique_all_conditions.csv", index=False)

print("\nMetadata prepared.")
print("Raw metadata shape:", meta_raw.shape)
print("Clean metadata shape:", metadata_clean.shape)
print("Unique metadata shape:", metadata_unique.shape)


# ============================================================
# 4. Extract and merge each condition
# ============================================================

conditions = ["gu1", "gu2", "gu3"]

for condition in conditions:
    condition_dir = DATA_DIR / condition

    print("\n" + "=" * 80)
    print("Processing condition:", condition)
    print("Folder:", condition_dir)
    print("Folder exists?", condition_dir.exists())

    if not condition_dir.exists():
        print(f"WARNING: folder not found: {condition_dir}")
        continue

    # Only use files whose filename contains _gu1_ / _gu2_ / _gu3_
    conn_files = sorted([
        f for f in condition_dir.glob("*.conn")
        if f"_{condition}_" in f.name
    ])

    print(f"Number of valid {condition} .conn files:", len(conn_files))

    rows = []
    failed_files = []

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
        metadata_unique.drop(columns=["exact_id"], errors="ignore"),
        on="canonical_id",
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

    dataset_lang["lang_binary"] = dataset_lang["lang"].apply(
        lambda x: "C" if x == "C" else "Others"
    )

    dataset_lang_path = OUTPUT_DIR / f"dataset_lang_{condition}.csv"
    dataset_lang.to_csv(dataset_lang_path, index=False)

    # Age dataset: 5-7 vs 8-12 only
    dataset_age = dataset[
        dataset["age_group"].isin(["5-7", "8-12"])
    ].copy()

    dataset_age_path = OUTPUT_DIR / f"dataset_age_5-7_vs_8-12_{condition}.csv"
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

    print("\nLanguage binary counts:")
    print(dataset_lang["lang_binary"].value_counts(dropna=False))

    print("\nAge group counts:")
    print(dataset["age_group"].value_counts(dropna=False))

print("\nAll conditions processed.")
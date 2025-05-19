import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import pandas as pd

# Dataset information
datasets = {
    "RAVDESS": {
        "kaggle_name": "uwrfkaggler/ravdess-emotional-speech-audio",
        "zip_file": "ravdess-emotional-speech-audio.zip"
    },
    # "CREMA-D": {
    #     "kaggle_name": "demonji/crema-d",
    #     "zip_file": "crema-d.zip"
    # },
    "TESS": {
        "kaggle_name": "ejlok1/toronto-emotional-speech-set-tess",
        "zip_file": "toronto-emotional-speech-set-tess.zip"
    }
}

BASE_DIR = "datasets"
RAW_DIR = os.path.join(BASE_DIR, "raw") # datasets/raw
PROC_DIR = os.path.join(BASE_DIR, "processed") # datasets/processed
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# Authenticate Kaggle
api = KaggleApi()
api.authenticate()

# Download & unzip datasets
for name, info in datasets.items():
    print(f"\n📥 Downloading {name}...")
    zip_path = os.path.join(RAW_DIR, info["zip_file"])

    if not os.path.exists(zip_path.replace('.zip', '')):  # Skip if already extracted
        api.dataset_download_files(info["kaggle_name"], path=RAW_DIR, unzip=False)
        print(f"✅ Downloaded {info['zip_file']}")

        print(f"📂 Extracting {info['zip_file']}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(RAW_DIR, name))
        os.remove(zip_path)
        print(f"✅ Extracted to {os.path.join(RAW_DIR, name)}")
    else:
        print(f"⏩ {name} already exists, skipping.")

# Emotion mapping from filenames
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
    'DIS': 'disgust', 'ANG': 'angry', 'FEA': 'fearful',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad',
}

def parse_emotion(filename, dataset):
    if dataset == "RAVDESS":
        parts = filename.split('-')
        if len(parts) > 2:
            return emotion_map.get(parts[2], 'unknown')
    elif dataset == "CREMA-D":
        parts = filename.split('_')
        if len(parts) > 2:
            return emotion_map.get(parts[2].split('.')[0], 'unknown')
    elif dataset == "TESS":
        for emo in emotion_map.values():
            if emo in filename.lower():
                return emo
    return "unknown"

# Process and organize all audio files into one folder
print("\n🚀 Organizing audio files...")

data = []

for dataset_name in tqdm(datasets.keys(), desc="🔄 Processing Datasets"):
    root = os.path.join(RAW_DIR, dataset_name)
    for root_dir, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith(".wav"):
                src_path = os.path.join(root_dir, file)
                emotion = parse_emotion(file, dataset_name)

                dest_filename = f"{dataset_name}_{file}"
                dest_path = os.path.join(PROC_DIR, dest_filename)

                shutil.copy2(src_path, dest_path)

                data.append({
                    "filepath": dest_path.replace("\\", "/"),  # Normalize Windows paths
                    "emotion": emotion,
                    "source_dataset": dataset_name
                })

# Create CSV
df = pd.DataFrame(data)
df = df[df["emotion"] != "unknown"]  # Remove unknowns
csv_path = os.path.join(BASE_DIR, "metadata.csv")
df.to_csv(csv_path, index=False)

print(f"\n✅ All files organized and labeled.")
print(f"📄 Metadata saved to: {csv_path}")
print(f"🎧 Total usable samples: {len(df)}")

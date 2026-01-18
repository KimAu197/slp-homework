import os
import parselmouth
from parselmouth.praat import call
import csv

output_files = {
    "my_samples": "my_features.csv",
    "MSP_samples": "msp_features.csv"
}

WORD_COUNT_MSP = {
    "Happy": 19,
    "Angry": 13,
    "Sad": 17,
    "Afraid": 31,
    "Surprised": 16,
    "Disgusted": 26,
    "Neutral": 9
}

# My sentense: My neighbor bought a new car last week because his old one stopped working.

WORD_COUNT_MY = {
    "Happy": 12,
    "Angry": 12,
    "Sad": 12,
    "Afraid": 12,
    "Surprised": 12,
    "Disgusted": 12,
    "Neutral": 12
}

def extract_features(sound_path, emotion_name, dataset_type="my"):

    sound = parselmouth.Sound(sound_path)
    dur = sound.get_total_duration()

    # Pitch
    pitch = call(sound, "To Pitch", 0.0, 75, 600)

    pitch_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    pitch_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    pitch_sd = call(pitch, "Get standard deviation", 0, 0, "Hertz")

    # Intensity
    inte = call(sound, "To Intensity", 100.0, 0.0, "yes") 
    int_min = call(inte, "Get minimum", 0, 0, "Parabolic")
    int_max = call(inte, "Get maximum", 0, 0, "Parabolic")
    int_mean = call(inte, "Get mean", 0, 0, "energy")
    int_sd = call(inte, "Get standard deviation", 0, 0)

    # Jitter 
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # Shimmer 
    shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # HNR 
    har = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_mean = call(har, "Get mean", 0, 0)

    # Speaking rate 
    if dataset_type == "my":
        n_words = WORD_COUNT_MY.get(emotion_name)
    elif dataset_type == "msp":
        n_words = WORD_COUNT_MSP.get(emotion_name)
    
    speaking_rate = n_words / dur

    return [
        pitch_min, pitch_max, pitch_mean, pitch_sd,
        int_min, int_max, int_mean, int_sd,
        speaking_rate, jitter, shimmer, hnr_mean
    ]


def process_folder(folder_name):

    csv_path = output_files[folder_name]

    if "my" in folder_name.lower():
        dataset_type = "my"  
    else:
        dataset_type = "msp"

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        emotion = row["Speech File"]
        wav_path = os.path.join(folder_name, f"{emotion}.wav")

        features = extract_features(wav_path, emotion, dataset_type)
        (row["Min Pitch"], row["Max Pitch"], row["Mean Pitch"], row["Sd Pitch"],
            row["Min Intensity"], row["Max Intensity"], row["Mean Intensity"], row["Sd Intensity"],
            row["Speaking Rate"], row["Jitter"], row["Shimmer"], row["HNR"]) = features

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f" Updated {csv_path}")


if __name__ == "__main__":
    for folder in ["my_samples", "MSP_samples"]:
        print(f"Processing {folder}")
        process_folder(folder)
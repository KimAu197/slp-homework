# HW3: Emotion Recognition in Speech

jl7199 Jinzi Luo

## Overview

This project implements emotion recognition from speech using acoustic-prosodic features. The dataset consists of 2,324 WAV files from the Emotional Prosody Speech and Transcripts corpus, covering 7 speakers and 15 emotion classes.

## File Structure

```
jl7199/
├── feature_analysis.ipynb   # Feature extraction and analysis (Part 1)
├── classification.ipynb     # Classification experiments (Part 2)
├── report.pdf               # Written report with observations and error analysis
├── README                   # This file
└── hw3_speech_files/        # Speech data directory (not included in submission)
```

## Requirements

### Python Dependencies

- Python 3.8+
- numpy
- pandas
- matplotlib
- parselmouth (Python wrapper for Praat)
- scikit-learn

Install dependencies:
```bash
pip install numpy pandas matplotlib praat-parselmouth scikit-learn
```

### External Tools

- **openSMILE**: Required for IS09 feature extraction

Build openSMILE:
```bash
git clone https://github.com/audeering/opensmile.git
cd opensmile/
bash build.sh
```

## Running the Code

### Part 1: Feature Analysis

Open and run `feature_analysis.ipynb` in Jupyter Notebook/Lab.

This notebook:
1. Extracts 6 acoustic features per utterance using Parselmouth:
   - Min, max, mean pitch (75-600 Hz range, autocorrelation method)
   - Min, max, mean intensity (pitch floor 75 Hz)
2. Applies per-speaker z-score normalization
3. Generates 12 plots (6 features x 2 normalization states)

**Configuration:**
- Pitch range: 75-600 Hz
- Pitch method: autocorrelation
- Intensity pitch floor: 75 Hz
- Audio channel: left channel (channel 1)

**Output files:**
- `parselmouth_features.csv`: Raw extracted features
- `parselmouth_normalized_features.csv`: Z-score normalized features

### Part 2: Classification

Open and run `classification.ipynb` in Jupyter Notebook/Lab.

This notebook:
1. Extracts IS09 emotion features using openSMILE (384 features)
2. Combines with Parselmouth prosodic features (6 features)
3. Applies per-speaker z-score normalization
4. Performs leave-one-speaker-out cross-validation (7 folds)
5. Trains and evaluates classifiers (SVM and MLP)

**Classifier configurations:**
- SVM: RBF kernel, C=10, gamma='scale'
- MLP: hidden layers (256, 64), ReLU activation, early stopping

**Output files:**
- `opensmile_features/`: Directory containing per-utterance openSMILE features
- `opensmile_is09_features.csv`: Combined IS09 features for all utterances

## Normalization Method

Per-speaker z-score normalization is applied to remove speaker-specific differences:

1. For each speaker, concatenate all frame-level pitch/intensity values from all utterances
2. Compute global mean (mu) and standard deviation (sigma) per speaker
3. Normalize each frame: z = (x - mu) / sigma
4. Compute min/max/mean from normalized values

This method controls for inter-speaker variability while preserving emotion-driven acoustic variations.

## Evaluation Metrics

- Per-speaker accuracy and weighted F1 score
- Aggregated metrics computed as weighted averages:
  - Aggregated accuracy = sum(acc_i * n_i) / sum(n_i)
  - Aggregated F1 = sum(F1_i * n_i) / sum(n_i)

Where acc_i and F1_i are per-speaker metrics, and n_i is the number of test samples.

## Notes

- The `hw3_speech_files/` directory containing WAV files should be placed in the same directory as the notebooks
- openSMILE should be cloned and built in the same directory (path: `./opensmile/`)
- All outputs are retained in the notebooks for reproducibility
- Run cells in sequential order to reproduce results


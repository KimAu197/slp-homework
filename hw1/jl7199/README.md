# HW1 — Homework 1: Speech Analysis

## Directory Structure

bonus.Manipulation     ← Praat manipulation file (for bonus problem)

bonus.wav               ← Modified “happy” version of neutral utterance

feature_extraction.py   ← Python script to extract features

my_features.csv         ← CSV output for my recordings

msp_features.csv        ← CSV output for MSP podcast samples

my_samples/             ← folder containing my recorded WAV files

MSP_samples/            ← (not include but essential for .py) folder containing MSP podcast samples

Report.pdf               ← my written report for quesiton 2, 4, 5

## How to Run the Feature Extraction Script

1. installed the required Python package:
   ```bash
   pip install praat-parselmouth 
   ```
   
2. Place my_samples/ folder and MSP_samples/ folder (if applicable) in the same directory as feature_extraction.py. (must be the same name)

3. Run the script:
    ```bash
    python feature_extraction.py
    ```
The script will look for two directories: my_samples and MSP_samples. It will generate or overwrite my_features.csv and msp_features.csv based on the WAV files in those folders.

## References

Feature extraction is implemented via parselmouth.praat.call(...) commands in Python to mirror Praat operations. For intensity, jitter, shimmer, and HNR, parameter settings follow the course assignment specification. Speaking rate is computed as #words/duration (word counts were counted manually for both datasets). For bonus problems, follow the slices of week 4.

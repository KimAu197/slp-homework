# Dialogue Act Recognition (HW2)

This repository contains the notebooks I used for Homework 2 on dialogue act recognition. The workflow has two main stages:

1. `features.ipynb` extracts text and speech features from the provided Switchboard-style transcripts and audio.
2. `classification.ipynb` trains/evaluates dialogue-act classifiers (speech-only, text-only, and multimodal) and produces submission files.

## Prerequisites

- Python 3.9+ with Jupyter (Notebook or Lab).
- Create/activate a virtual environment, then install the required libraries:

```bash
pip install pandas numpy scikit-learn nltk torch torchaudio transformers librosa soundfile praat-parselmouth seaborn matplotlib
```

NLTK tokenizers and Praat-Parselmouth need the standard English models; the notebook downloads them automatically if missing.

## Data Layout

Place the course-provided data under `data/`:

- `data/train.csv`, `data/valid.csv`, `data/test.csv` – metadata + transcripts.
- `data/wav/` – raw audio segments (unzipped from `wav.zip` if necessary).

The notebooks assume this directory structure via the `DATA_DIR = "./data"` constant.

## Running the Pipeline

1. **Launch Jupyter** in the repo root and open `features.ipynb`.
2. Execute the notebook top-to-bottom. It will:
   - Build text features (surface stats, POS distributions, LIWC counts, TF‑IDF, DistilBERT embeddings).
   - Build speech features (pitch/intensity stats, jitter/shimmer, MFCCs, WavLM embeddings).
   - Write the engineered features to CSV/NumPy files such as `speech_features_[train|valid|test].csv` and cache BERT/WavLM embeddings under `cache_embed/`.
3. Open `classification.ipynb` and run all cells to:
   - Train the speech-only Random Forest, text-only linear SVM (SGDClassifier), and joint multimodal MLP.
   - Save the best joint model weights to `best_joint_mlp.pt`.
   - Generate predictions for the speech/text/multimodal test splits (`test_<UNI>_speech.csv`, etc.).
4. The notebook also renders the required evaluation plots (confusion matrices) and error analysis discussion.

## Reproducing Results

- To reuse a trained multimodal model, keep `best_joint_mlp.pt` in the repo root; the notebook reloads it when `torch.load` is called.

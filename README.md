# COMS6706W Advanced Spoken Language Processing - Homework

**Course:** COMS6706W Advanced Spoken Language Processing  
**Semester:** Fall 2025

This repository contains three homework assignments focusing on speech analysis, dialogue act recognition, and emotion recognition in speech.

---

## 📁 Repository Structure

```
homework/
├── hw1/                    # Speech Analysis
├── hw2/                    # Dialogue Act Recognition
├── hw3/                    # Emotion Recognition in Speech
└── README.md              # This file
```

---

## 📝 Assignment Overview

### HW1: Speech Analysis

**Topics:** Acoustic feature extraction, prosodic analysis, speech manipulation

**Key Features:**
- Extraction of acoustic-prosodic features (pitch, intensity, jitter, shimmer, HNR)
- Analysis of emotional speech samples from MSP Podcast corpus
- Personal speech recordings across 7 emotion categories
- Praat-based speech manipulation (bonus)

**Technologies:** Python, Parselmouth (Praat wrapper), NumPy, Pandas

**Deliverables:**
- Feature extraction script (`feature_extraction.py`)
- CSV files with extracted features
- Written report analyzing acoustic patterns
- Modified audio samples

[→ See detailed HW1 README](hw1/jl7199/README.md)

---

### HW2: Dialogue Act Recognition

**Topics:** Multimodal classification, feature engineering, neural networks

**Key Features:**
- Text feature extraction (surface statistics, POS tagging, LIWC, TF-IDF, DistilBERT embeddings)
- Speech feature extraction (pitch/intensity statistics, jitter/shimmer, MFCCs, WavLM embeddings)
- Three classification approaches:
  - Speech-only (Random Forest)
  - Text-only (Linear SVM)
  - Multimodal (MLP neural network)
- Leave-one-speaker-out cross-validation

**Technologies:** Python, PyTorch, Transformers (HuggingFace), scikit-learn, librosa, Parselmouth, NLTK

**Dataset:** Switchboard-style dialogue corpus with dialogue act annotations

**Deliverables:**
- Feature extraction notebook (`features.ipynb`)
- Classification notebook (`classification.ipynb`)
- Trained models and predictions
- Performance analysis and error analysis

[→ See detailed HW2 README](hw2/jl7199/task1/README.md)

---

### HW3: Emotion Recognition in Speech

**Topics:** Emotion classification, openSMILE features, speaker normalization

**Key Features:**
- Parselmouth-based prosodic feature extraction (pitch, intensity)
- openSMILE IS09 emotion feature set (384 features)
- Per-speaker z-score normalization
- Leave-one-speaker-out cross-validation (7 speakers)
- Multiple classifiers (SVM with RBF kernel, MLP)

**Technologies:** Python, Parselmouth, openSMILE, scikit-learn, NumPy, Pandas, Matplotlib

**Dataset:** Emotional Prosody Speech and Transcripts corpus (2,324 utterances, 15 emotion classes)

**Deliverables:**
- Feature analysis notebook (`feature_analysis.ipynb`)
- Classification notebook (`classification.ipynb`)
- Extracted features (CSV files)
- Written report with visualizations and error analysis

[→ See detailed HW3 README](hw3/jl7199/README.md)

---

## 🛠️ Common Dependencies

### Python Environment

All assignments require Python 3.8+ with Jupyter Notebook/Lab. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Core Libraries

```bash
# Common across all assignments
pip install numpy pandas matplotlib scikit-learn

# Speech processing
pip install praat-parselmouth librosa soundfile torchaudio

# NLP and deep learning
pip install torch transformers nltk

# Additional utilities
pip install seaborn jupyter
```

### External Tools

**openSMILE** (required for HW3):
```bash
git clone https://github.com/audeering/opensmile.git
cd opensmile/
bash build.sh
```

---

## 🚀 Quick Start Guide

### HW1: Speech Analysis

```bash
cd hw1/jl7199
pip install praat-parselmouth
python feature_extraction.py
```

### HW2: Dialogue Act Recognition

```bash
cd hw2/jl7199/task1
jupyter notebook features.ipynb      # Extract features first
jupyter notebook classification.ipynb # Then train classifiers
```

### HW3: Emotion Recognition

```bash
cd hw3/jl7199
jupyter notebook feature_analysis.ipynb  # Part 1: Feature extraction
jupyter notebook classification.ipynb    # Part 2: Classification
```

---

## 📊 Key Results Summary

### HW1: Speech Analysis
- Successfully extracted 7 acoustic features across 7 emotion categories
- Identified emotion-specific acoustic patterns (e.g., higher pitch in happy/surprised, lower in sad)
- Implemented Praat-based speech manipulation for emotional transformation

### HW2: Dialogue Act Recognition
- **Speech-only model:** Random Forest with prosodic + WavLM features
- **Text-only model:** Linear SVM with TF-IDF + DistilBERT embeddings
- **Multimodal model:** MLP combining speech and text features (best performance)
- Achieved strong performance on Switchboard dialogue act classification

### HW3: Emotion Recognition
- Extracted 390 features (6 Parselmouth + 384 openSMILE IS09)
- Applied per-speaker z-score normalization to control for speaker variability
- **SVM (RBF kernel):** Best performance with C=10, gamma='scale'
- **MLP:** Hidden layers (256, 64) with early stopping
- Achieved competitive results on 15-class emotion recognition task

---

## 📖 Learning Outcomes

Through these assignments, I gained hands-on experience with:

1. **Acoustic-Prosodic Analysis**
   - Feature extraction using Praat/Parselmouth
   - Understanding pitch, intensity, voice quality measures
   - Speech manipulation and synthesis

2. **Feature Engineering**
   - Text features: surface statistics, POS tagging, LIWC, TF-IDF
   - Speech features: MFCCs, spectral features, prosodic statistics
   - Pre-trained embeddings: DistilBERT, WavLM

3. **Machine Learning for Speech**
   - Traditional ML: Random Forest, SVM, feature selection
   - Deep learning: MLP, PyTorch implementation
   - Cross-validation strategies for speaker-independent evaluation

4. **Multimodal Fusion**
   - Early fusion of speech and text modalities
   - Handling heterogeneous feature spaces
   - Balancing modality contributions

5. **Evaluation and Analysis**
   - Performance metrics: accuracy, F1-score, confusion matrices
   - Error analysis and failure case examination
   - Speaker normalization techniques

---

## 📚 References

### Tools and Libraries
- **Parselmouth:** Python wrapper for Praat ([GitHub](https://github.com/YannickJadoul/Parselmouth))
- **openSMILE:** Open-source toolkit for audio feature extraction ([GitHub](https://github.com/audeering/opensmile))
- **Transformers:** HuggingFace library for pre-trained models ([Docs](https://huggingface.co/docs/transformers))
- **librosa:** Python library for audio analysis ([Docs](https://librosa.org/))

### Datasets
- **MSP Podcast:** Emotional speech corpus (HW1)
- **Switchboard:** Dialogue corpus with dialogue act annotations (HW2)
- **Emotional Prosody Speech and Transcripts:** Multi-speaker emotion corpus (HW3)

### Key Papers
- Eyben et al. (2009): "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)"
- Schuller et al. (2009): "The INTERSPEECH 2009 Emotion Challenge"
- Chen et al. (2022): "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"



*Last Updated: January 2026*

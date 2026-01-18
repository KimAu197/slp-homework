import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 下载 NLTK 必需资源
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')


DATA_DIR = "./data"
OUT_DIR = "./task1"

ID_COLS = ["dialog_id", "speaker", "da_tag", "start_time", "end_time"]
TEXT_COL = "transcript"   # 如果你数据叫 utterance/text，自行改掉这里

# TF-IDF 参数
NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 2000  
nltk.data.path.append("/Users/kenzieluo/nltk_data")




from nltk import word_tokenize, pos_tag

def structural_features(text):
    tokens = word_tokenize(text)
    return {
        "len_chars": len(text),
        "len_tokens": len(tokens),
        "num_punct": sum(1 for c in text if c in ".,?!;:"),
        "ends_question": int(text.strip().endswith("?")),
    }

def pos_features(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    buckets = {"NOUN":0, "VERB":0, "ADJ":0, "ADV":0}
    for _, tag in tagged:
        if tag.startswith("NN"): buckets["NOUN"] += 1
        elif tag.startswith("VB"): buckets["VERB"] += 1
        elif tag.startswith("JJ"): buckets["ADJ"] += 1
        elif tag.startswith("RB"): buckets["ADV"] += 1

    total = len(tokens) if len(tokens) > 0 else 1
    return {f"pos_prop_{k}": buckets[k]/total for k in buckets}




train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
valid = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    lowercase=True,
    strip_accents="unicode"
)

# Fit TF-IDF on train
tfidf_train = vectorizer.fit_transform(train[TEXT_COL])
tfidf_valid = vectorizer.transform(valid[TEXT_COL])
tfidf_test  = vectorizer.transform(test[TEXT_COL])

tfidf_cols = [f"tfidf_{v}" for v in vectorizer.get_feature_names_out()]


def extract_interpretable(df):
    all_feats = []
    for text in df[TEXT_COL].fillna(""):
        feats = {}
        feats.update(structural_features(text))
        feats.update(pos_features(text))
        all_feats.append(feats)
    return pd.DataFrame(all_feats)

interp_train = extract_interpretable(train)
interp_valid = extract_interpretable(valid)
interp_test  = extract_interpretable(test)



liwc_cols = train.columns[ train.columns.get_loc("end_time")+1: ]
def assemble(df, interp_df, tfidf_matrix):
    base = df[ID_COLS].reset_index(drop=True)
    liwc = df[liwc_cols].reset_index(drop=True)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)

    return pd.concat([base, interp_df, liwc, tfidf_df], axis=1)

text_train = assemble(train, interp_train, tfidf_train)
text_valid = assemble(valid, interp_valid, tfidf_valid)
text_test  = assemble(test,  interp_test,  tfidf_test)

os.makedirs(OUT_DIR, exist_ok=True)

text_train.to_csv(os.path.join(OUT_DIR, "text_features_train.csv"), index=False)
text_valid.to_csv(os.path.join(OUT_DIR, "text_features_valid.csv"), index=False)
text_test.to_csv(os.path.join(OUT_DIR,  "text_features_test.csv"),  index=False)

print("Saved text feature files!")
print(text_train.shape, text_valid.shape, text_test.shape)




def load_segment(wav_path, start, end):
    snd = parselmouth.Sound(wav_path)
    # Praat 时间是秒
    return snd.extract_part(from_time=start, to_time=end, preserve_times=False)




def extract_speech_features(segment: parselmouth.Sound):
    feats = {}

    ### Pitch ###
    pitch_obj = call(segment, "To Pitch", 0.0, 75, 600)
    pitch_values = pitch_obj.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # remove unvoiced
    
    if len(pitch_values) > 0:
        feats["pitch_mean"] = np.mean(pitch_values)
        feats["pitch_std"]  = np.std(pitch_values)
        feats["pitch_min"]  = np.min(pitch_values)
        feats["pitch_max"]  = np.max(pitch_values)
    else:
        feats["pitch_mean"] = feats["pitch_std"] = feats["pitch_min"] = feats["pitch_max"] = 0.0

    ### Intensity ###a
    intensity_obj = call(segment, "To Intensity", 100.0, 0.0)
    intensity_values = intensity_obj.values[0]
    feats["intensity_mean"] = np.mean(intensity_values)
    feats["intensity_max"] = np.max(intensity_values)

    ### Jitter ###
    point_proc = call(segment, "To PointProcess (periodic, cc)", 75, 600)
    try:
        feats["jitter_local"] = call(point_proc, "Get jitter (local)", 0, 0, 75, 600, 1.3)
    except:
        feats["jitter_local"] = 0.0

    ### Shimmer ###
    try:
        feats["shimmer_local"] = call([segment, point_proc], "Get shimmer (local)", 0, 0, 75, 600, 1.3, 1.6)
    except:
        feats["shimmer_local"] = 0.0

    ### Harmonicity (HNR) ###
    try:
        hnr_obj = call(segment, "To Harmonicity (cc)", 0.01, 75, 600)
        feats["hnr_mean"] = hnr_obj.values.mean()
    except:
        feats["hnr_mean"] = 0.0

    ### MFCC ###
    mfcc_obj = call(segment, "To MFCC", 13, 0.025, 0.01, 50, 15000)
    for i in range(1, 14):  # MFCC1-MFCC13
        try:
            col = mfcc_obj.to_array()[i-1]
            feats[f"mfcc_{i}"] = float(np.mean(col))
        except:
            feats[f"mfcc_{i}"] = 0.0

    return feats


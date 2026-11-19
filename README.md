# Real-time ASL Sign Language Translator

Reads American Sign Language alphabet letters (A–Z) from the shape of a hand, using MediaPipe hand landmarks and a small neural network. Runs on a live webcam feed, or on a single photo in the web demo.

**Live demo:** coming soon · **Portfolio:** [janvikgabani.github.io](https://janvikgabani.github.io)

## How it works

1. **Data collection** (`data_collection/collect_data.py`) — MediaPipe Hands records 21 landmarks per frame from a webcam, labelled by keypress.
2. **Preprocessing** (`preprocessing/preprocess.py`) — each hand is normalised so the model sees shape rather than position: the wrist is moved to the origin, the hand is scaled to unit size, and left hands are mirrored onto right. That gives 63 numbers (21 landmarks × x/y/z).
3. **Training** (`training/train_model.py`) — a TensorFlow MLP, 256 → 128 → 64 → 26, with batch normalisation and dropout.
4. **Inference** — `realtime/realtime_translator.py` for live webcam use with prediction smoothing over a 7-frame window; `app.py` for the single-photo web demo.
5. **NumPy export** (`inference/export_numpy.py`) — folds the batch-norm layers into plain affine maps and saves the weights, so the deployed demo runs on NumPy alone with no deep-learning runtime. Verified to match Keras to within 2e-06.

## Results, and what they actually mean

| Split | Accuracy |
|---|---|
| Random 85/15 split | **1.000** |
| Capture-order split (whole contiguous runs held out) | **1.000** |
| Chance | 0.038 |

Both numbers are 100%, and both overstate how well this would work for a stranger.

The dataset is 2,081 frames, 80 per letter, recorded by **two people in a single session**. Frames were captured while holding a pose, so consecutive frames are near-duplicates — the mean distance between neighbouring frames of a letter is 0.18, against 0.38 for random frames of the same letter. A shuffled split therefore puts near-copies on both sides of the train/test line, which is why `training/evaluate_grouped.py` exists: it holds out the last contiguous quarter of every letter instead.

That the honest split also scores 100% says the real story is **separability, not model quality**: different letters sit far apart in normalised landmark space (mean distance 1.68) compared with the spread within one letter (0.38). Once MediaPipe has done the hard work of locating the hand, telling the ASL letters apart is close to linearly separable.

Expect noticeably worse accuracy on a new person, camera or lighting setup. The letters most likely to be confused are the ones that differ subtly in the fingers: **M / N / S / T**, and **R / U / V**.

**J and Z are signed with motion** in real ASL. This model classifies a single frame, so it only captures their static hand shape. Handling them properly needs a temporal model (LSTM or TCN) over a sequence of frames.

## Run the web demo locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce the whole pipeline

```bash
pip install -r requirements-train.txt

# 1. preprocess the committed landmark data
python preprocessing/preprocess.py --raw-dir data/raw --output-dir data/processed

# 2. train
python training/train_model.py --data data/processed/processed.npz --epochs 60

# 3. evaluate, both ways
python training/evaluate.py --data data/processed/processed.npz
python training/evaluate_grouped.py --data data/processed/processed.npz

# 4. export for the web demo
python inference/export_numpy.py

# 5. live webcam translator
python realtime/realtime_translator.py --camera 0 --tts
```

To collect more data (strongly recommended before trusting it on new users):

```bash
python data_collection/collect_data.py --camera 0   # press A–Z to label, Q to quit
```

## Layout

| Path | What it is |
|---|---|
| `app.py` | Streamlit photo demo |
| `data/raw/` | 2,081 labelled landmark frames (the training data) |
| `data_collection/` | Webcam landmark capture |
| `preprocessing/` | Normalisation into 63-value feature vectors |
| `training/` | Model definition, training, both evaluations |
| `inference/` | Keras predictor with smoothing, and the NumPy export |
| `realtime/` | Live webcam translator, plus a sentence-building variant |
| `utils/` | MediaPipe helpers and landmark normalisation |
| `docs/` | Technical documentation and report |
| `models/` | Trained model, label encoder, NumPy export |

## Tech

Python 3.12 · MediaPipe 0.10 (Hands) · TensorFlow 2.17 · scikit-learn · NumPy · Streamlit

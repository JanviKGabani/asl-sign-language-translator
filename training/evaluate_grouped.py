"""Honest evaluation with a capture-order split.

The raw data was recorded by holding each hand pose while frames were captured, so
consecutive frames of a letter are near-duplicates. A random train/test split puts
those near-duplicates on both sides and reports close to 100% accuracy, which says
nothing about new captures.

This script instead holds out the LAST contiguous fraction of every letter's frames,
so no held-out frame has a near-duplicate in training. It reports both numbers so the
gap between them is visible.
"""
import argparse, os, sys
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from training.model import build_mlp


def fit_and_score(X_tr, y_tr, X_te, y_te, n_classes, epochs, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = build_mlp(input_dim=X_tr.shape[1], num_classes=n_classes)
    model.fit(X_tr, tf.keras.utils.to_categorical(y_tr, n_classes),
              epochs=epochs, batch_size=64, verbose=0)
    pred = model.predict(X_te, verbose=0).argmax(1)
    return accuracy_score(y_te, pred), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/processed.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of each letter's LAST frames held out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y_raw = d["X"], d["y"]
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    n_classes = len(enc.classes_)

    # --- 1. random split (the optimistic, leaky number) ---
    tr, te = train_test_split(np.arange(len(X)), test_size=args.holdout,
                              stratify=y, random_state=args.seed)
    acc_random, _ = fit_and_score(X[tr], y[tr], X[te], y[te], n_classes, args.epochs, args.seed)

    # --- 2. capture-order split: last N% of each letter's frames held out ---
    tr, te = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]          # already in capture order
        cut = int(len(idx) * (1 - args.holdout))
        tr.extend(idx[:cut]); te.extend(idx[cut:])
    tr, te = np.array(tr), np.array(te)
    acc_order, pred = fit_and_score(X[tr], y[tr], X[te], y[te], n_classes, args.epochs, args.seed)

    print(f"Random split accuracy        : {acc_random:.4f}   <- leaks near-duplicate frames")
    print(f"Capture-order split accuracy : {acc_order:.4f}   <- the honest number")
    print(f"Chance level                 : {1/n_classes:.4f}")
    print(f"\nTrain {len(tr)} frames / test {len(te)} frames\n")
    print(classification_report(y[te], pred, target_names=enc.classes_, zero_division=0))

    out = os.path.join(os.path.dirname(args.data), "grouped_eval.txt")
    with open(out, "w") as f:
        f.write(f"random_split_accuracy={acc_random:.4f}\n")
        f.write(f"capture_order_split_accuracy={acc_order:.4f}\n")
        f.write(f"holdout_fraction={args.holdout}\n")
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()

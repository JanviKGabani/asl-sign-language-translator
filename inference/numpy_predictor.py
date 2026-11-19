"""Run the trained ASL classifier using NumPy only, with no TensorFlow dependency.

Mirrors the Keras graph built in training/model.py: Dense -> BatchNorm -> (Dropout)
repeated, then a softmax output. Dropout is the identity at inference and each
BatchNorm was folded into a scale and shift by inference/export_numpy.py.
"""
import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class NumpyASLModel:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.classes = [str(c) for c in z["classes"]]
        self.kinds = [str(k) for k in z["layer_kinds"]]
        self.names = [str(n) for n in z["layer_names"]]
        self.params = {k: z[k] for k in z.files if "__" in k}
        self.n_dense = sum(1 for k in self.kinds if k == "dense")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(np.asarray(X, dtype=np.float32))
        dense_seen = 0
        for kind, name in zip(self.kinds, self.names):
            if kind == "dense":
                h = h @ self.params[f"{name}__W"] + self.params[f"{name}__b"]
                dense_seen += 1
                if dense_seen == self.n_dense:      # final layer -> softmax
                    return _softmax(h)
                h = _relu(h)
            else:                                    # folded batch norm
                h = h * self.params[f"{name}__scale"] + self.params[f"{name}__shift"]
        raise RuntimeError("No output layer found in the exported model")

    def predict_top(self, feature: np.ndarray, k: int = 3):
        """Return the top-k (letter, probability) pairs for one 63-value feature vector."""
        probs = self.predict_proba(feature)[0]
        idx = np.argsort(probs)[::-1][:k]
        return [(self.classes[i], float(probs[i])) for i in idx]

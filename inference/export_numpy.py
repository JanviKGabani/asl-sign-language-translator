"""Export the trained Keras MLP to plain NumPy arrays.

At inference the network is just a fixed chain of affine maps: Dropout is the identity,
and BatchNormalization folds into a scale and shift. Exporting the weights lets the demo
run on NumPy alone, so the deployed app does not need TensorFlow at all.
"""
import argparse, os, pickle, sys
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/asl_mlp.keras")
    ap.add_argument("--encoder", default="models/label_encoder.pkl")
    ap.add_argument("--out", default="models/asl_mlp_numpy.npz")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    with open(args.encoder, "rb") as f:
        classes = pickle.load(f).classes_

    arrays, order = {}, []
    for layer in model.layers:
        w = layer.get_weights()
        if isinstance(layer, tf.keras.layers.Dense):
            order.append(("dense", layer.name))
            arrays[f"{layer.name}__W"], arrays[f"{layer.name}__b"] = w[0], w[1]
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            order.append(("bn", layer.name))
            gamma, beta, mean, var = w
            # fold into y = x * scale + shift
            scale = gamma / np.sqrt(var + layer.epsilon)
            arrays[f"{layer.name}__scale"] = scale
            arrays[f"{layer.name}__shift"] = beta - mean * scale

    np.savez_compressed(
        args.out,
        classes=np.array([str(c) for c in classes], dtype="U8"),
        layer_kinds=np.array([k for k, _ in order], dtype="U16"),
        layer_names=np.array([n for _, n in order], dtype="U64"),
        **arrays,
    )

    # verify the NumPy path matches Keras
    from inference.numpy_predictor import NumpyASLModel
    nm = NumpyASLModel(args.out)
    X = np.random.RandomState(0).randn(64, model.input_shape[1]).astype(np.float32)
    diff = np.abs(model.predict(X, verbose=0) - nm.predict_proba(X)).max()
    print(f"Saved {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")
    print(f"Max difference vs Keras on 64 random inputs: {diff:.2e}")
    assert diff < 1e-5, "NumPy export does not match Keras"
    print("Export verified.")


if __name__ == "__main__":
    main()

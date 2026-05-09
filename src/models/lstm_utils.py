"""
SIPT Pro v4 — LSTM Autoencoder utilities
Préparation des séquences, scoring, batch scoring.
"""
import numpy as np
import pandas as pd


def prepare_lstm_sequences(df: pd.DataFrame, meta: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    Construit les séquences d'entraînement à partir des clients normaux.
    Normalisation z-score par client.
    """
    normal_cids = meta[meta["is_fraudster"] == False]["client_id"].tolist()
    sequences   = []
    for cid in normal_cids:
        vals = (
            df[df["client_id"] == cid]
            .sort_values("timestamp")["consumption_kwh"]
            .values.astype(float)
        )
        std_c = vals.std()
        if std_c < 1e-6:
            continue
        vals_norm = (vals - vals.mean()) / std_c
        for i in range(len(vals_norm) - window_size + 1):
            sequences.append(vals_norm[i: i + window_size])
    X = np.array(sequences, dtype=np.float32)[..., np.newaxis]
    return X


def lstm_score(model, vals: np.ndarray, window_size: int) -> float | None:
    """
    Calcule le MSE de reconstruction LSTM pour une série 1D (fenêtre).
    Retourne None si modèle indisponible ou données insuffisantes.
    """
    if model is None or len(vals) < window_size:
        return None
    w = vals[-window_size:].astype(np.float32)
    std_w = w.std()
    if std_w < 1e-6:
        return 0.0
    w_norm = (w - w.mean()) / std_w
    seq    = w_norm.reshape(1, window_size, 1)
    try:
        recon = model.predict(seq, verbose=0)
        mse   = float(np.mean((w_norm - recon[0, :, 0]) ** 2))
        return mse
    except Exception:
        return None


def batch_lstm_score(model, threshold, windows: np.ndarray):
    """
    Calcule les MSE pour un batch de fenêtres (N, W).
    windows : np.array de shape (N, window_size)
    Retourne (mses, anomalies) où mses est un array 1D, anomalies un array bool.
    """
    if model is None or threshold is None:
        return np.zeros(len(windows)), np.zeros(len(windows), dtype=bool)
    
    # Récupérer la taille attendue par le modèle
    expected_len = model.input_shape[1]  # ex: 30
    actual_len = windows.shape[1]
    
    if actual_len != expected_len:
        # Ajuster la fenêtre : prendre les dernières valeurs attendues
        print(f"⚠ Ajustement taille fenêtre : {actual_len} -> {expected_len}")
        windows = windows[:, -expected_len:]  # tronquer
    
    # Normalisation z-score par fenêtre
    normed = []
    for w in windows:
        std_w = w.std()
        if std_w > 1e-6:
            normed.append((w - w.mean()) / std_w)
        else:
            normed.append(w * 0)  # constante -> tout à zéro
    X = np.array(normed, dtype=np.float32)[..., np.newaxis]  # (N, W, 1)

    recons = model.predict(X, batch_size=64, verbose=0)  # (N, W, 1)
    mses = np.mean((X[:, :, 0] - recons[:, :, 0]) ** 2, axis=1)
    anomalies = mses > threshold
    return mses, anomalies
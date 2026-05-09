"""
SIPT Pro v4 — Génération, Entraînement & Sauvegarde
Isolation Forest + LSTM Autoencoder · Détection hybride
Étendu : injection régionale, feeders, CUSUM-ready dataset.
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION RÉSEAU
# ─────────────────────────────────────────────────────────────────────────────
REGIONS = ["NORD", "SUD", "EST", "OUEST", "CENTRE"]

CLIENT_PROFILES = {
    "RESIDENTIEL": {
        "base_range": (3, 18), "weekend_factor": 1.25,
        "noise_std": 0.12, "seasonal": True,
        "description": "Foyer domestique – consommation modérée, sensible aux week-ends",
    },
    "INDUSTRIEL": {
        "base_range": (80, 250), "weekend_factor": 0.60,
        "noise_std": 0.04, "seasonal": False,
        "description": "Site industriel – forte consommation, chute le weekend",
    },
    "COMMERCIAL": {
        "base_range": (25, 90), "weekend_factor": 0.45,
        "noise_std": 0.09, "seasonal": True,
        "description": "Commerce/bureau – pic en semaine, quasi-nul le weekend",
    },
    "AGRICOLE": {
        "base_range": (8, 40), "weekend_factor": 1.05,
        "noise_std": 0.18, "seasonal": True,
        "description": "Exploitation agricole – consommation saisonnière",
    },
}

PROFILE_WEIGHTS = [0.55, 0.15, 0.20, 0.10]

FEATURE_NAMES = [
    "Conso moyenne (kWh/j)",
    "Volatilité (écart-type)",
    "Tendance linéaire (slope)",
    "Variation journalière moy",
    "Ratio 1ère/2ème période",
    "Amplitude max-min",
    "Coefficient de variation",
    "Jours très faible conso (<0.5 kWh)",
    "Ratio Weekend/Weekday",
    "Volatilité Weekend vs Weekday",
    "Autocorrélation temporelle (lag-1)",
    "Instabilité directionnelle",
]

WINDOW_SIZE          = 30   # fenêtre commune IF + LSTM
TECHNICAL_LOSS_RATIO = 0.05  # 5% pertes techniques normales
FEEDER_LOSS_RATIO    = 0.04  # 4% pertes feeder
FEEDERS_PER_REGION   = 3     # feeders par région


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION DATASET
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(n_clients: int = 300, days: int = 90, seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    records, client_meta = [], []
    profile_names = list(CLIENT_PROFILES.keys())

    for c in range(n_clients):
        profile_name = np.random.choice(profile_names, p=PROFILE_WEIGHTS)
        profile      = CLIENT_PROFILES[profile_name]
        region       = np.random.choice(REGIONS)
        base         = np.random.uniform(*profile["base_range"])
        is_fraudster = np.random.random() < 0.05
        fraud_mode   = np.random.choice(["bypass", "tamper", "inject"]) if is_fraudster else None
        client_id    = f"{profile_name[:3]}_{region[:3]}_{c:04d}"

        client_meta.append({
            "client_id":    client_id,
            "region":       region,
            "profile":      profile_name,
            "is_fraudster": is_fraudster,
            "fraud_mode":   fraud_mode if is_fraudster else "—",
        })

        for d in range(days):
            seasonal = 1.0
            if profile["seasonal"]:
                seasonal = 1.35 if (d // 30) % 6 <= 2 else 0.80
            dow   = d % 7
            wk    = profile["weekend_factor"] if dow >= 5 else 1.0
            noise = np.random.normal(1, profile["noise_std"])
            cons  = base * seasonal * wk * noise

            if is_fraudster and np.random.random() < 0.30:
                if fraud_mode == "bypass":
                    cons *= np.random.uniform(0.05, 0.30)
                elif fraud_mode == "tamper":
                    cons *= np.random.uniform(0.35, 0.65)
                else:
                    cons -= np.random.uniform(base * 0.5, base * 0.85)

            records.append({
                "timestamp":       pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "client_id":       client_id,
                "region":          region,
                "profile":         profile_name,
                "consumption_kwh": max(0.0, round(cons, 2)),
            })

    return pd.DataFrame(records), pd.DataFrame(client_meta)


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION TOPOLOGIE FEEDERS
# ─────────────────────────────────────────────────────────────────────────────
def generate_feeder_mapping(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Assigne chaque client à un feeder (REGION_FEEDER_N).
    2-3 feeders par région, répartition aléatoire équilibrée.
    """
    rows = []
    for region in REGIONS:
        region_clients = meta[meta["region"] == region]["client_id"].tolist()
        np.random.shuffle(region_clients)
        n_feeders = FEEDERS_PER_REGION
        feeder_ids = [f"{region}_F{k+1}" for k in range(n_feeders)]
        chunks = np.array_split(region_clients, n_feeders)
        for fid, chunk in zip(feeder_ids, chunks):
            for cid in chunk:
                rows.append({
                    "client_id": cid,
                    "feeder_id": fid,
                    "region":    region,
                })
    return pd.DataFrame(rows)


def generate_feeder_injection(df: pd.DataFrame, feeder_map: pd.DataFrame,
                               fraud_ratio: float = 0.08) -> pd.DataFrame:
    """
    Génère les données d'injection par feeder et par jour.
    Fraude piquage : réduction de l'injection sur certains feeders (~fraud_ratio des jours).
    """
    rows  = []
    dates = sorted(df["timestamp"].dt.date.unique())

    for feeder_id in feeder_map["feeder_id"].unique():
        clients = feeder_map[feeder_map["feeder_id"] == feeder_id]["client_id"].tolist()
        region  = feeder_map[feeder_map["feeder_id"] == feeder_id]["region"].iloc[0]

        # Décide si ce feeder a un piquage (15% des feeders)
        has_piquage = np.random.random() < 0.15
        piquage_days = set()
        if has_piquage:
            n_piquage = int(len(dates) * fraud_ratio)
            piquage_days = set(np.random.choice(range(len(dates)), n_piquage, replace=False))

        for d_idx, date in enumerate(dates):
            day_df    = df[df["timestamp"].dt.date == date]
            feeder_df = day_df[day_df["client_id"].isin(clients)]
            total     = feeder_df["consumption_kwh"].sum()
            losses    = total * FEEDER_LOSS_RATIO
            normal_inj = total + losses

            if has_piquage and d_idx in piquage_days:
                # Piquage : l'injection est réduite (énergie volée non comptabilisée)
                theft = normal_inj * np.random.uniform(0.12, 0.30)
                injection = max(0, normal_inj - theft)
            else:
                # Injection légèrement bruitée autour de la valeur normale
                injection = normal_inj * np.random.uniform(0.98, 1.02)

            rows.append({
                "date":         str(date),
                "feeder_id":    feeder_id,
                "region":       region,
                "injection_kwh": round(injection, 1),
                "total_conso":  round(total, 1),
                "n_clients":    len(clients),
            })

    return pd.DataFrame(rows)


def generate_region_injection(df: pd.DataFrame, feeder_injection: pd.DataFrame,
                               fraud_ratio: float = 0.06) -> pd.DataFrame:
    """
    Agrège les injections par région + introduit des piquages au niveau régional.
    """
    rows  = []
    dates = sorted(df["timestamp"].dt.date.unique())

    for region in REGIONS:
        has_regional_fraud = np.random.random() < 0.20
        fraud_days = set()
        if has_regional_fraud:
            n_f = int(len(dates) * fraud_ratio)
            fraud_days = set(np.random.choice(range(len(dates)), n_f, replace=False))

        for d_idx, date in enumerate(dates):
            region_df  = df[(df["timestamp"].dt.date == date) & (df["region"] == region)]
            total_conso = region_df["consumption_kwh"].sum()
            losses      = total_conso * TECHNICAL_LOSS_RATIO
            base_inj    = total_conso + losses

            if has_regional_fraud and d_idx in fraud_days:
                theft = base_inj * np.random.uniform(0.10, 0.25)
                injection = max(0, base_inj - theft)
            else:
                injection = base_inj * np.random.uniform(0.98, 1.02)

            rows.append({
                "date":         str(date),
                "region":       region,
                "injection_kwh": round(injection, 1),
                "total_conso":  round(total_conso, 1),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (fenêtre glissante — pour Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(s: np.ndarray) -> np.ndarray:
    s        = np.array(s, dtype=float)
    n        = len(s)
    wk_vals  = s[np.arange(n) % 7 >= 5]
    wd_vals  = s[np.arange(n) % 7 <  5]
    diff_s   = np.diff(s)
    mean_s   = np.mean(s)   if n > 0 else 0.0
    std_s    = np.std(s)    if n > 1 else 0.0
    trend    = np.polyfit(np.arange(n), s, 1)[0] if n >= 2 else 0.0
    h1       = np.mean(s[:n // 2]) if n >= 2 else mean_s
    h2       = np.mean(s[n // 2:]) if n >= 2 else mean_s
    mean_wk  = np.mean(wk_vals) if len(wk_vals) > 0 else mean_s
    mean_wd  = np.mean(wd_vals) if len(wd_vals) > 0 else mean_s
    std_wk   = np.std(wk_vals)  if len(wk_vals) > 1 else std_s
    std_wd   = np.std(wd_vals)  if len(wd_vals) > 1 else std_s
    if n > 2 and np.std(s[:-1]) > 0 and np.std(s[1:]) > 0:
        autocorr = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    else:
        autocorr = 0.0
    return np.array([
        mean_s,
        std_s,
        trend,
        np.mean(diff_s)  if n > 1 else 0.0,
        h1 / h2          if h2 > 0 else 1.0,
        np.max(s) - np.min(s),
        std_s / mean_s   if mean_s > 0 else 0.0,
        float(np.sum(s < 0.5)),
        mean_wk / mean_wd if mean_wd > 0 else 1.0,
        std_wk  / std_wd  if std_wd  > 0 else 1.0,
        autocorr,
        float(np.sum(diff_s[1:] * diff_s[:-1] < 0)) if n > 2 else 0.0,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LSTM AUTOENCODER
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm_autoencoder(window_size: int):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout,
    )
    inp     = Input(shape=(window_size, 1), name="seq_input")
    x       = LSTM(64, activation="tanh", return_sequences=True,  name="enc1")(inp)
    x       = Dropout(0.1)(x)
    encoded = LSTM(32, activation="tanh", return_sequences=False, name="enc2")(x)
    x       = RepeatVector(window_size, name="bottleneck")(encoded)
    x       = LSTM(32, activation="tanh", return_sequences=True,  name="dec1")(x)
    x       = Dropout(0.1)(x)
    x       = LSTM(64, activation="tanh", return_sequences=True,  name="dec2")(x)
    out     = TimeDistributed(Dense(1), name="reconstruction")(x)
    model   = Model(inp, out, name="LSTM_Autoencoder_SIPT")
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_lstm_sequences(df: pd.DataFrame, meta: pd.DataFrame, window_size: int):
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


def compute_lstm_threshold(model, X_normal: np.ndarray, percentile: float = 95.0) -> float:
    recons = model.predict(X_normal, batch_size=512, verbose=0)
    errors = np.mean((X_normal[:, :, 0] - recons[:, :, 0]) ** 2, axis=1)
    return float(np.percentile(errors, percentile))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  SIPT Pro v4 — Pipeline Isolation Forest + LSTM + Topologie")
    print("=" * 70)

    # ── 1. Génération ──────────────────────────────────────────────────────────
    print("\n[1/8] Génération du dataset (seed aléatoire)...")
    seed = int(time.time() * 1000) % 99999
    df, meta = generate_dataset(n_clients=300, days=90, seed=seed)
    df.to_csv("electricity_fraud_dataset.csv", index=False)
    meta.to_csv("client_metadata.csv", index=False)
    joblib.dump(seed, "dataset_seed.pkl")

    n_fraud = meta["is_fraudster"].sum()
    print(f"  ✓ {len(df):,} obs | {df['client_id'].nunique()} clients | seed={seed}")
    print(f"  ✓ {n_fraud} fraudeurs ({n_fraud / len(meta):.1%})")
    for p in CLIENT_PROFILES:
        cnt = (meta["profile"] == p).sum()
        print(f"     · {p:<12} : {cnt:>3} clients")

    # ── 2. Topologie Feeders ───────────────────────────────────────────────────
    print("\n[2/8] Génération topologie feeders...")
    feeder_map = generate_feeder_mapping(meta)
    feeder_map.to_csv("client_feeder_map.csv", index=False)
    print(f"  ✓ {feeder_map['feeder_id'].nunique()} feeders générés")

    feeder_injection = generate_feeder_injection(df, feeder_map)
    feeder_injection.to_csv("feeder_injection.csv", index=False)
    print(f"  ✓ feeder_injection.csv — {len(feeder_injection):,} entrées")

    region_injection = generate_region_injection(df, feeder_injection)
    region_injection.to_csv("region_injection.csv", index=False)
    print(f"  ✓ region_injection.csv — {len(region_injection):,} entrées")

    # ── 3. Features IF ─────────────────────────────────────────────────────────
    print(f"\n[3/8] Extraction features IF (fenêtre {WINDOW_SIZE}j)...")
    features_list, client_ids_feat = [], []
    for cid, grp in df.groupby("client_id"):
        vals = grp.sort_values("timestamp")["consumption_kwh"].values
        if len(vals) >= WINDOW_SIZE:
            features_list.append(extract_features(vals[-WINDOW_SIZE:]))
            client_ids_feat.append(cid)
    X = np.array(features_list)
    print(f"  ✓ {len(X)} vecteurs ({X.shape[1]} features)")

    # ── 4. Isolation Forest ────────────────────────────────────────────────────
    print("\n[4/8] Isolation Forest...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso      = IsolationForest(
        n_estimators=200, contamination=0.05,
        max_features=1.0, bootstrap=False,
        random_state=None, n_jobs=-1,
    )
    iso.fit(X_scaled)
    scores_if = iso.decision_function(X_scaled)
    thresh_if = np.percentile(scores_if, 5)
    print(f"  ✓ Seuil IF : {thresh_if:.4f} | {np.sum(scores_if < thresh_if)} anomalies")

    # ── 5. Sauvegarde IF ───────────────────────────────────────────────────────
    print("\n[5/8] Sauvegarde artefacts IF...")
    joblib.dump(iso,            "isolation_forest.pkl")
    joblib.dump(scaler,         "scaler.pkl")
    joblib.dump(FEATURE_NAMES,  "feature_names.pkl")
    joblib.dump(X_scaled[:100], "background_samples.pkl")
    feat_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
    feat_df.insert(0, "client_id", client_ids_feat)
    feat_df["score_IF"] = scores_if
    feat_df["label_IF"] = iso.predict(X_scaled)
    feat_df.to_csv("features_debug.csv", index=False)
    print("  ✓ isolation_forest.pkl | scaler.pkl | features_debug.csv")

    # ── 6. LSTM Autoencoder ────────────────────────────────────────────────────
    print(f"\n[6/8] LSTM Autoencoder (fenêtre {WINDOW_SIZE}j)...")
    lstm_ok = False
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        X_lstm = prepare_lstm_sequences(df, meta, WINDOW_SIZE)
        print(f"  ✓ {len(X_lstm)} séquences normales")

        lstm_model = build_lstm_autoencoder(WINDOW_SIZE)
        callbacks  = [
            EarlyStopping(monitor="val_loss", patience=5,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
        ]
        history = lstm_model.fit(
            X_lstm, X_lstm,
            epochs=60, batch_size=128,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1, shuffle=True,
        )
        val_loss    = min(history.history["val_loss"])
        lstm_thresh = compute_lstm_threshold(lstm_model, X_lstm, percentile=95.0)
        print(f"  ✓ Val loss : {val_loss:.6f} | Seuil LSTM (95e pct) : {lstm_thresh:.6f}")

        lstm_model.save("lstm_autoencoder.keras")
        joblib.dump(lstm_thresh, "lstm_threshold.pkl")
        print("  ✓ lstm_autoencoder.keras | lstm_threshold.pkl")
        lstm_ok = True

    except ImportError:
        print("  ⚠  TensorFlow absent → pip install tensorflow")
    except Exception as e:
        print(f"  ⚠  LSTM échoué : {e}")

    # ── 7. SHAP background ─────────────────────────────────────────────────────
    print("\n[7/8] Sauvegarde background SHAP (100 échantillons)...")
    joblib.dump(X_scaled[:100], "background_samples.pkl")
    print("  ✓ background_samples.pkl")

    # ── 8. Résumé ──────────────────────────────────────────────────────────────
    print("\n[8/8] Rapport final.")
    print("=" * 70)
    print(f"  IF      : ✓ seuil {thresh_if:.4f}")
    print(f"  LSTM    : {'✓' if lstm_ok else '✗  (lancer après pip install tensorflow)'}")
    print(f"  Feeders : ✓ {feeder_map['feeder_id'].nunique()} feeders | "
          f"{len(feeder_injection):,} injections")
    print(f"  Régions : ✓ {len(region_injection):,} entrées d'injection")
    print(f"  Seed    : {seed}")
    print("  → streamlit run streamlit_app.py")
    print("=" * 70)

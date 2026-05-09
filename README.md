# ⚡ SIPT Pro v4 — Système Intégré de Protection du Réseau Électrique

Plateforme hybride de détection de fraude électrique combinant **Isolation Forest**, **LSTM Autoencoder**, **CUSUM**, **bilan énergétique** et **topologie réseau** dans un dashboard SCADA haute visibilité.

---

## Architecture du projet

```
sipt_pro_v4/
├── train.py                        # Génération données + entraînement modèles
├── streamlit_app.py                # Entrypoint dashboard
├── requirements.txt
├── README.md
└── src/
    ├── config.py                   # Constantes globales (chemins, seuils, couleurs)
    ├── models/
    │   ├── loaders.py              # Chargement IF, LSTM, SHAP
    │   ├── isolation.py            # extract_features, shap_explain, combined_level, risk_pct
    │   └── lstm_utils.py           # prepare_lstm_sequences, lstm_score, batch_lstm_score
    ├── detection/
    │   ├── energy_balance.py       # EnergyBalanceDetector : bilan régional + prob piquage
    │   ├── cusum.py                # CUSUM : détection micro-dérives
    │   ├── topology.py             # TopologyFraudDetector : bilans feeders
    │   └── false_positive.py       # Away detection, capteur bloqué, qualité données, EMA
    ├── registry/
    │   └── manager.py              # CRUD registre JSON (détections, away, faults, FP)
    └── ui/
        ├── components.py           # CSS SCADA v4, kpi_grid, ticker, fraud_card
        └── pages/
            ├── vue_reseau.py       # Tab : Vue réseau & bilan régional
            ├── surveillance.py     # Tab : Simulation temps réel semaine glissante
            ├── fraudsters.py       # Tab : Fraudeurs avérés
            ├── suspects.py         # Tab : Suspects en cours
            ├── analytics.py        # Tab : Analytique + profil client + topologie
            ├── exclusions.py       # Tab : Away, pannes capteurs, faux positifs
            └── model_info.py       # Tab : Documentation pédagogique des modèles
```

---

## Installation

**Prérequis** : Python 3.9+

```bash
# 1. Cloner / extraire le projet
cd sipt_pro_v4

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

> **Note CPU-only** : Si vous n'avez pas de GPU, remplacez `tensorflow` par `tensorflow-cpu` dans `requirements.txt` avant l'installation.

---

## Génération du dataset et entraînement

```bash
python train.py
```

Ce script effectue en **8 étapes** :

| Étape | Action | Fichiers produits |
|-------|--------|-------------------|
| 1 | Génération dataset 300 clients × 90 jours | `electricity_fraud_dataset.csv`, `client_metadata.csv` |
| 2 | Topologie feeders + injections | `client_feeder_map.csv`, `feeder_injection.csv`, `region_injection.csv` |
| 3 | Extraction features Isolation Forest (12 features) | — |
| 4 | Entraînement Isolation Forest | — |
| 5 | Sauvegarde artefacts IF | `isolation_forest.pkl`, `scaler.pkl`, `features_debug.csv` |
| 6 | Entraînement LSTM Autoencoder (si TensorFlow présent) | `lstm_autoencoder.keras`, `lstm_threshold.pkl` |
| 7 | Background SHAP | `background_samples.pkl` |
| 8 | Rapport final | — |

---

## Lancement du dashboard

```bash
streamlit run streamlit_app.py
```

Le dashboard s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501).

---

## Système de détection hybride

### ① Isolation Forest (IF)
- **12 features** statistiques extraites sur une fenêtre glissante de 30 jours
- Score ∈ [-0.5, 0.5] — plus négatif = plus anormal
- Détecte : bypasses brutaux, tampering, injections

### ② LSTM Autoencoder
- Entraîné **uniquement sur des clients normaux**
- Fenêtre 30 jours, normalisation z-score par client
- Reconstruction → MSE élevée = pattern temporel anormal
- Architecture : Encoder LSTM(64→32) → Bottleneck → Decoder LSTM(32→64) → Dense(1)

### ③ CUSUM (micro-dérives)
- Algorithme CUSUM unilatéral (côté négatif)
- Paramètres : `k = 0.5σ`, `h = 5σ`
- Détecte les chutes lentes et persistantes (bypass progressif)
- Seul : niveau "warning" · Combiné à IF/LSTM : amplifie le risque

### ④ Bilan énergétique (piquage direct)
- Injection régionale vs consommation totale + pertes (5%)
- Probabilité piquage = `écart / (0.2 × injection)`, bornée à [0, 1]
- Seuils : Warning > 50%, Alerte > 70%

### ⑤ Topologie réseau (branchement avant compteur)
- Bilan par feeder (segment réseau : Région → Feeder → Clients)
- Alerte persistante si écart > 12% sur ≥ 3 jours consécutifs

### Score hybride combiné
```
Risk = IF × 50% + LSTM × 35% + CUSUM × 15%
```

| Combinaison | Niveau | Couleur |
|-------------|--------|---------|
| IF + LSTM ou IF + CUSUM | DOUBLE ALERTE | 🔴 Rouge |
| IF seul | ALERTE IF | 🔴 Rouge clair |
| LSTM seul | SUSPECT LSTM | 🟡 Orange |
| CUSUM seul | MICRO-DÉRIVE | 🟣 Violet |

---

## Gestion des faux positifs

| Mécanisme | Déclenchement | Effet |
|-----------|--------------|-------|
| **Away auto** | Conso < 0.5 kWh × 7 jours consécutifs | Alertes suspendues |
| **Away manuel** | Opérateur (tab Exclusions) | Alertes suspendues |
| **Capteur bloqué auto** | Variance < 0.02 ou valeurs constantes | Skip IF/LSTM |
| **Capteur bloqué manuel** | Opérateur (tab Exclusions) | Skip IF/LSTM |
| **Qualité données** | < 70% valeurs valides dans la fenêtre | Skip détection |
| **Baseline EMA** | α = 0.05 (adaptation lente du profil) | Réduction FP long-terme |

---

## Registre persistant

Le fichier `fraud_registry.json` stocke par client :
- Historique des détections (timestamp, niveau, risque %)
- Statut confirmé (≥ 2 détections → **AVÉRÉ**)
- Périodes d'absence
- Pannes capteurs déclarées
- Journal des faux positifs
- Baseline EMA

Le registre **survit aux redémarrages** du dashboard.

---

## Interface SCADA v4

Thème haute visibilité (fond `#0A0F1A`) :
- 🔴 **Alerte** `#FF4C4C`
- 🟡 **Warning** `#FFB347`
- 🔵 **Info** `#00E5FF`
- 🟢 **OK** `#00FFA5`
- 🟣 **CUSUM** `#A855F7`

Polices : `Share Tech Mono` (SCADA) + `Inter` (corps de texte).

**Mode Présentation** : toggle sidebar → KPIs et textes agrandis pour projection.

---

## Onglets du dashboard

| Onglet | Contenu |
|--------|---------|
| 🌐 Vue Réseau | Bilan énergétique par région, prob. piquage, tendance historique |
| 📡 Surveillance | Simulation semaine glissante, détection temps réel, ticker alertes |
| 🔴 Fraudeurs | Registre confirmé avec SHAP, historique détections |
| ⚡ Suspects | Clients en cours d'investigation |
| 📊 Analytique | Distributions, profil client, vue topologie feeders |
| 🛡 Exclusions | Away periods, pannes capteurs, journal faux positifs |
| ℹ️ Modèles | Documentation pédagogique architecture hybride |

"""
SIPT Pro v4 — Détecteur de fraude topologique (branchement avant compteur)
Analyse les bilans au niveau des feeders (segments réseau).
"""
import numpy as np
import pandas as pd

FEEDER_LOSS_RATIO   = 0.04    # 4% pertes techniques feeder
FEEDER_ALERT_DAYS   = 3       # jours persistants pour alerte
FEEDER_BALANCE_THR  = 0.12    # écart > 12% → suspect


class TopologyFraudDetector:
    """
    Analyse le bilan énergétique par feeder (segment réseau).
    Un feeder qui présente un déficit persistant > seuil est suspect.
    """

    def compute_feeder_balance(
        self,
        feeder_id: str,
        clients: list,
        df_day: pd.DataFrame,
        injection_kwh: float,
    ) -> dict:
        """
        Calcule le bilan d'un feeder pour une journée.
        """
        feeder_df = df_day[df_day["client_id"].isin(clients)]
        total     = feeder_df["consumption_kwh"].sum()
        losses    = total * FEEDER_LOSS_RATIO
        expected  = total + losses
        deviation = injection_kwh - expected
        ratio     = deviation / injection_kwh if injection_kwh > 0 else 0.0
        return {
            "feeder_id":    feeder_id,
            "n_clients":    len(clients),
            "total_conso":  round(total, 1),
            "losses":       round(losses, 1),
            "expected":     round(expected, 1),
            "injection":    round(injection_kwh, 1),
            "deviation":    round(deviation, 1),
            "ratio":        round(ratio, 4),
            "suspicious":   ratio > FEEDER_BALANCE_THR,
        }

    def evaluate_feeders(
        self,
        feeder_map: pd.DataFrame,
        feeder_injection: pd.DataFrame,
        df: pd.DataFrame,
        n_recent_days: int = 7,
    ) -> pd.DataFrame:
        """
        Évalue tous les feeders sur les n_recent_days derniers jours.
        Retourne un DataFrame avec les statistiques par feeder.
        """
        if feeder_map is None or feeder_map.empty:
            return pd.DataFrame()

        all_dates = sorted(df["timestamp"].dt.date.unique())[-n_recent_days:]
        results   = []

        for feeder_id in feeder_map["feeder_id"].unique():
            clients  = feeder_map[feeder_map["feeder_id"] == feeder_id]["client_id"].tolist()
            region   = feeder_map[feeder_map["feeder_id"] == feeder_id]["region"].iloc[0]

            # Injection du feeder sur les jours récents
            if feeder_injection is not None and not feeder_injection.empty:
                inj_rows = feeder_injection[feeder_injection["feeder_id"] == feeder_id]
            else:
                inj_rows = pd.DataFrame()

            daily_results = []
            for date in all_dates:
                day_df = df[df["timestamp"].dt.date == date]
                # Injection par défaut si absente
                if not inj_rows.empty:
                    inj_row = inj_rows[inj_rows["date"] == str(date)]
                    inj_kwh = float(inj_row["injection_kwh"].iloc[0]) if not inj_row.empty else \
                              day_df[day_df["client_id"].isin(clients)]["consumption_kwh"].sum() * 1.08
                else:
                    inj_kwh = day_df[day_df["client_id"].isin(clients)]["consumption_kwh"].sum() * 1.08

                daily_results.append(self.compute_feeder_balance(feeder_id, clients, day_df, inj_kwh))

            if not daily_results:
                continue

            # Agrégation sur n_recent_days
            avg_ratio   = float(np.mean([r["ratio"]     for r in daily_results]))
            avg_dev     = float(np.mean([r["deviation"]  for r in daily_results]))
            n_suspicious = sum(1 for r in daily_results if r["suspicious"])

            results.append({
                "feeder_id":     feeder_id,
                "region":        region,
                "n_clients":     len(clients),
                "avg_ratio":     round(avg_ratio, 4),
                "avg_deviation": round(avg_dev, 1),
                "n_suspicious":  n_suspicious,
                "persistent_alert": n_suspicious >= FEEDER_ALERT_DAYS,
                "clients":       clients,
            })

        return pd.DataFrame(results) if results else pd.DataFrame()

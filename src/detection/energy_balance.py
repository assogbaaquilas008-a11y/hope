"""
SIPT Pro v4 — Détecteur de bilan énergétique (piquage direct sur ligne)
Calcule l'écart entre injection régionale et consommation totale + pertes.
"""
import numpy as np
import pandas as pd
from typing import Optional


TECHNICAL_LOSS_RATIO = 0.05   # 5% pertes techniques normales


class EnergyBalanceDetector:
    """
    Détecte les piquages directs sur ligne en comparant :
      injection régionale (kWh) ↔ somme des consommations + pertes techniques
    """

    def compute_region_balance(
        self,
        region: str,
        df_day: pd.DataFrame,
        injection_kwh: float,
    ) -> dict:
        """
        Pour une région et un jour donné :
          - total_consumption = somme des conso clients de la région
          - expected_injection = total_consumption / (1 - loss_ratio)
          - balance_deviation = injection_kwh - (total_consumption + losses)
          - piquage_prob = min(1, deviation / (injection * 0.2))
        """
        clients_in_region = df_day[df_day["region"] == region]
        total_consumption = clients_in_region["consumption_kwh"].sum()
        losses            = total_consumption * TECHNICAL_LOSS_RATIO
        expected          = total_consumption + losses

        deviation     = injection_kwh - expected   # positif = manque d'énergie comptabilisée
        rel_deviation = deviation / injection_kwh if injection_kwh > 0 else 0.0
        piquage_prob  = float(np.clip(deviation / (injection_kwh * 0.20), 0.0, 1.0)) if injection_kwh > 0 else 0.0

        return {
            "region":            region,
            "total_consumption": round(total_consumption, 1),
            "losses":            round(losses, 1),
            "expected":          round(expected, 1),
            "injection":         round(injection_kwh, 1),
            "balance_deviation": round(deviation, 1),
            "rel_deviation":     round(rel_deviation, 4),
            "piquage_prob":      round(piquage_prob, 4),
            "alert":             piquage_prob > 0.70,
            "warning":           piquage_prob > 0.50,
        }

    def compute_historical_balances(
        self,
        df: pd.DataFrame,
        region_injection: pd.DataFrame,
        region: str,
        n_days: int = 30,
    ) -> pd.DataFrame:
        """
        Calcule l'historique des déviations pour une région sur n_days jours.
        Utilisé pour afficher la tendance dans Vue Réseau.
        """
        records = []
        if region_injection is None or region_injection.empty:
            return pd.DataFrame()

        region_inj = region_injection[region_injection["region"] == region].copy()
        region_inj = region_inj.sort_values("date").tail(n_days)

        for _, row in region_inj.iterrows():
            day_df  = df[df["timestamp"].dt.date == pd.Timestamp(row["date"]).date()]
            result  = self.compute_region_balance(region, day_df, row["injection_kwh"])
            result["date"] = row["date"]
            records.append(result)

        return pd.DataFrame(records) if records else pd.DataFrame()

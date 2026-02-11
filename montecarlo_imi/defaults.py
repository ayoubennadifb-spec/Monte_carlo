from __future__ import annotations

import pandas as pd

from .simulation import MaterialPolicy, Seasonality, SimulationConfig

DEFAULT_CONFIG = SimulationConfig(
    horizon_months=24,
    sigma_prod=0.10,
    n_iter=3000,
    seed=42,
    annual_min=11000.0,
    annual_max=15000.0,
    p_delay_1=0.15,
    p_delay_2=0.05,
    delay_1=1,
    delay_2=2,
)

DEFAULT_SEASONALITY = Seasonality(hiver=0.23, printemps=0.24, ete=0.26, automne=0.27)


def _default_materials() -> list[dict]:
    return [
        {
            "name": "MAP",
            "is_internal": True,
            "x": 0.40,
            "review_period": 1,
            "ss_months": 0.5,
            "lt_min": 0.0,
            "lt_mode": 0.0,
            "lt_max": 1.0,
        },
        {
            "name": "SA",
            "is_internal": False,
            "x": 0.565,
            "review_period": 1,
            "ss_months": 1.0,
            # Observé: 2–3 jours (approx. en mois: jours / 30)
            "lt_min": 2.0 / 30.0,
            "lt_mode": 2.5 / 30.0,
            "lt_max": 3.0 / 30.0,
        },
        {
            "name": "Mica",
            "is_internal": False,
            "x": 0.02,
            "review_period": 3,
            "ss_months": 2.0,
            # Observé: 2–3 jours (approx. en mois: jours / 30)
            "lt_min": 2.0 / 30.0,
            "lt_mode": 2.5 / 30.0,
            "lt_max": 3.0 / 30.0,
        },
        {
            "name": "Silice",
            "is_internal": False,
            "x": 0.005,
            "review_period": 3,
            "ss_months": 3.0,
            "lt_min": 1.0,
            "lt_mode": 2.0,
            "lt_max": 4.0,
        },
        {
            "name": "Huile",
            "is_internal": False,
            "x": 0.01,
            "review_period": 3,
            "ss_months": 2.0,
            # Observé: 30–40 jours (approx. en mois: jours / 30)
            "lt_min": 30.0 / 30.0,
            "lt_mode": 35.0 / 30.0,
            "lt_max": 40.0 / 30.0,
        },
    ]


DEFAULT_MATERIALS_DF = pd.DataFrame(_default_materials())


def materials_df_to_policies(df: pd.DataFrame) -> tuple[list[MaterialPolicy], set[str]]:
    df = df.reset_index(drop=True)
    materials: list[MaterialPolicy] = []
    internal: set[str] = set()

    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if bool(row.get("is_internal", False)):
            internal.add(name)
        materials.append(
            MaterialPolicy(
                name=name,
                x=float(row["x"]),
                review_period=int(row["review_period"]),
                ss_months=float(row["ss_months"]),
                lt_min=float(row["lt_min"]),
                lt_mode=float(row["lt_mode"]),
                lt_max=float(row["lt_max"]),
                stock_max_months=float(row.get("stock_max_months", 0.0)),
            )
        )

    return materials, internal

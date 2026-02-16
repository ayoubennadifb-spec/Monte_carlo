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
            "name": "SA",
            "x": 0.565,
            "review_period": 1,
            "ss_months": 1.0,
            "source": "local",
            # LT local observé: 2–3 jours (approx. en mois: jours / 30)
            "lt_min_local": 2.0 / 30.0,
            "lt_mode_local": 2.5 / 30.0,
            "lt_max_local": 3.0 / 30.0,
            # LT import: à renseigner (par défaut = local)
            "lt_min_import": 2.0 / 30.0,
            "lt_mode_import": 2.5 / 30.0,
            "lt_max_import": 3.0 / 30.0,
            "cost_local": 0.0,
            "cost_import": 0.0,
        },
        {
            "name": "Mica",
            "x": 0.02,
            "review_period": 3,
            "ss_months": 2.0,
            "source": "local",
            # LT local observé: 2–3 jours
            "lt_min_local": 2.0 / 30.0,
            "lt_mode_local": 2.5 / 30.0,
            "lt_max_local": 3.0 / 30.0,
            "lt_min_import": 2.0 / 30.0,
            "lt_mode_import": 2.5 / 30.0,
            "lt_max_import": 3.0 / 30.0,
            "cost_local": 0.0,
            "cost_import": 0.0,
        },
        {
            "name": "Silice",
            "x": 0.005,
            "review_period": 3,
            "ss_months": 3.0,
            "source": "import",
            # Silice (fumed silica): à paramétrer (valeurs par défaut indicatives)
            "lt_min_local": 1.0,
            "lt_mode_local": 2.0,
            "lt_max_local": 4.0,
            "lt_min_import": 1.0,
            "lt_mode_import": 2.0,
            "lt_max_import": 4.0,
            "cost_local": 0.0,
            "cost_import": 0.0,
        },
        {
            "name": "Huile",
            "x": 0.01,
            "review_period": 3,
            "ss_months": 2.0,
            "source": "import",
            # LT local observé: 30–40 jours (approx. en mois: jours / 30)
            "lt_min_local": 30.0 / 30.0,
            "lt_mode_local": 35.0 / 30.0,
            "lt_max_local": 40.0 / 30.0,
            "lt_min_import": 30.0 / 30.0,
            "lt_mode_import": 35.0 / 30.0,
            "lt_max_import": 40.0 / 30.0,
            "cost_local": 0.0,
            "cost_import": 0.0,
        },
    ]


DEFAULT_MATERIALS_DF = pd.DataFrame(_default_materials())


def materials_df_to_policies(df: pd.DataFrame) -> tuple[list[MaterialPolicy], set[str]]:
    df = df.reset_index(drop=True)
    materials: list[MaterialPolicy] = []
    internal: set[str] = set()

    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        source = str(row.get("source", "local")).strip().lower()
        is_import = source.startswith("imp")

        lt_min = float(row["lt_min_import"] if is_import else row["lt_min_local"])
        lt_mode = float(row["lt_mode_import"] if is_import else row["lt_mode_local"])
        lt_max = float(row["lt_max_import"] if is_import else row["lt_max_local"])
        cost = float(row["cost_import"] if is_import else row["cost_local"])

        materials.append(
            MaterialPolicy(
                name=name,
                x=float(row["x"]),
                review_period=int(row["review_period"]),
                ss_months=float(row["ss_months"]),
                lt_min=lt_min,
                lt_mode=lt_mode,
                lt_max=lt_max,
                cost_per_ton=cost,
                source="import" if is_import else "local",
                stock_max_months=float(row.get("stock_max_months", 0.0)),
            )
        )

    return materials, internal

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MaterialPolicy:
    name: str
    x: float  # fraction massique
    review_period: int  # R_i (mois)
    ss_months: float  # stock de sécurité (mois de conso)
    lt_min: float  # lead time min (mois)
    lt_mode: float  # lead time mode (mois)
    lt_max: float  # lead time max (mois)
    cost_per_ton: float = 0.0  # MAD/t (ou autre devise) - utilisé pour arbitrer Local vs Import
    source: str = "local"  # "local" | "import" (sert à activer les retards exceptionnels)
    stock_max_months: float = 0.0  # (non utilisé dans la dynamique) conservé pour compat si besoin


@dataclass
class SimulationConfig:
    horizon_months: int = 24
    sigma_prod: float = 0.10  # variabilité mensuelle de production autour du plan
    n_iter: int = 3000  # itérations Monte Carlo
    seed: int = 42

    # Incertitude sur la production annuelle (Uniforme) - Option B
    annual_min: float = 11000.0
    annual_max: float = 15000.0

    # Retards exceptionnels (pour matières externes)
    p_delay_1: float = 0.15
    p_delay_2: float = 0.05
    delay_1: int = 1
    delay_2: int = 2


@dataclass
class Seasonality:
    hiver: float = 0.23
    printemps: float = 0.24
    ete: float = 0.26
    automne: float = 0.27

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def seasonal_plan_12_months(annual_production: float, seasonality: Seasonality) -> np.ndarray:
    """
    Construit 12 mois de production planifiée à partir de la saisonnalité fournie :
      - Hiver, Printemps, Été, Automne
    Chaque saison = 3 mois.
    """
    winter_m = (seasonality.hiver * annual_production) / 3.0
    spring_m = (seasonality.printemps * annual_production) / 3.0
    summer_m = (seasonality.ete * annual_production) / 3.0
    autumn_m = (seasonality.automne * annual_production) / 3.0

    year_pattern = [winter_m] * 3 + [spring_m] * 3 + [summer_m] * 3 + [autumn_m] * 3
    return np.array(year_pattern, dtype=float)


def build_plan_for_horizon(
    annuals_per_year: np.ndarray, horizon_months: int, seasonality: Seasonality
) -> np.ndarray:
    """
    Construit un plan sur horizon_months en concaténant des plans de 12 mois.
    Les annuals_per_year peuvent contenir plus d'une année (Option B).
    """
    chunks = [seasonal_plan_12_months(float(a), seasonality) for a in annuals_per_year]
    plan = np.concatenate(chunks) if chunks else np.zeros(0, dtype=float)
    return plan[:horizon_months].astype(float, copy=False)


def sample_triangular_int(rng: np.random.Generator, a: float, c: float, b: float) -> int:
    """
    Échantillonne un lead time en mois (triangulaire) puis le convertit en un décalage entier en mois.

    Convention (modèle mensuel) :
    - La revue/commande est faite en début de mois.
    - Les délais peuvent être fractionnaires (ex. 2 jours ~ 0.067 mois).
    - Une livraison avec LT < 1 mois arrive dans le même mois (décalage 0).

    On convertit donc en décalage entier par plancher : offset = floor(LT_mois).
    """
    # Cas dégénéré (délai déterministe) : min = mode = max
    if float(a) == float(b):
        val = float(a)
    else:
        val = float(rng.triangular(a, c, b))
    return max(0, int(math.floor(val)))


def sample_delay(rng: np.random.Generator, cfg: SimulationConfig) -> int:
    """Retard additionnel ΔLT pour matières externes."""
    u = rng.random()
    if u < cfg.p_delay_2:
        return cfg.delay_2
    if u < cfg.p_delay_2 + cfg.p_delay_1:
        return cfg.delay_1
    return 0


def simulate_one_path_with_history(
    rng: np.random.Generator,
    cfg: SimulationConfig,
    materials: List[MaterialPolicy],
    plan_prod: np.ndarray,
    initial_stocks: Dict[str, float],
    target_levels: Dict[str, float],
    internal_materials: set,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Simule une trajectoire sur T mois avec politique (R,S), et retourne:
      - stock_min[name] : stock minimum sur l'horizon (utile pour le risque rupture)
      - stock_hist[name][t] : stock fin de mois t
    """
    T = cfg.horizon_months
    receipts = {m.name: np.zeros(T, dtype=float) for m in materials}
    stock = {m.name: float(initial_stocks[m.name]) for m in materials}
    stock_hist = {m.name: np.zeros(T, dtype=float) for m in materials}

    stock_min = {m.name: stock[m.name] for m in materials}

    for t in range(T):
        # 1) Réceptions (début de mois)
        for m in materials:
            stock[m.name] += receipts[m.name][t]

        # 2) Commandes (R,S) si mois de revue (début de mois, après réceptions)
        for m in materials:
            if (t % m.review_period) == 0:
                # Pipeline = somme des réceptions futures déjà planifiées (hors mois t déjà reçu)
                pipeline = receipts[m.name][(t + 1) :].sum()
                inv_position = stock[m.name] + pipeline
                q = max(0.0, target_levels[m.name] - inv_position)

                if q > 0:
                    lt = sample_triangular_int(rng, m.lt_min, m.lt_mode, m.lt_max)

                    # Retards exceptionnels: appliqués uniquement aux matières externes importées
                    if (m.name not in internal_materials) and (str(m.source).lower().startswith("imp")):
                        lt += sample_delay(rng, cfg)

                    arrival_t = t + lt
                    if arrival_t < T:
                        if arrival_t == t:
                            stock[m.name] += q
                        else:
                            receipts[m.name][arrival_t] += q

        # 3) Production aléatoire autour du plan + consommation (sur le mois)
        eps = rng.normal(0.0, cfg.sigma_prod)
        prod_t = max(0.0, plan_prod[t] * (1.0 + eps))

        for m in materials:
            stock[m.name] -= m.x * prod_t
            stock_hist[m.name][t] = stock[m.name]
            stock_min[m.name] = min(stock_min[m.name], stock[m.name])

    return stock_min, stock_hist


def simulate_one_path_costs(
    rng: np.random.Generator,
    cfg: SimulationConfig,
    materials: List[MaterialPolicy],
    plan_prod: np.ndarray,
    initial_stocks: Dict[str, float],
    target_levels: Dict[str, float],
    internal_materials: set,
    profit_per_ton_abc: float,
) -> Tuple[Dict[str, float], float, float, bool]:
    """
    Simule 1 trajectoire et calcule un bilan économique simple :
      - coût d'achat = somme(q commandée * coût/tonne)
      - coût de rupture = profit perdu si au moins une matière est en rupture (stock < 0) sur un mois

    Hypothèse :
      - Si, sur un mois t, au moins une matière a un stock < 0 après consommation,
        alors la production ABC du mois est considérée perdue (profit = 0 sur ce mois).
    """
    T = cfg.horizon_months
    receipts = {m.name: np.zeros(T, dtype=float) for m in materials}
    stock = {m.name: float(initial_stocks[m.name]) for m in materials}
    stock_min = {m.name: stock[m.name] for m in materials}

    purchase_cost = 0.0
    rupture_cost = 0.0
    any_stockout = False

    for t in range(T):
        # 1) Réceptions (début de mois)
        for m in materials:
            stock[m.name] += receipts[m.name][t]

        # 2) Commandes (R,S) si mois de revue (début de mois, après réceptions)
        for m in materials:
            if (t % m.review_period) == 0:
                pipeline = receipts[m.name][(t + 1) :].sum()
                inv_position = stock[m.name] + pipeline
                q = max(0.0, target_levels[m.name] - inv_position)

                if q > 0:
                    purchase_cost += float(q) * float(m.cost_per_ton)

                    lt = sample_triangular_int(rng, m.lt_min, m.lt_mode, m.lt_max)
                    # Retards exceptionnels : appliqués uniquement aux matières externes importées
                    if (m.name not in internal_materials) and (str(m.source).lower().startswith("imp")):
                        lt += sample_delay(rng, cfg)

                    arrival_t = t + lt
                    if arrival_t < T:
                        if arrival_t == t:
                            stock[m.name] += q
                        else:
                            receipts[m.name][arrival_t] += q

        # 3) Production aléatoire autour du plan + consommation (sur le mois)
        eps = rng.normal(0.0, cfg.sigma_prod)
        prod_t = max(0.0, float(plan_prod[t]) * (1.0 + float(eps)))

        month_stockout = False
        for m in materials:
            stock[m.name] -= m.x * prod_t
            stock_min[m.name] = min(stock_min[m.name], stock[m.name])
            if stock[m.name] < 0:
                month_stockout = True

        if month_stockout:
            any_stockout = True
            rupture_cost += float(profit_per_ton_abc) * float(prod_t)

    return stock_min, purchase_cost, rupture_cost, any_stockout


@dataclass
class CostMonteCarloResults:
    total_cost: np.ndarray
    purchase_cost: np.ndarray
    rupture_cost: np.ndarray
    p_any_stockout: float
    p_any_stockout_by_material: Dict[str, float]


def run_monte_carlo_costs(
    cfg: SimulationConfig,
    materials: List[MaterialPolicy],
    seasonality: Seasonality,
    internal_materials: set,
    profit_per_ton_abc: float,
) -> CostMonteCarloResults:
    """
    Lance N scénarios Monte Carlo et renvoie les coûts (achat + rupture) pour arbitrer Local vs Import.
    """
    rng = np.random.default_rng(cfg.seed)

    years = int(math.ceil(cfg.horizon_months / 12))
    total_cost = np.zeros(cfg.n_iter, dtype=float)
    purchase_cost = np.zeros(cfg.n_iter, dtype=float)
    rupture_cost = np.zeros(cfg.n_iter, dtype=float)
    any_stockout = np.zeros(cfg.n_iter, dtype=bool)

    stock_mins = {m.name: np.zeros(cfg.n_iter, dtype=float) for m in materials}

    for k in range(cfg.n_iter):
        annuals = rng.uniform(cfg.annual_min, cfg.annual_max, size=years).astype(float)
        plan_prod = build_plan_for_horizon(annuals, cfg.horizon_months, seasonality)

        avg_prod = float(plan_prod.mean())
        avg_cons = {m.name: float(m.x * avg_prod) for m in materials}

        target_levels: Dict[str, float] = {}
        initial_stocks: Dict[str, float] = {}
        for m in materials:
            target_levels[m.name] = float(
                (m.lt_mode + m.review_period + m.ss_months) * avg_cons[m.name]
            )
            initial_stocks[m.name] = float((m.lt_mode + m.ss_months) * avg_cons[m.name])

        stock_min_k, pc, rc, any_so = simulate_one_path_costs(
            rng=rng,
            cfg=cfg,
            materials=materials,
            plan_prod=plan_prod,
            initial_stocks=initial_stocks,
            target_levels=target_levels,
            internal_materials=internal_materials,
            profit_per_ton_abc=profit_per_ton_abc,
        )

        purchase_cost[k] = float(pc)
        rupture_cost[k] = float(rc)
        total_cost[k] = float(pc) + float(rc)
        any_stockout[k] = bool(any_so)
        for m in materials:
            stock_mins[m.name][k] = float(stock_min_k[m.name])

    p_any_stockout = float(np.mean(any_stockout))
    p_by_material = {m.name: float(np.mean(stock_mins[m.name] < 0)) for m in materials}

    return CostMonteCarloResults(
        total_cost=total_cost,
        purchase_cost=purchase_cost,
        rupture_cost=rupture_cost,
        p_any_stockout=p_any_stockout,
        p_any_stockout_by_material=p_by_material,
    )


@dataclass
class MonteCarloResults:
    months: np.ndarray
    annual_draws: np.ndarray
    avg_prod_per_run: np.ndarray
    stock_paths: Dict[str, np.ndarray]
    stock_mins: Dict[str, np.ndarray]
    example_hist: Dict[str, np.ndarray]
    example_plan: np.ndarray
    example_annuals: np.ndarray


def run_monte_carlo(
    cfg: SimulationConfig,
    materials: List[MaterialPolicy],
    seasonality: Seasonality,
    internal_materials: set,
) -> MonteCarloResults:
    rng = np.random.default_rng(cfg.seed)

    stock_paths = {
        m.name: np.zeros((cfg.n_iter, cfg.horizon_months), dtype=float) for m in materials
    }
    stock_mins = {m.name: np.zeros(cfg.n_iter, dtype=float) for m in materials}
    example_hist: Dict[str, np.ndarray] | None = None
    example_plan: np.ndarray | None = None

    years = int(math.ceil(cfg.horizon_months / 12))
    annual_draws = np.zeros((cfg.n_iter, years), dtype=float)
    avg_prod_per_run = np.zeros(cfg.n_iter, dtype=float)
    example_annuals: np.ndarray | None = None

    for k in range(cfg.n_iter):
        # 1) Tirage du volume annuel par année (Option B)
        annuals = rng.uniform(cfg.annual_min, cfg.annual_max, size=years).astype(float)
        annual_draws[k, :] = annuals

        # 2) Construction du plan de production sur l'horizon
        plan_prod = build_plan_for_horizon(annuals, cfg.horizon_months, seasonality)

        # 3) Dimensionnement (placeholders) des niveaux cibles S* et stocks initiaux S0
        avg_prod = float(plan_prod.mean())
        avg_prod_per_run[k] = avg_prod
        avg_cons = {m.name: float(m.x * avg_prod) for m in materials}

        target_levels: Dict[str, float] = {}
        initial_stocks: Dict[str, float] = {}
        for m in materials:
            target_levels[m.name] = float(
                (m.lt_mode + m.review_period + m.ss_months) * avg_cons[m.name]
            )
            initial_stocks[m.name] = float((m.lt_mode + m.ss_months) * avg_cons[m.name])

        # 4) Simulation d'une trajectoire
        stock_min_k, hist_k = simulate_one_path_with_history(
            rng, cfg, materials, plan_prod, initial_stocks, target_levels, internal_materials
        )
        for m in materials:
            stock_paths[m.name][k, :] = hist_k[m.name]
            stock_mins[m.name][k] = stock_min_k[m.name]
        if k == 0:
            example_hist = hist_k
            example_plan = plan_prod
            example_annuals = annuals.copy()

    months = np.arange(1, cfg.horizon_months + 1)

    if example_hist is None:
        example_hist = {m.name: np.zeros(cfg.horizon_months, dtype=float) for m in materials}
    if example_plan is None:
        example_plan = np.zeros(cfg.horizon_months, dtype=float)
    if example_annuals is None:
        example_annuals = np.zeros(years, dtype=float)

    return MonteCarloResults(
        months=months,
        annual_draws=annual_draws,
        avg_prod_per_run=avg_prod_per_run,
        stock_paths=stock_paths,
        stock_mins=stock_mins,
        example_hist=example_hist,
        example_plan=example_plan,
        example_annuals=example_annuals,
    )


def summarize_results(results: MonteCarloResults, materials: List[MaterialPolicy]):
    import pandas as pd

    rows: List[Dict[str, float | str]] = []
    for m in materials:
        name = m.name
        mins = results.stock_mins[name]
        paths = results.stock_paths[name]

        p_any_stockout = float(np.mean(mins < 0))
        p5_min, p50_min, p95_min = np.percentile(mins, [5, 50, 95])

        max_per_run = paths.max(axis=1)
        p95_max = float(np.percentile(max_per_run, 95))

        rows.append(
            {
                "name": name,
                "p_any_stockout": p_any_stockout,
                "p5_min": float(p5_min),
                "p50_min": float(p50_min),
                "p95_min": float(p95_min),
                "p95_max": p95_max,
            }
        )

    return pd.DataFrame(rows)

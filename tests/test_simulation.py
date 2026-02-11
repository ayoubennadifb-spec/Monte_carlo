import numpy as np

from montecarlo_imi.defaults import DEFAULT_MATERIALS_DF, DEFAULT_SEASONALITY, materials_df_to_policies
from montecarlo_imi.simulation import SimulationConfig, run_monte_carlo, summarize_results


def test_shapes():
    materials, internal = materials_df_to_policies(DEFAULT_MATERIALS_DF)
    cfg = SimulationConfig(n_iter=10, horizon_months=24, seed=42)
    results = run_monte_carlo(cfg, materials, DEFAULT_SEASONALITY, internal)
    for m in materials:
        assert results.stock_paths[m.name].shape == (cfg.n_iter, cfg.horizon_months)
        assert results.stock_mins[m.name].shape == (cfg.n_iter,)


def test_determinism_same_seed():
    materials, internal = materials_df_to_policies(DEFAULT_MATERIALS_DF)
    cfg = SimulationConfig(n_iter=25, horizon_months=24, seed=42)
    r1 = run_monte_carlo(cfg, materials, DEFAULT_SEASONALITY, internal)
    r2 = run_monte_carlo(cfg, materials, DEFAULT_SEASONALITY, internal)
    for m in materials:
        assert np.array_equal(r1.stock_mins[m.name], r2.stock_mins[m.name])
        assert np.array_equal(r1.stock_paths[m.name], r2.stock_paths[m.name])


def test_notebook_defaults_regression_light():
    materials, internal = materials_df_to_policies(DEFAULT_MATERIALS_DF)
    cfg = SimulationConfig(
        horizon_months=24,
        sigma_prod=0.10,
        n_iter=200,
        seed=42,
        annual_min=11000.0,
        annual_max=15000.0,
        p_delay_1=0.15,
        p_delay_2=0.05,
        delay_1=1,
        delay_2=2,
    )
    results = run_monte_carlo(cfg, materials, DEFAULT_SEASONALITY, internal)
    kpi = summarize_results(results, materials).set_index("name")

    expected = {
        "MAP": {
            "p_any_stockout": 0.005,
            "p5_min": 31.516173378346696,
            "p50_min": 92.53008122525677,
            "p95_min": 143.72598652498053,
            "p95_max": 370.78796232224676,
        },
        "SA": {
            "p_any_stockout": 0.86,
            "p5_min": -745.3509950435677,
            "p50_min": -142.0524825674026,
            "p95_min": 88.04879783689934,
            "p95_max": 895.517209906405,
        },
        "Mica": {
            "p_any_stockout": 0.185,
            "p5_min": -7.2553652923203815,
            "p50_min": 17.680444530153416,
            "p95_min": 36.78724988460753,
            "p95_max": 101.48822168193861,
        },
        "Silice": {
            "p_any_stockout": 0.01,
            "p5_min": 3.2052987828805426,
            "p50_min": 7.737952318903675,
            "p95_min": 13.720983434135062,
            "p95_max": 36.6524359112054,
        },
        "Huile": {
            "p_any_stockout": 0.14,
            "p5_min": -6.083973666054741,
            "p50_min": 10.242458515326772,
            "p95_min": 19.748288094319562,
            "p95_max": 53.11727001208413,
        },
    }

    for name, vals in expected.items():
        for key, exp in vals.items():
            got = float(kpi.loc[name, key])
            assert np.isclose(got, exp, rtol=1e-9, atol=1e-6), f"{name}.{key}: {got} != {exp}"

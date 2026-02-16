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
        "SA": {
            "p_any_stockout": 0.0,
            "p5_min": 402.4678881436355,
            "p50_min": 499.6773438952011,
            "p95_min": 592.0309484301054,
            "p95_max": 896.6734187259415,
        },
        "Mica": {
            "p_any_stockout": 0.0,
            "p5_min": 25.06057832448582,
            "p50_min": 35.11631218596105,
            "p95_min": 42.72256403636265,
            "p95_max": 101.08325784792533,
        },
        "Silice": {
            "p_any_stockout": 0.02,
            "p5_min": 1.8899744928688503,
            "p50_min": 8.920823499402542,
            "p95_min": 14.16840090651368,
            "p95_max": 36.38957978944955,
        },
        "Huile": {
            "p_any_stockout": 0.125,
            "p5_min": -2.556909901034629,
            "p50_min": 9.991855366063668,
            "p95_min": 20.571386867798977,
            "p95_max": 52.9854762931068,
        },
    }

    for name, vals in expected.items():
        for key, exp in vals.items():
            got = float(kpi.loc[name, key])
            assert np.isclose(got, exp, rtol=1e-9, atol=1e-6), f"{name}.{key}: {got} != {exp}"

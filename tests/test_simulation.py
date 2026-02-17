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
            "p5_min": 371.74156910676336,
            "p50_min": 472.6081830794792,
            "p95_min": 553.8827548404718,
            "p95_max": 870.5866229224276,
        },
        "Mica": {
            "p_any_stockout": 0.0,
            "p5_min": 25.84531170766448,
            "p50_min": 34.06990098217503,
            "p95_min": 40.2042037622783,
            "p95_max": 101.71810217095058,
        },
        "Silice": {
            "p_any_stockout": 0.0,
            "p5_min": 4.396833464063283,
            "p50_min": 10.813586198639463,
            "p95_min": 16.190362270306583,
            "p95_max": 32.91450931539802,
        },
        "Huile": {
            "p_any_stockout": 0.07,
            "p5_min": -1.3726527994669394,
            "p50_min": 11.2079345157159,
            "p95_min": 19.229128602621452,
            "p95_max": 53.25247474868084,
        },
    }

    for name, vals in expected.items():
        for key, exp in vals.items():
            got = float(kpi.loc[name, key])
            assert np.isclose(got, exp, rtol=1e-9, atol=1e-6), f"{name}.{key}: {got} != {exp}"

import numpy as np

from montecarlo_imi.ramp_simulation import RampConfig, run_ramp_monte_carlo


def test_ramp_shapes():
    cfg = RampConfig(months=60, n_sims=10, seed=42)
    r = run_ramp_monte_carlo(cfg)
    for key in ["Capacity_tpy", "Dm_tpy", "Daddr_tpy", "DemandClients_tpy", "Sales_tpy"]:
        assert r.mc[key].shape == (cfg.n_sims, cfg.months + 1)
    assert len(r.summary_df) == cfg.months + 1
    assert set(r.demo.keys()) == {"Capacity_tpy", "Dm_tpy", "Daddr_tpy", "DemandClients_tpy", "Sales_tpy"}


def test_ramp_determinism_same_seed():
    cfg = RampConfig(months=60, n_sims=50, seed=42)
    r1 = run_ramp_monte_carlo(cfg)
    r2 = run_ramp_monte_carlo(cfg)
    for key in ["Capacity_tpy", "Dm_tpy", "Daddr_tpy", "DemandClients_tpy", "Sales_tpy"]:
        assert np.array_equal(r1.mc[key], r2.mc[key])


def test_ramp_regression_light_defaults():
    cfg = RampConfig(months=60, n_sims=200, seed=42, scenario="central")
    r = run_ramp_monte_carlo(cfg)
    df = r.summary_df.set_index("Month")

    expected = {
        12: {
            "Capacity_P50_tpy": 1271.8187362759854,
            "Market_P50_tpy": 25194.062953287677,
            "Addressable_P50_tpy": 803.4558458709585,
            "DemandClients_P50_tpy": 316.7794454828977,
            "Sales_P50_tpy": 316.7794454828977,
            "Sales_P10_tpy": 0.0,
            "Sales_P90_tpy": 779.8302935725036,
        },
        24: {
            "Capacity_P50_tpy": 7755.302642018058,
            "Market_P50_tpy": 26771.77859107603,
            "Addressable_P50_tpy": 3076.8057971028547,
            "DemandClients_P50_tpy": 1464.6497333513298,
            "Sales_P50_tpy": 1464.6497333513298,
            "Sales_P10_tpy": 811.7115500795678,
            "Sales_P90_tpy": 2618.8609989715123,
        },
        60: {
            "Capacity_P50_tpy": 12747.016632984589,
            "Market_P50_tpy": 32415.60046305401,
            "Addressable_P50_tpy": 18415.60467685025,
            "DemandClients_P50_tpy": 12087.348015044692,
            "Sales_P50_tpy": 11933.160366005017,
            "Sales_P10_tpy": 9229.55516419463,
            "Sales_P90_tpy": 12577.051172316833,
        },
    }

    for month, cols in expected.items():
        for col, exp in cols.items():
            got = float(df.loc[month, col])
            assert np.isclose(got, exp, rtol=1e-9, atol=1e-6), f"{month}.{col}: {got} != {exp}"


from __future__ import annotations

import io
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from montecarlo_imi.defaults import DEFAULT_CONFIG, DEFAULT_MATERIALS_DF, DEFAULT_SEASONALITY
from montecarlo_imi.ramp_simulation import (
    RampConfig,
    TriangularPrior,
    build_excel_bytes,
    ocp_new_line_presets,
    imports_series,
    logistic,
    market_demand_path,
    market_summary,
    run_ramp_monte_carlo,
)
from montecarlo_imi.simulation import (
    MaterialPolicy,
    Seasonality,
    SimulationConfig,
    build_plan_for_horizon,
    run_monte_carlo,
    summarize_results,
)

MAX_ITER = 200_000
SEASONALITY_TOL = 1e-6


def _reset_supply_defaults() -> None:
    st.session_state["supply_cfg"] = asdict(DEFAULT_CONFIG)
    st.session_state["supply_seasonality"] = DEFAULT_SEASONALITY.as_dict()
    st.session_state["supply_materials_df"] = DEFAULT_MATERIALS_DF.copy()
    st.session_state["supply_results"] = None
    st.session_state["supply_kpi_df"] = None


def _reset_ramp_defaults() -> None:
    cfg = RampConfig()
    st.session_state["ramp_cfg"] = asdict(cfg)
    st.session_state["ramp_imports_df"] = (
        pd.DataFrame(
            {"Year": list(cfg.imports_tpy_by_year.keys()), "Imports_tons": list(cfg.imports_tpy_by_year.values())}
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )
    st.session_state["ramp_results"] = None


def _ensure_state() -> None:
    if "supply_cfg" not in st.session_state:
        _reset_supply_defaults()
    if "ramp_cfg" not in st.session_state or "ramp_imports_df" not in st.session_state:
        _reset_ramp_defaults()


def _validate_materials_df(df: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    required_cols = [
        "name",
        "is_internal",
        "x",
        "review_period",
        "ss_months",
        "lt_min",
        "lt_mode",
        "lt_max",
    ]

    errors: list[str] = []
    warnings: list[str] = []

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Colonne manquante: {col}")
    if errors:
        return errors, warnings, df

    df_work = df.reset_index(drop=True).copy()
    df_work["name"] = df_work["name"].astype(str).str.strip()

    empty_names = df_work["name"].eq("") | df_work["name"].str.lower().eq("nan")
    if empty_names.any():
        bad_rows = (np.where(empty_names)[0] + 1).tolist()
        errors.append(f"Nom matière vide (lignes): {bad_rows}")

    dupes = df_work["name"][df_work["name"].duplicated()].unique().tolist()
    if dupes:
        errors.append(f"Noms de matières dupliqués: {dupes}")

    def _num_col(col: str) -> pd.Series:
        return pd.to_numeric(df_work[col], errors="coerce")

    numeric_cols = ["x", "review_period", "ss_months", "lt_min", "lt_mode", "lt_max"]
    for col in numeric_cols:
        s = _num_col(col)
        if s.isna().any():
            bad_rows = (np.where(s.isna())[0] + 1).tolist()
            errors.append(f"Valeurs non numériques dans '{col}' (lignes): {bad_rows}")
        df_work[col] = s

    rp = df_work["review_period"]
    non_int = ~np.isclose(rp, np.round(rp), atol=0, rtol=0)
    if non_int.any():
        bad_rows = (np.where(non_int)[0] + 1).tolist()
        errors.append(f"'review_period' doit être entier (lignes): {bad_rows}")
    too_small = rp < 1
    if too_small.any():
        bad_rows = (np.where(too_small)[0] + 1).tolist()
        errors.append(f"'review_period' doit être >= 1 (lignes): {bad_rows}")

    x = df_work["x"]
    out = (x < 0) | (x > 1)
    if out.any():
        bad_rows = (np.where(out)[0] + 1).tolist()
        errors.append(f"'x' doit être entre 0 et 1 (lignes): {bad_rows}")
    sum_x = float(x.sum()) if len(x) else 0.0
    if len(x) and not np.isclose(sum_x, 1.0):
        warnings.append(f"Somme des x = {sum_x:.6g} (≠ 1.0).")

    ss = df_work["ss_months"]
    neg_ss = ss < 0
    if neg_ss.any():
        bad_rows = (np.where(neg_ss)[0] + 1).tolist()
        errors.append(f"'ss_months' doit être >= 0 (lignes): {bad_rows}")

    for col in ["lt_min", "lt_mode", "lt_max"]:
        neg = df_work[col] < 0
        if neg.any():
            bad_rows = (np.where(neg)[0] + 1).tolist()
            errors.append(f"'{col}' doit être >= 0 (lignes): {bad_rows}")

    bad_lt = (df_work["lt_min"] > df_work["lt_mode"]) | (df_work["lt_mode"] > df_work["lt_max"])
    if bad_lt.any():
        bad_rows = (np.where(bad_lt)[0] + 1).tolist()
        errors.append(f"Contraintes LT violées (lt_min ≤ lt_mode ≤ lt_max) (lignes): {bad_rows}")

    df_work["review_period"] = df_work["review_period"].round().astype(int)
    return errors, warnings, df_work


def _materials_df_to_policies(df: pd.DataFrame) -> tuple[list[MaterialPolicy], set[str]]:
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
            )
        )

    return materials, internal


def _seasonality_inputs(key_prefix: str, seasonality_state: dict) -> Seasonality:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hiver = st.number_input(
            "Hiver",
            min_value=0.0,
            max_value=1.0,
            value=float(seasonality_state["hiver"]),
            step=0.01,
            key=f"{key_prefix}_hiver",
        )
    with c2:
        printemps = st.number_input(
            "Printemps",
            min_value=0.0,
            max_value=1.0,
            value=float(seasonality_state["printemps"]),
            step=0.01,
            key=f"{key_prefix}_printemps",
        )
    with c3:
        ete = st.number_input(
            "Été",
            min_value=0.0,
            max_value=1.0,
            value=float(seasonality_state["ete"]),
            step=0.01,
            key=f"{key_prefix}_ete",
        )
    with c4:
        automne = st.number_input(
            "Automne",
            min_value=0.0,
            max_value=1.0,
            value=float(seasonality_state["automne"]),
            step=0.01,
            key=f"{key_prefix}_automne",
        )

    st.session_state["supply_seasonality"] = {
        "hiver": hiver,
        "printemps": printemps,
        "ete": ete,
        "automne": automne,
    }
    return Seasonality(hiver=hiver, printemps=printemps, ete=ete, automne=automne)


def _render_supply_chain() -> None:
    tabs = st.tabs(["Paramètres", "Résultats", "Export"])

    with tabs[0]:
        st.subheader("Supply Chain — Paramètres")

        if st.button("Reset notebook defaults", key="supply_reset"):
            _reset_supply_defaults()
            st.rerun()

        cfg_state: dict = st.session_state["supply_cfg"]

        st.markdown("### Simulation")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            horizon_months = st.slider(
                "Horizon (mois)",
                min_value=1,
                max_value=240,
                value=int(cfg_state["horizon_months"]),
                step=1,
                key="supply_horizon",
            )
        with c2:
            annual_min = st.number_input(
                "Prod annuelle min (t/an)",
                min_value=0.0,
                value=float(cfg_state["annual_min"]),
                step=100.0,
                key="supply_annual_min",
            )
        with c3:
            annual_max = st.number_input(
                "Prod annuelle max (t/an)",
                min_value=0.0,
                value=float(cfg_state["annual_max"]),
                step=100.0,
                key="supply_annual_max",
            )
        with c4:
            sigma_pct = st.slider(
                "Variabilité σ (%, mensuel)",
                min_value=0.0,
                max_value=50.0,
                value=float(cfg_state["sigma_prod"]) * 100.0,
                step=0.5,
                key="supply_sigma",
            )
        with c5:
            n_iter = st.number_input(
                "Itérations",
                min_value=1,
                value=int(cfg_state["n_iter"]),
                step=100,
                key="supply_iter",
            )
        with c6:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=int(cfg_state["seed"]),
                step=1,
                key="supply_seed",
            )

        st.session_state["supply_cfg"] = {
            **cfg_state,
            "horizon_months": int(horizon_months),
            "annual_min": float(annual_min),
            "annual_max": float(annual_max),
            "sigma_prod": float(sigma_pct) / 100.0,
            "n_iter": int(n_iter),
            "seed": int(seed),
        }

        if horizon_months > 120:
            st.warning("Horizon > 120 mois: attention performance/mémoire.")

        if n_iter > MAX_ITER:
            st.error(f"Itérations > {MAX_ITER}: bloqué pour éviter un crash mémoire.")

        if float(st.session_state["supply_cfg"]["annual_min"]) > float(st.session_state["supply_cfg"]["annual_max"]):
            st.error("annual_min doit être <= annual_max (bloquant).")

        with st.expander("Retards exceptionnels (matières externes)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                p_delay_1 = st.number_input(
                    "P(delay 1 mois)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg_state["p_delay_1"]),
                    step=0.01,
                    key="supply_p_delay_1",
                )
            with c2:
                p_delay_2 = st.number_input(
                    "P(delay 2 mois)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg_state["p_delay_2"]),
                    step=0.01,
                    key="supply_p_delay_2",
                )
            with c3:
                delay_1 = st.number_input(
                    "Delay 1 (mois)",
                    min_value=0,
                    value=int(cfg_state["delay_1"]),
                    step=1,
                    key="supply_delay_1",
                )
            with c4:
                delay_2 = st.number_input(
                    "Delay 2 (mois)",
                    min_value=0,
                    value=int(cfg_state["delay_2"]),
                    step=1,
                    key="supply_delay_2",
                )

            st.session_state["supply_cfg"] = {
                **st.session_state["supply_cfg"],
                "p_delay_1": float(p_delay_1),
                "p_delay_2": float(p_delay_2),
                "delay_1": int(delay_1),
                "delay_2": int(delay_2),
            }

        prob_sum = float(st.session_state["supply_cfg"]["p_delay_1"]) + float(st.session_state["supply_cfg"]["p_delay_2"])
        if prob_sum > 1.0:
            st.error("p_delay_1 + p_delay_2 doit être <= 1.0")

        st.markdown("### Saisonnalité")
        seasonality = _seasonality_inputs("supply_season", st.session_state["supply_seasonality"])
        season_sum = seasonality.hiver + seasonality.printemps + seasonality.ete + seasonality.automne
        st.write(f"Somme = `{season_sum:.6g}`")
        if abs(season_sum - 1.0) > SEASONALITY_TOL:
            st.error("La somme des saisons doit être ~ 1.0 (bloquant).")

        season_df = pd.DataFrame(
            {
                "Saison": ["Hiver", "Printemps", "Été", "Automne"],
                "Part": [seasonality.hiver, seasonality.printemps, seasonality.ete, seasonality.automne],
            }
        )
        fig_season = px.bar(season_df, x="Saison", y="Part", range_y=[0, 1])
        fig_season.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_season, use_container_width=True)

        st.markdown("### Preview plan de production")
        preview_annual = 0.5 * (
            float(st.session_state["supply_cfg"]["annual_min"]) + float(st.session_state["supply_cfg"]["annual_max"])
        )
        years = int(np.ceil(int(st.session_state["supply_cfg"]["horizon_months"]) / 12))
        annuals_preview = np.full(years, preview_annual, dtype=float)
        plan_prod_preview = build_plan_for_horizon(
            annuals_per_year=annuals_preview,
            horizon_months=int(st.session_state["supply_cfg"]["horizon_months"]),
            seasonality=seasonality,
        )
        preview_df = pd.DataFrame({"Mois": np.arange(1, len(plan_prod_preview) + 1), "Production": plan_prod_preview})
        fig_plan = px.line(preview_df, x="Mois", y="Production")
        fig_plan.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_plan, use_container_width=True)
        st.caption(f"Preview basé sur une production annuelle constante = {preview_annual:.0f} t/an.")

        st.markdown("### Matières & politiques")
        edited_df = st.data_editor(
            st.session_state["supply_materials_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="supply_materials_editor",
            column_config={
                "name": st.column_config.TextColumn("Nom", required=True),
                "is_internal": st.column_config.CheckboxColumn("Interne (OCP)", default=False),
                "x": st.column_config.NumberColumn("x (fraction)", min_value=0.0, max_value=1.0, step=0.001),
                "review_period": st.column_config.NumberColumn("R (mois)", min_value=1, step=1),
                "ss_months": st.column_config.NumberColumn("SS (mois)", min_value=0.0, step=0.1),
                "lt_min": st.column_config.NumberColumn("LT min (mois)", min_value=0.0, step=0.001, format="%.3f"),
                "lt_mode": st.column_config.NumberColumn("LT mode (mois)", min_value=0.0, step=0.001, format="%.3f"),
                "lt_max": st.column_config.NumberColumn("LT max (mois)", min_value=0.0, step=0.001, format="%.3f"),
            },
        )
        st.session_state["supply_materials_df"] = edited_df

        mat_errors, mat_warnings, validated_df = _validate_materials_df(edited_df)
        for w in mat_warnings:
            st.warning(w)
        for e in mat_errors:
            st.error(e)

        can_run = True
        if mat_errors:
            can_run = False
        if n_iter > MAX_ITER:
            can_run = False
        if prob_sum > 1.0:
            can_run = False
        if float(st.session_state["supply_cfg"]["annual_min"]) > float(st.session_state["supply_cfg"]["annual_max"]):
            can_run = False
        if abs(season_sum - 1.0) > SEASONALITY_TOL:
            can_run = False

        n_mat = len(validated_df)
        if n_mat == 0:
            st.error("Aucune matière définie.")
            can_run = False

        est_bytes = int(n_iter) * int(horizon_months) * max(1, n_mat) * 8
        est_mb = est_bytes / (1024 * 1024)
        st.caption(f"Estimation mémoire stock_paths ≈ {est_mb:.1f} MB (hors overhead).")

        if st.button("Run simulation", type="primary", disabled=not can_run, key="supply_run"):
            cfg = SimulationConfig(**st.session_state["supply_cfg"])
            materials, internal = _materials_df_to_policies(validated_df)

            start = time.time()
            with st.spinner("Simulation en cours..."):
                results = run_monte_carlo(
                    cfg=cfg,
                    materials=materials,
                    seasonality=seasonality,
                    internal_materials=internal,
                )
                kpi_df = summarize_results(results, materials)
            elapsed = time.time() - start

            st.session_state["supply_results"] = results
            st.session_state["supply_kpi_df"] = kpi_df
            st.session_state["supply_last_run_meta"] = {
                "elapsed_s": elapsed,
                "cfg": asdict(cfg),
                "seasonality": seasonality.as_dict(),
                "materials": validated_df.to_dict(orient="records"),
            }
            st.success(f"Monte Carlo terminé en {elapsed:.2f}s.")

    with tabs[1]:
        st.subheader("Supply Chain — Résultats")
        results = st.session_state.get("supply_results")
        kpi_df = st.session_state.get("supply_kpi_df")

        if results is None or kpi_df is None:
            st.info("Lance une simulation dans l’onglet Paramètres.")
        else:
            meta = st.session_state.get("supply_last_run_meta", {})
            elapsed = meta.get("elapsed_s", None)
            cfg = meta.get("cfg", {})
            if elapsed is not None:
                st.caption(
                    f"Run: {elapsed:.2f}s | sigma_prod={cfg.get('sigma_prod', 0):.0%} | "
                    f"itérations={cfg.get('n_iter')} | horizon={cfg.get('horizon_months')} mois"
                )

            st.markdown("### Plan de production (exemple)")
            plan_df = pd.DataFrame(
                {"Mois": results.months, "Production planifiée": results.example_plan[: len(results.months)]}
            )
            fig_plan_run = px.line(plan_df, x="Mois", y="Production planifiée")
            fig_plan_run.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_plan_run, use_container_width=True)
            if results.example_annuals.size:
                years_txt = " | ".join([f"Année {i+1}={a:.0f} t/an" for i, a in enumerate(results.example_annuals)])
                st.caption(f"Exemple (tirage annuel): {years_txt}")

            st.markdown("### Distribution des productions annuelles tirées")
            annual_draws = results.annual_draws
            if annual_draws.size:
                n_years = annual_draws.shape[1]
                for y in range(n_years):
                    fig_ann = px.histogram(
                        pd.DataFrame({"Production annuelle": annual_draws[:, y]}),
                        x="Production annuelle",
                        nbins=40,
                    )
                    fig_ann.update_layout(
                        title=f"Tirages production annuelle — Année {y+1}",
                        height=260,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_ann, use_container_width=True)

            st.markdown("### KPI")
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)

            st.markdown("### Courbes par matière")
            mat_names = kpi_df["name"].tolist()
            for name in mat_names:
                with st.expander(f"{name}", expanded=True):
                    months = results.months
                    hist = results.example_hist[name]

                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=months, y=hist, mode="lines", name="Stock"))
                    fig1.add_hline(y=0, line_dash="dash")
                    fig1.update_layout(
                        title=f"Trajectoire exemple — {name}",
                        xaxis_title="Mois",
                        yaxis_title="Stock (t)",
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                    paths = results.stock_paths[name]
                    p5 = np.percentile(paths, 5, axis=0)
                    p50 = np.percentile(paths, 50, axis=0)
                    p95 = np.percentile(paths, 95, axis=0)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=months, y=p50, mode="lines", name="P50"))
                    fig2.add_trace(
                        go.Scatter(
                            x=months,
                            y=p95,
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig2.add_trace(
                        go.Scatter(
                            x=months,
                            y=p5,
                            mode="lines",
                            fill="tonexty",
                            fillcolor="rgba(0, 0, 255, 0.15)",
                            line=dict(width=0),
                            name="P5–P95",
                        )
                    )
                    fig2.add_hline(y=0, line_dash="dash")
                    fig2.update_layout(
                        title=f"Monte Carlo — {name} (P5/P50/P95) — Option B",
                        xaxis_title="Mois",
                        yaxis_title="Stock (t)",
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    mins = results.stock_mins[name]
                    fig3 = px.histogram(pd.DataFrame({"Stock minimum": mins}), x="Stock minimum", nbins=40)
                    fig3.add_vline(x=0, line_dash="dash")
                    fig3.update_layout(
                        title=f"Distribution du stock minimum — {name}",
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig3, use_container_width=True)

    with tabs[2]:
        st.subheader("Supply Chain — Export")
        results = st.session_state.get("supply_results")
        kpi_df = st.session_state.get("supply_kpi_df")
        meta = st.session_state.get("supply_last_run_meta")

        if results is None or kpi_df is None or meta is None:
            st.info("Lance une simulation d’abord.")
        else:
            csv_bytes = kpi_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger KPI (CSV)", data=csv_bytes, file_name="kpi_supply.csv", mime="text/csv"
            )

            config_json = json.dumps(
                {
                    "cfg": meta["cfg"],
                    "seasonality": meta["seasonality"],
                    "materials": meta["materials"],
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            st.download_button(
                "Télécharger config (JSON)",
                data=config_json,
                file_name="config_supply.json",
                mime="application/json",
            )

            with st.expander("Exporter trajectoires (NPZ)", expanded=False):
                st.caption("Peut être volumineux.")
                buf = io.BytesIO()
                arrays: dict[str, np.ndarray] = {
                    "months": results.months,
                    "example_plan": results.example_plan,
                    "example_annuals": results.example_annuals,
                    "annual_draws": results.annual_draws,
                    "avg_prod_per_run": results.avg_prod_per_run,
                }
                for name, arr in results.stock_mins.items():
                    arrays[f"stock_mins__{name}"] = arr
                for name, arr in results.stock_paths.items():
                    arrays[f"stock_paths__{name}"] = arr
                np.savez_compressed(buf, **arrays)
                st.download_button(
                    "Télécharger trajectoires (NPZ)",
                    data=buf.getvalue(),
                    file_name="trajectoires_supply.npz",
                    mime="application/octet-stream",
                )


def _render_ramp_legacy() -> None:
    tabs = st.tabs(["Paramètres", "Résultats", "Export"])

    with tabs[0]:
        st.subheader("ABC Ramp — Paramètres")
        if st.button("Reset ramp defaults", key="ramp_reset"):
            _reset_ramp_defaults()
            st.rerun()

        cfg_state: dict = st.session_state["ramp_cfg"]
        thr_state: dict = st.session_state["ramp_thresholds"]

        st.markdown("### Horizon")
        months = st.slider("Horizon (mois)", 1, 240, int(cfg_state["months"]), 1, key="ramp_months")

        st.markdown("### Production / capacité")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            Q_max = st.number_input(
                "Q_max (t/an)",
                min_value=0.0,
                value=float(cfg_state["Q_max"]),
                step=500.0,
                key="ramp_qmax",
            )
        with c2:
            util_target = st.number_input(
                "Utilisation cible (0..1)",
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_state["util_target"]),
                step=0.01,
                key="ramp_util",
            )
        with c3:
            t0_cap = st.number_input(
                "t0 capacité (mois)",
                min_value=0.0,
                value=float(cfg_state["t0_cap"]),
                step=1.0,
                key="ramp_t0_cap",
            )
        with c4:
            k_cap = st.number_input(
                "k capacité",
                min_value=0.0,
                value=float(cfg_state["k_cap"]),
                step=0.01,
                key="ramp_k_cap",
            )

        st.markdown("### Marché / demande")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            lam0 = st.number_input(
                "lam0 (clients/mois)",
                min_value=0.0,
                value=float(cfg_state["lam0"]),
                step=0.05,
                key="ramp_lam0",
            )
        with c2:
            lam_max = st.number_input(
                "lam_max (clients/mois)",
                min_value=0.0,
                value=float(cfg_state["lam_max"]),
                step=0.1,
                key="ramp_lammax",
            )
        with c3:
            t0_lam = st.number_input(
                "t0 acquisition (mois)",
                min_value=0.0,
                value=float(cfg_state["t0_lam"]),
                step=1.0,
                key="ramp_t0_lam",
            )
        with c4:
            k_lam = st.number_input(
                "k acquisition",
                min_value=0.0,
                value=float(cfg_state["k_lam"]),
                step=0.01,
                key="ramp_k_lam",
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mean_target = st.number_input(
                "Demande moyenne cible (t/mois/client)",
                min_value=0.0,
                value=float(cfg_state["mean_target"]),
                step=1.0,
                key="ramp_mean_target",
            )
        with c2:
            sigma = st.number_input(
                "Sigma lognormal",
                min_value=0.0,
                value=float(cfg_state["sigma"]),
                step=0.05,
                key="ramp_sigma",
            )
        with c3:
            onboard_mean = st.number_input(
                "Onboarding mean (mois)",
                min_value=0.0,
                value=float(cfg_state["onboard_mean"]),
                step=0.5,
                key="ramp_onboard_mean",
            )
        with c4:
            onboard_sd = st.number_input(
                "Onboarding sd (mois)",
                min_value=0.0,
                value=float(cfg_state["onboard_sd"]),
                step=0.5,
                key="ramp_onboard_sd",
            )

        st.markdown("### Dynamique client")
        c1, c2, c3 = st.columns(3)
        with c1:
            cust_growth_mean = st.number_input(
                "Croissance mean (mois)",
                value=float(cfg_state["cust_growth_mean"]),
                step=0.005,
                format="%.4f",
                key="ramp_cg_mean",
            )
        with c2:
            cust_growth_sd = st.number_input(
                "Croissance sd",
                min_value=0.0,
                value=float(cfg_state["cust_growth_sd"]),
                step=0.005,
                format="%.4f",
                key="ramp_cg_sd",
            )
        with c3:
            churn_monthly = st.number_input(
                "Churn mensuel (0..1)",
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_state["churn_monthly"]),
                step=0.005,
                format="%.4f",
                key="ramp_churn",
            )

        st.markdown("### Aléas production")
        c1, c2 = st.columns(2)
        with c1:
            unplanned_downtime_mean = st.number_input(
                "Downtime mean (0..1)",
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_state["unplanned_downtime_mean"]),
                step=0.005,
                format="%.4f",
                key="ramp_dt_mean",
            )
        with c2:
            unplanned_downtime_sd = st.number_input(
                "Downtime sd (0..1)",
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_state["unplanned_downtime_sd"]),
                step=0.005,
                format="%.4f",
                key="ramp_dt_sd",
            )

        st.markdown("### Monte Carlo")
        c1, c2 = st.columns(2)
        with c1:
            n_sims = st.number_input(
                "Nb trajectoires",
                min_value=1,
                value=int(cfg_state["n_sims"]),
                step=100,
                key="ramp_n_sims",
            )
        with c2:
            seed = st.number_input("Seed", min_value=0, value=int(cfg_state["seed"]), step=1, key="ramp_seed")

        st.markdown("### Seuils (Ventes P50)")
        c1, c2 = st.columns(2)
        with c1:
            threshold_1 = st.number_input(
                "Seuil 1 (t/an)",
                min_value=0.0,
                value=float(thr_state["threshold_1"]),
                step=500.0,
                key="ramp_thr1",
            )
        with c2:
            threshold_2 = st.number_input(
                "Seuil 2 (t/an)",
                min_value=0.0,
                value=float(thr_state["threshold_2"]),
                step=500.0,
                key="ramp_thr2",
            )

        st.session_state["ramp_thresholds"] = {"threshold_1": float(threshold_1), "threshold_2": float(threshold_2)}
        st.session_state["ramp_cfg"] = {
            "months": int(months),
            "Q_max": float(Q_max),
            "util_target": float(util_target),
            "t0_cap": float(t0_cap),
            "k_cap": float(k_cap),
            "lam0": float(lam0),
            "lam_max": float(lam_max),
            "t0_lam": float(t0_lam),
            "k_lam": float(k_lam),
            "mean_target": float(mean_target),
            "sigma": float(sigma),
            "onboard_mean": float(onboard_mean),
            "onboard_sd": float(onboard_sd),
            "cust_growth_mean": float(cust_growth_mean),
            "cust_growth_sd": float(cust_growth_sd),
            "churn_monthly": float(churn_monthly),
            "unplanned_downtime_mean": float(unplanned_downtime_mean),
            "unplanned_downtime_sd": float(unplanned_downtime_sd),
            "n_sims": int(n_sims),
            "seed": int(seed),
        }

        est_mb = int(n_sims) * (int(months) + 1) * 3 * 8 / (1024 * 1024)
        st.caption(f"Estimation mémoire demand/prod/sales ≈ {est_mb:.1f} MB.")

        if st.button("Run ramp simulation", type="primary", key="ramp_run"):
            cfg = RampConfig(**st.session_state["ramp_cfg"])
            start = time.time()
            with st.spinner("Simulation en cours..."):
                results = run_ramp_monte_carlo(cfg)
            elapsed = time.time() - start
            st.session_state["ramp_results"] = results
            st.session_state["ramp_last_run_meta"] = {"elapsed_s": elapsed, "cfg": cfg.as_dict()}
            st.success(f"Monte Carlo terminé en {elapsed:.2f}s.")

    with tabs[1]:
        st.subheader("ABC Ramp — Résultats")
        results = st.session_state.get("ramp_results")
        meta = st.session_state.get("ramp_last_run_meta", {})
        if results is None:
            st.info("Lance une simulation dans l’onglet Paramètres.")
        else:
            elapsed = meta.get("elapsed_s", None)
            cfg = meta.get("cfg", {})
            if elapsed is not None:
                st.caption(f"Run: {elapsed:.2f}s | sims={cfg.get('n_sims')} | horizon={cfg.get('months')} mois")

            cap_df = results.cap_df
            fig_cap = px.line(cap_df, x="Month", y="Capacity_tpy", title="Capacité effective (t/an)")
            fig_cap.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_cap, use_container_width=True)

            df = results.summary_df
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Capacity_P50_tpy"], name="Capacité P50", mode="lines"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Demand_P50_tpy"], name="Demande P50", mode="lines"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Sales_P50_tpy"], name="Ventes P50", mode="lines"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Sales_P90_tpy"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Sales_P10_tpy"], mode="lines", fill="tonexty", fillcolor="rgba(0, 128, 0, 0.15)", line=dict(width=0), name="Ventes P10–P90"))
            fig_main.update_layout(
                title="Capacité / Demande / Ventes (annualisé) — P50 + bande ventes P10–P90",
                xaxis_title="Mois",
                yaxis_title="t/an",
                height=360,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_main, use_container_width=True)

            thresholds = st.session_state.get("ramp_thresholds", {"threshold_1": 11000.0, "threshold_2": 15000.0})
            m1 = first_cross(df["Sales_P50_tpy"].to_numpy(), float(thresholds["threshold_1"]))
            m2 = first_cross(df["Sales_P50_tpy"].to_numpy(), float(thresholds["threshold_2"]))
            st.markdown("### Atteinte des seuils (Ventes P50)")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Seuil (t/an)": [float(thresholds["threshold_1"]), float(thresholds["threshold_2"])],
                        "Mois (P50 ventes)": [m1, m2],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Courbes quantiles (annualisé)")
            for label, q in [("Demande", results.q_d), ("Production", results.q_p), ("Ventes", results.q_s)]:
                dff = pd.DataFrame(
                    {
                        "Month": results.t,
                        "P10": annualize(q["P10"]),
                        "P50": annualize(q["P50"]),
                        "P90": annualize(q["P90"]),
                        "Mean": annualize(q["Mean"]),
                    }
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dff["Month"], y=dff["P50"], name="P50", mode="lines"))
                fig.add_trace(go.Scatter(x=dff["Month"], y=dff["Mean"], name="Mean", mode="lines"))
                fig.add_trace(go.Scatter(x=dff["Month"], y=dff["P90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=dff["Month"], y=dff["P10"], mode="lines", fill="tonexty", fillcolor="rgba(0, 0, 255, 0.12)", line=dict(width=0), name="P10–P90"))
                fig.update_layout(
                    title=f"{label} — P50/Mean + bande P10–P90",
                    xaxis_title="Mois",
                    yaxis_title="t/an",
                    height=320,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("ABC Ramp — Export")
        results = st.session_state.get("ramp_results")
        if results is None:
            st.info("Lance une simulation d’abord.")
        else:
            csv_bytes = results.summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger résultats (CSV)",
                data=csv_bytes,
                file_name="abc_ramp_results.csv",
                mime="text/csv",
            )

            config_json = json.dumps(
                {"cfg": results.cfg.as_dict(), "thresholds": st.session_state.get("ramp_thresholds", {})},
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            st.download_button(
                "Télécharger config (JSON)",
                data=config_json,
                file_name="abc_ramp_config.json",
                mime="application/json",
            )

            with st.expander("Exporter Excel (XLSX)", expanded=False):
                st.caption("Inclut paramètres + feuille résultats + chart.")
                xlsx = build_excel_bytes(results)
                st.download_button(
                    "Télécharger Excel (XLSX)",
                    data=xlsx,
                    file_name="abc_ramp_simulation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


def _render_ramp() -> None:
    tabs = st.tabs(["Paramètres", "Résultats", "Export"])

    def _validate_imports_df(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
        errors: list[str] = []
        required = ["Year", "Imports_tons"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Colonne manquante: {col}")
        if errors:
            return errors, df

        work = df[required].copy()
        work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
        work["Imports_tons"] = pd.to_numeric(work["Imports_tons"], errors="coerce")

        if len(work) == 0:
            errors.append("La table imports est vide.")
            return errors, df

        if work.isna().any(axis=None):
            bad_rows = (np.where(work.isna().any(axis=1))[0] + 1).tolist()
            errors.append(f"Valeurs manquantes/non numériques (imports) (lignes): {bad_rows}")
            return errors, df

        work["Year"] = work["Year"].astype(int)
        dup_mask = work["Year"].duplicated()
        if dup_mask.any():
            years = work.loc[dup_mask, "Year"].tolist()
            errors.append(f"Années dupliquées (imports): {years}")

        neg = work["Imports_tons"] < 0
        if neg.any():
            bad_rows = (np.where(neg)[0] + 1).tolist()
            errors.append(f"Imports négatifs (lignes): {bad_rows}")

        work = work.sort_values("Year").reset_index(drop=True)
        return errors, work

    with tabs[0]:
        st.subheader("ABC Ramp — Paramètres")
        if st.button("Reset notebook defaults", key="ramp_reset"):
            _reset_ramp_defaults()
            st.rerun()

        cfg_state: dict = st.session_state["ramp_cfg"]

        st.markdown("### Presets (scénarios)")
        presets = ocp_new_line_presets()
        preset_name = st.selectbox(
            "Choisir un scénario",
            options=["(aucun)"] + list(presets.keys()),
            index=0,
            help="Applique un ensemble d'hypothèses cohérentes (marché, part, capacité, clients).",
            key="ramp_preset_name",
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            apply_preset = st.button("Appliquer le preset", disabled=preset_name == "(aucun)", key="ramp_apply_preset")
        with c2:
            if preset_name != "(aucun)":
                st.caption("Après application, tu peux ajuster les paramètres manuellement.")

        if apply_preset and preset_name != "(aucun)":
            patch = presets[str(preset_name)]
            st.session_state["ramp_cfg"] = {**st.session_state["ramp_cfg"], **patch}
            st.session_state["ramp_results"] = None
            st.rerun()

        st.markdown("### Simulation")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            months = st.slider(
                "Horizon (mois)",
                min_value=1,
                max_value=240,
                value=int(cfg_state["months"]),
                step=1,
                key="ramp_months",
            )
        with c2:
            n_sims = st.number_input(
                "Simulations (n_sims)",
                min_value=1,
                value=int(cfg_state["n_sims"]),
                step=100,
                key="ramp_n_sims",
            )
        with c3:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=int(cfg_state["seed"]),
                step=1,
                key="ramp_seed",
            )
        with c4:
            scenario = st.selectbox(
                "Scénario",
                options=["prudent", "central", "ambitieux"],
                index=["prudent", "central", "ambitieux"].index(str(cfg_state.get("scenario", "central"))),
                key="ramp_scenario",
            )

        if months > 120:
            st.warning("Horizon > 120 mois: attention performance/mémoire.")
        if int(n_sims) > MAX_ITER:
            st.error(f"n_sims > {MAX_ITER}: bloqué pour éviter un crash mémoire.")

        st.markdown("### Données marché (imports)")
        imports_df_edited = st.data_editor(
            st.session_state["ramp_imports_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="ramp_imports_editor",
            column_config={
                "Year": st.column_config.NumberColumn("Année", min_value=1900, step=1),
                "Imports_tons": st.column_config.NumberColumn("Imports (t/an)", min_value=0.0, step=100.0),
            },
        )
        st.session_state["ramp_imports_df"] = imports_df_edited
        import_errors, imports_df_valid = _validate_imports_df(imports_df_edited)
        for e in import_errors:
            st.error(e)

        years = imports_df_valid["Year"].tolist() if not import_errors else []
        if years:
            baseline_year = st.selectbox(
                "Année de référence (D0)",
                options=years,
                index=(
                    years.index(int(cfg_state.get("baseline_year", years[-1])))
                    if int(cfg_state.get("baseline_year", years[-1])) in years
                    else len(years) - 1
                ),
                key="ramp_baseline_year",
            )
        else:
            baseline_year = int(cfg_state.get("baseline_year", 2024))

        with st.expander("Croissance & volatilité", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                growth_prudent = st.number_input(
                    "Croissance prudent (annuelle)",
                    value=float(cfg_state["growth_prudent"]),
                    step=0.005,
                    format="%.3f",
                    help="Ex: 0.03 = 3%",
                    key="ramp_growth_prudent",
                )
            with c2:
                growth_ambitieux = st.number_input(
                    "Croissance ambitieux (annuelle)",
                    value=float(cfg_state["growth_ambitieux"]),
                    step=0.005,
                    format="%.3f",
                    help="Ex: 0.08 = 8%",
                    key="ramp_growth_ambitieux",
                )
            with c3:
                central_growth_mode = st.radio(
                    "Croissance centrale",
                    options=["cagr", "override"],
                    index=0 if str(cfg_state.get("central_growth_mode", "cagr")) == "cagr" else 1,
                    horizontal=True,
                    format_func=lambda x: "CAGR (calculé)" if x == "cagr" else "Override",
                    key="ramp_central_growth_mode",
                )
            with c4:
                growth_central_override = st.number_input(
                    "Override croissance centrale",
                    value=float(cfg_state["growth_central_override"]),
                    step=0.005,
                    format="%.3f",
                    help="Utilisé si 'Override' est sélectionné.",
                    disabled=str(central_growth_mode) != "override",
                    key="ramp_growth_central_override",
                )

            c1, c2, c3 = st.columns(3)
            with c1:
                sigma_y_mode = st.radio(
                    "σ annuel du marché",
                    options=["cv", "override"],
                    index=0 if str(cfg_state.get("sigma_y_mode", "cv")) == "cv" else 1,
                    horizontal=True,
                    format_func=lambda x: "CV (historique)" if x == "cv" else "Override",
                    key="ramp_sigma_y_mode",
                )
            with c2:
                sigma_y_override = st.number_input(
                    "Override σ annuel",
                    min_value=0.0,
                    value=float(cfg_state["sigma_y_override"]),
                    step=0.01,
                    format="%.3f",
                    disabled=str(sigma_y_mode) != "override",
                    key="ramp_sigma_y_override",
                )
            with c3:
                market_noise = st.checkbox(
                    "Activer bruit mensuel du marché",
                    value=bool(cfg_state.get("market_noise", True)),
                    key="ramp_market_noise",
                )

        with st.expander("Part de marché s(t)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                s_max = st.number_input(
                    "s_max (0..1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg_state["s_max"]),
                    step=0.01,
                    key="ramp_s_max",
                )
            with c2:
                t0_s = st.number_input(
                    "t0_s (mois)",
                    min_value=0.0,
                    value=float(cfg_state["t0_s"]),
                    step=1.0,
                    key="ramp_t0_s",
                )
            with c3:
                k_s = st.number_input(
                    "k_s",
                    min_value=0.0,
                    value=float(cfg_state["k_s"]),
                    step=0.01,
                    key="ramp_k_s",
                )
            if float(k_s) > 0:
                dt_10_90_s = 4.394 / float(k_s)
                st.caption(f"Montée 10→90% ≈ {dt_10_90_s:.1f} mois | s(t0_s)=s_max/2")
            else:
                st.caption("k_s=0 ⇒ pas de montée (part de marché constante).")

        with st.expander("Capacité / production", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                Q_max = st.number_input(
                    "Q_max (t/an)",
                    min_value=0.0,
                    value=float(cfg_state["Q_max"]),
                    step=500.0,
                    key="ramp_Q_max",
                )
            with c2:
                u_target = st.number_input(
                    "u_target (0..1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg_state["u_target"]),
                    step=0.01,
                    key="ramp_u_target",
                )
            with c3:
                t0_cap = st.number_input(
                    "t0_cap (mois)",
                    min_value=0.0,
                    value=float(cfg_state["t0_cap"]),
                    step=1.0,
                    key="ramp_t0_cap",
                )
            with c4:
                k_cap = st.number_input(
                    "k_cap",
                    min_value=0.0,
                    value=float(cfg_state["k_cap"]),
                    step=0.01,
                    key="ramp_k_cap",
                )

            c1, c2 = st.columns(2)
            with c1:
                downtime_mean = st.number_input(
                    "Downtime mean (fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg_state["downtime_mean"]),
                    step=0.005,
                    key="ramp_downtime_mean",
                )
            with c2:
                downtime_sd = st.number_input(
                    "Downtime sd (fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg_state["downtime_sd"]),
                    step=0.005,
                    key="ramp_downtime_sd",
                )
            if float(k_cap) > 0:
                q_inf = float(Q_max) * float(u_target)
                dt_10_90 = 4.394 / float(k_cap)
                st.caption(f"Capacité mature ≈ {q_inf:,.0f} t/an | montée 10→90% ≈ {dt_10_90:.1f} mois")
            else:
                st.caption("k_cap=0 ⇒ pas de montée (capacité constante).")

        with st.expander("Clients / demande", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                lam0 = st.number_input(
                    "λ0 (nouveaux clients/mois)",
                    min_value=0.0,
                    value=float(cfg_state["lam0"]),
                    step=0.05,
                    key="ramp_lam0",
                )
            with c2:
                lam_max = st.number_input(
                    "λ_max (nouveaux clients/mois)",
                    min_value=0.0,
                    value=float(cfg_state["lam_max"]),
                    step=0.1,
                    key="ramp_lam_max",
                )
            with c3:
                t0_lam = st.number_input(
                    "t0_lam (mois)",
                    min_value=0.0,
                    value=float(cfg_state["t0_lam"]),
                    step=1.0,
                    key="ramp_t0_lam",
                )
            with c4:
                k_lam = st.number_input(
                    "k_lam",
                    min_value=0.0,
                    value=float(cfg_state["k_lam"]),
                    step=0.01,
                    key="ramp_k_lam",
                )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                onboard_mean = st.number_input(
                    "Onboarding mean (mois)",
                    min_value=0.0,
                    value=float(cfg_state["onboard_mean"]),
                    step=0.5,
                    key="ramp_onboard_mean",
                )
            with c2:
                onboard_sd = st.number_input(
                    "Onboarding sd (mois)",
                    min_value=0.0,
                    value=float(cfg_state["onboard_sd"]),
                    step=0.5,
                    key="ramp_onboard_sd",
                )
            with c3:
                mean_target_tpm = st.number_input(
                    "Demande moyenne cible (t/mois/client)",
                    min_value=0.0,
                    value=float(cfg_state["mean_target_tpm"]),
                    step=0.5,
                    key="ramp_mean_target_tpm",
                )
            with c4:
                sigma_logn = st.number_input(
                    "Sigma lognormal (log-space)",
                    min_value=0.0,
                    value=float(cfg_state["sigma_logn"]),
                    step=0.05,
                    key="ramp_sigma_logn",
                )

            c1, c2, c3 = st.columns(3)
            with c1:
                cust_growth_mean = st.number_input(
                    "Croissance demande/client mean (mensuelle)",
                    value=float(cfg_state["cust_growth_mean"]),
                    step=0.001,
                    format="%.3f",
                    key="ramp_cust_growth_mean",
                )
            with c2:
                cust_growth_sd = st.number_input(
                    "Croissance demande/client sd (mensuelle)",
                    min_value=0.0,
                    value=float(cfg_state["cust_growth_sd"]),
                    step=0.001,
                    format="%.3f",
                    key="ramp_cust_growth_sd",
                )
            with c3:
                churn_monthly = st.number_input(
                    "Churn mensuel (0..1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(cfg_state["churn_monthly"]),
                    step=0.005,
                    key="ramp_churn_monthly",
                )

        st.markdown("### Seuils & seed démo")
        c1, c2, c3 = st.columns(3)
        with c1:
            threshold_1 = st.number_input(
                "Seuil 1 (t/an)",
                min_value=0.0,
                value=float(cfg_state["threshold_1"]),
                step=500.0,
                key="ramp_thr1",
            )
        with c2:
            threshold_2 = st.number_input(
                "Seuil 2 (t/an)",
                min_value=0.0,
                value=float(cfg_state["threshold_2"]),
                step=500.0,
                key="ramp_thr2",
            )
        with c3:
            demo_seed = st.number_input(
                "Seed démo (trajectoire)",
                min_value=0,
                value=int(cfg_state["demo_seed"]),
                step=1,
                key="ramp_demo_seed",
            )

        new_cfg = {
            **cfg_state,
            "months": int(months),
            "n_sims": int(n_sims),
            "seed": int(seed),
            "scenario": str(scenario),
            "baseline_year": int(baseline_year),
            "growth_prudent": float(growth_prudent),
            "growth_ambitieux": float(growth_ambitieux),
            "central_growth_mode": str(central_growth_mode),
            "growth_central_override": float(growth_central_override),
            "sigma_y_mode": str(sigma_y_mode),
            "sigma_y_override": float(sigma_y_override),
            "market_noise": bool(market_noise),
            "s_max": float(s_max),
            "t0_s": float(t0_s),
            "k_s": float(k_s),
            "Q_max": float(Q_max),
            "u_target": float(u_target),
            "t0_cap": float(t0_cap),
            "k_cap": float(k_cap),
            "downtime_mean": float(downtime_mean),
            "downtime_sd": float(downtime_sd),
            "lam0": float(lam0),
            "lam_max": float(lam_max),
            "t0_lam": float(t0_lam),
            "k_lam": float(k_lam),
            "onboard_mean": float(onboard_mean),
            "onboard_sd": float(onboard_sd),
            "mean_target_tpm": float(mean_target_tpm),
            "sigma_logn": float(sigma_logn),
            "cust_growth_mean": float(cust_growth_mean),
            "cust_growth_sd": float(cust_growth_sd),
            "churn_monthly": float(churn_monthly),
            "demo_seed": int(demo_seed),
            "threshold_1": float(threshold_1),
            "threshold_2": float(threshold_2),
        }
        if not import_errors:
            new_cfg["imports_tpy_by_year"] = {
                int(y): float(v)
                for y, v in zip(imports_df_valid["Year"].tolist(), imports_df_valid["Imports_tons"].tolist())
            }
        st.session_state["ramp_cfg"] = new_cfg

        can_run = True
        if int(n_sims) > MAX_ITER:
            can_run = False
        if import_errors:
            can_run = False

        n_keys = 5
        est_bytes = int(n_sims) * (int(months) + 1) * n_keys * 8
        est_mb = est_bytes / (1024 * 1024)
        st.caption(f"Estimation mémoire MC ≈ {est_mb:.1f} MB (5 séries, float64).")
        if est_mb > 2000:
            st.error("Estimation mémoire > 2 GB: bloqué pour éviter un crash.")
            can_run = False

        def _tri_rel(x: float, rel: float) -> TriangularPrior:
            x = float(x)
            lo = x * (1.0 - float(rel))
            hi = x * (1.0 + float(rel))
            if lo > hi:
                lo, hi = hi, lo
            return TriangularPrior(low=lo, mode=x, high=hi)

        def _tri_abs(x: float, d: float, clamp01: bool = False) -> TriangularPrior:
            x = float(x)
            lo = x - float(d)
            hi = x + float(d)
            if clamp01:
                lo = max(0.0, min(1.0, lo))
                hi = max(0.0, min(1.0, hi))
            else:
                lo = max(0.0, lo)
                hi = max(0.0, hi)
            if lo > hi:
                lo, hi = hi, lo
            return TriangularPrior(low=lo, mode=x, high=hi)

        priors: dict[str, TriangularPrior] | None = None
        priors_dump: dict[str, dict[str, float]] | None = None

        with st.expander("Incertitude sur paramètres (priors) — optionnel", expanded=False):
            st.caption(
                "Utile quand tu n'as pas de base de calibration. "
                "Cela ajoute une incertitude épistémique en tirant des paramètres par trajectoire (triangulaire)."
            )
            enable_priors = st.checkbox("Activer l'incertitude sur paramètres", value=False, key="ramp_enable_priors")
            level = st.selectbox(
                "Niveau d'incertitude",
                options=["Faible", "Moyen", "Fort"],
                index=1,
                disabled=not enable_priors,
                key="ramp_priors_level",
            )

            if enable_priors:
                if level == "Faible":
                    rel_q = 0.10
                    rel_k = 0.30
                    rel_lam = 0.30
                    rel_mean = 0.20
                    dt_t0 = 6.0
                    abs_frac = 0.05
                    abs_dt = 0.02
                    abs_churn = 0.005
                    abs_on_m = 2.0
                    abs_on_sd = 1.0
                    abs_sigma_logn = 0.10
                elif level == "Fort":
                    rel_q = 0.30
                    rel_k = 0.70
                    rel_lam = 0.70
                    rel_mean = 0.40
                    dt_t0 = 18.0
                    abs_frac = 0.15
                    abs_dt = 0.06
                    abs_churn = 0.015
                    abs_on_m = 4.0
                    abs_on_sd = 2.0
                    abs_sigma_logn = 0.30
                else:  # Moyen
                    rel_q = 0.20
                    rel_k = 0.50
                    rel_lam = 0.50
                    rel_mean = 0.30
                    dt_t0 = 12.0
                    abs_frac = 0.10
                    abs_dt = 0.04
                    abs_churn = 0.010
                    abs_on_m = 3.0
                    abs_on_sd = 1.5
                    abs_sigma_logn = 0.20

                mode_cfg = st.session_state["ramp_cfg"]
                priors = {
                    # Part de marché
                    "s_max": _tri_abs(float(mode_cfg["s_max"]), abs_frac, clamp01=True),
                    "t0_s": TriangularPrior(
                        low=max(0.0, float(mode_cfg["t0_s"]) - dt_t0),
                        mode=float(mode_cfg["t0_s"]),
                        high=float(mode_cfg["t0_s"]) + dt_t0,
                    ),
                    "k_s": _tri_rel(float(mode_cfg["k_s"]), rel_k),
                    # Capacité / production
                    "Q_max": _tri_rel(float(mode_cfg["Q_max"]), rel_q),
                    "u_target": _tri_abs(float(mode_cfg["u_target"]), abs_frac, clamp01=True),
                    "t0_cap": TriangularPrior(
                        low=max(0.0, float(mode_cfg["t0_cap"]) - dt_t0),
                        mode=float(mode_cfg["t0_cap"]),
                        high=float(mode_cfg["t0_cap"]) + dt_t0,
                    ),
                    "k_cap": _tri_rel(float(mode_cfg["k_cap"]), rel_k),
                    "downtime_mean": _tri_abs(float(mode_cfg["downtime_mean"]), abs_dt, clamp01=True),
                    "downtime_sd": _tri_abs(float(mode_cfg["downtime_sd"]), abs_dt, clamp01=True),
                    # Clients / demande
                    "lam0": _tri_rel(float(mode_cfg["lam0"]), rel_lam),
                    "lam_max": _tri_rel(float(mode_cfg["lam_max"]), rel_lam),
                    "t0_lam": TriangularPrior(
                        low=max(0.0, float(mode_cfg["t0_lam"]) - dt_t0),
                        mode=float(mode_cfg["t0_lam"]),
                        high=float(mode_cfg["t0_lam"]) + dt_t0,
                    ),
                    "k_lam": _tri_rel(float(mode_cfg["k_lam"]), rel_k),
                    "onboard_mean": _tri_abs(float(mode_cfg["onboard_mean"]), abs_on_m, clamp01=False),
                    "onboard_sd": _tri_abs(float(mode_cfg["onboard_sd"]), abs_on_sd, clamp01=False),
                    "mean_target_tpm": _tri_rel(float(mode_cfg["mean_target_tpm"]), rel_mean),
                    "sigma_logn": _tri_abs(float(mode_cfg["sigma_logn"]), abs_sigma_logn, clamp01=False),
                    "cust_growth_mean": _tri_rel(float(mode_cfg["cust_growth_mean"]), 0.50),
                    "cust_growth_sd": _tri_rel(float(mode_cfg["cust_growth_sd"]), 0.50),
                    "churn_monthly": _tri_abs(float(mode_cfg["churn_monthly"]), abs_churn, clamp01=True),
                }

                priors_dump = {
                    k: {"low": float(v.low), "mode": float(v.mode), "high": float(v.high)} for k, v in priors.items()
                }
                prior_df = pd.DataFrame(
                    [{"param": k, **priors_dump[k]} for k in sorted(priors_dump.keys())]
                )
                st.dataframe(prior_df, hide_index=True, use_container_width=True)
                st.caption("Interprétation: pour chaque trajectoire Monte Carlo, on tire un jeu de paramètres dans ces bornes.")

        if st.button("Run simulation", type="primary", disabled=not can_run, key="ramp_run"):
            cfg = RampConfig(**st.session_state["ramp_cfg"])
            start = time.time()
            with st.spinner("Simulation en cours..."):
                results = run_ramp_monte_carlo(cfg, priors=priors)
            elapsed = time.time() - start

            st.session_state["ramp_results"] = results
            st.session_state["ramp_last_run_meta"] = {"elapsed_s": elapsed, "cfg": cfg.as_dict(), "priors": priors_dump}
            st.success(f"Monte Carlo terminé en {elapsed:.2f}s.")

    with tabs[1]:
        st.subheader("ABC Ramp — Résultats")
        results = st.session_state.get("ramp_results")
        if results is None:
            st.info("Lance une simulation dans l’onglet Paramètres.")
        else:
            meta = st.session_state.get("ramp_last_run_meta", {})
            elapsed = meta.get("elapsed_s", None)
            cfg = meta.get("cfg", {})
            if elapsed is not None:
                priors_meta = meta.get("priors", None)
                prior_flag = " | priors=ON" if priors_meta else ""
                st.caption(
                    f"Run: {elapsed:.2f}s | scénario={cfg.get('scenario', 'central')} | "
                    f"months={cfg.get('months')} | n_sims={cfg.get('n_sims')} | market_noise={cfg.get('market_noise')}"
                    f"{prior_flag}"
                )

            st.markdown("### Imports & statistiques")
            c1, c2 = st.columns([2, 1])
            with c1:
                imp_plot_df = results.imports.reset_index().rename(columns={"Year": "Année"})
                fig_imp = px.line(imp_plot_df, x="Année", y="Imports_tons", markers=True, title="Imports historiques (t/an)")
                fig_imp.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
            with c2:
                st.dataframe(results.summary_market, hide_index=True, use_container_width=True)
                st.caption(
                    f"D0={results.stats.D0_tpy:.0f} t/an | σ_y={results.stats.sigma_y:.3f} | σ_m={results.stats.sigma_m:.3f}"
                )

            st.markdown("### Projection marché (scénarios, sans bruit)")
            t = results.t
            order = ["prudent", "central", "ambitieux"]
            rng_dummy = np.random.default_rng(0)
            market_paths = {
                scen: market_demand_path(
                    months=int(results.cfg.months),
                    D0_tpy=float(results.stats.D0_tpy),
                    growth_annual=float(results.growth_scenarios[scen]),
                    sigma_m=float(results.stats.sigma_m),
                    noise=False,
                    rng_local=rng_dummy,
                )
                for scen in order
            }
            proj_df = pd.DataFrame({"Month": t, **{f"Marché — {s}": market_paths[s] for s in order}})
            fig_proj = px.line(proj_df, x="Month", y=[c for c in proj_df.columns if c != "Month"])
            fig_proj.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_proj, use_container_width=True)

            st.markdown("### Courbes (paramètres)")
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_s = px.line(pd.DataFrame({"Month": t, "s(t)": results.s_t}), x="Month", y="s(t)")
                fig_s.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_range=[0, 1],
                    title="Part de marché s(t)",
                )
                st.plotly_chart(fig_s, use_container_width=True)
            with c2:
                fig_q = px.line(pd.DataFrame({"Month": t, "Q(t)": results.Q_t}), x="Month", y="Q(t)")
                fig_q.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), title="Capacité Q(t) (t/an)")
                st.plotly_chart(fig_q, use_container_width=True)
            with c3:
                lam_t = float(results.cfg.lam0) + (float(results.cfg.lam_max) - float(results.cfg.lam0)) * logistic(
                    t, float(results.cfg.k_lam), float(results.cfg.t0_lam)
                )
                fig_l = px.line(pd.DataFrame({"Month": t, "λ(t)": lam_t}), x="Month", y="λ(t)")
                fig_l.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=40, b=10),
                    title="Acquisition clients λ(t) (clients/mois)",
                )
                st.plotly_chart(fig_l, use_container_width=True)

            Dm_sel = market_paths[str(results.cfg.scenario)]
            Daddr_sel = results.s_t * Dm_sel
            df_ma = pd.DataFrame({"Month": t, "Marché": Dm_sel, "Adressable": Daddr_sel})
            fig_ma = px.line(df_ma, x="Month", y=["Marché", "Adressable"])
            fig_ma.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                title=f"Marché vs adressable — scénario '{results.cfg.scenario}'",
            )
            st.plotly_chart(fig_ma, use_container_width=True)

            st.markdown("### Trajectoire exemple (1 simulation)")
            demo_df = pd.DataFrame(
                {
                    "Month": t,
                    "Capacité": results.demo["Capacity_tpy"],
                    "Marché": results.demo["Dm_tpy"],
                    "Adressable": results.demo["Daddr_tpy"],
                    "Demande clients": results.demo["DemandClients_tpy"],
                    "Ventes": results.demo["Sales_tpy"],
                }
            )
            fig_demo = px.line(demo_df, x="Month", y=["Capacité", "Marché", "Adressable", "Demande clients", "Ventes"])
            fig_demo.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_demo, use_container_width=True)

            st.markdown("### Monte Carlo — P50 + bande ventes P10–P90")
            df = results.summary_df
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Capacity_P50_tpy"], name="Capacité P50", mode="lines"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Market_P50_tpy"], name="Marché P50", mode="lines"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Addressable_P50_tpy"], name="Adressable P50", mode="lines"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["DemandClients_P50_tpy"], name="Demande clients P50", mode="lines"))
            fig_main.add_trace(go.Scatter(x=df["Month"], y=df["Sales_P50_tpy"], name="Ventes P50", mode="lines"))
            fig_main.add_trace(
                go.Scatter(
                    x=df["Month"],
                    y=df["Sales_P90_tpy"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig_main.add_trace(
                go.Scatter(
                    x=df["Month"],
                    y=df["Sales_P10_tpy"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(0, 128, 0, 0.15)",
                    line=dict(width=0),
                    name="Ventes P10–P90",
                )
            )
            fig_main.add_hline(y=float(results.cfg.threshold_1), line_dash="dash", line_color="gray")
            fig_main.add_hline(y=float(results.cfg.threshold_2), line_dash="dash", line_color="gray")
            fig_main.update_layout(
                xaxis_title="Mois",
                yaxis_title="t/an",
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_main, use_container_width=True)

            st.markdown("### Atteinte des seuils (Ventes P50)")
            st.dataframe(results.crossings_df, hide_index=True, use_container_width=True)

            with st.expander("Courbes quantiles (P10/P50/P90/Mean) — toutes séries", expanded=False):
                labels = {
                    "Capacity_tpy": "Capacité (t/an)",
                    "Dm_tpy": "Marché (t/an)",
                    "Daddr_tpy": "Adressable (t/an)",
                    "DemandClients_tpy": "Demande clients (t/an)",
                    "Sales_tpy": "Ventes (t/an)",
                }
                for key, label in labels.items():
                    q = results.q[key]
                    dff = pd.DataFrame(
                        {
                            "Month": t,
                            "P10": q["P10"],
                            "P50": q["P50"],
                            "P90": q["P90"],
                            "Mean": q["Mean"],
                        }
                    )
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dff["Month"], y=dff["P50"], name="P50", mode="lines"))
                    fig.add_trace(go.Scatter(x=dff["Month"], y=dff["Mean"], name="Mean", mode="lines"))
                    fig.add_trace(
                        go.Scatter(x=dff["Month"], y=dff["P90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=dff["Month"],
                            y=dff["P10"],
                            mode="lines",
                            fill="tonexty",
                            fillcolor="rgba(0, 0, 255, 0.12)",
                            line=dict(width=0),
                            name="P10–P90",
                        )
                    )
                    fig.update_layout(
                        title=f"{label} — P50/Mean + bande P10–P90",
                        xaxis_title="Mois",
                        yaxis_title="t/an",
                        height=320,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("ABC Ramp — Export")
        results = st.session_state.get("ramp_results")
        if results is None:
            st.info("Lance une simulation d’abord.")
        else:
            meta = st.session_state.get("ramp_last_run_meta", {})
            csv_bytes = results.summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger résultats (CSV)",
                data=csv_bytes,
                file_name="abc_ramp_results.csv",
                mime="text/csv",
            )

            payload = {"cfg": results.cfg.as_dict()}
            if meta.get("priors", None):
                payload["priors"] = meta["priors"]
            config_json = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "Télécharger config (JSON)",
                data=config_json,
                file_name="abc_ramp_config.json",
                mime="application/json",
            )

            with st.expander("Exporter Excel (XLSX)", expanded=False):
                st.caption("Inclut paramètres + feuille résultats + chart.")
                xlsx = build_excel_bytes(results)
                st.download_button(
                    "Télécharger Excel (XLSX)",
                    data=xlsx,
                    file_name="abc_ramp_simulation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with st.expander("Exporter Monte Carlo (NPZ)", expanded=False):
                st.caption("Peut être volumineux (toutes trajectoires).")
                buf = io.BytesIO()
                arrays: dict[str, np.ndarray] = {"t": results.t}
                for k, arr in results.mc.items():
                    arrays[f"mc__{k}"] = arr
                np.savez_compressed(buf, **arrays)
                st.download_button(
                    "Télécharger MC (NPZ)",
                    data=buf.getvalue(),
                    file_name="abc_ramp_mc.npz",
                    mime="application/octet-stream",
                )


def main() -> None:
    st.set_page_config(page_title="Monte Carlo ABC", layout="wide")
    _ensure_state()

    st.title("Monte Carlo — ABC")
    st.caption("Interface pour paramétrer et lancer plusieurs simulations Monte Carlo.")

    sim_tabs = st.tabs(["Supply Chain (Stocks)", "ABC Ramp (Marché/Capacité)"])
    with sim_tabs[0]:
        _render_supply_chain()
    with sim_tabs[1]:
        _render_ramp()


if __name__ == "__main__":
    main()

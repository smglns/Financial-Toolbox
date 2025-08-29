# app_streamlit.py
import math
import streamlit as st
import matplotlib.pyplot as plt

# import your existing functions
from MonteCarlo import (
    monte_carlo_gbm, summarize_results,
    simulate_gbm_path_series
)

st.set_page_config(page_title="Finance Toolbox", layout="wide")

st.title("Finance Toolbox")
tool = st.sidebar.selectbox("Choose a tool", ["Monte Carlo (GBM)", "CAPM"])

if tool == "Monte Carlo (GBM)":
    st.header("Monte Carlo (GBM) Portfolio Simulator")

    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("Initial portfolio value", min_value=0.0, value=10000.0, step=100.0)
        mu_mode = st.radio("Expected return input type", ["Continuous (GBM μ)", "Discrete (%)"], horizontal=True)
        mu_raw = st.number_input("Expected annual return value", value=0.07, step=0.01, format="%.4f")
    with col2:
        sigma = st.number_input("Annual volatility (σ)", min_value=0.0, value=0.15, step=0.01, format="%.4f")
        years = st.number_input("Years to simulate", min_value=0.1, value=10.0, step=0.5, format="%.1f")
        steps_per_year = st.number_input("Steps per year", min_value=1, value=252, step=1)
    with col3:
        n_simulations = st.number_input("Number of simulations", min_value=100, value=10000, step=100)
        seed = st.number_input("Random seed (optional)", min_value=0, value=0, step=1)
        use_seed = st.checkbox("Use seed", value=False)

    threshold = st.number_input("Optional threshold to check P(final value < threshold)", min_value=0.0, value=0.0, step=100.0)

    # convert to GBM μ if user gave discrete %
    mu = math.log(1.0 + mu_raw) if mu_mode == "Discrete (%)" else mu_raw
    seed_val = int(seed) if use_seed else None

    if st.button("Run Simulation"):
        results = monte_carlo_gbm(S0, mu, sigma, years, int(steps_per_year), int(n_simulations), seed_val)
        summary, var, prob_below = summarize_results(results, S0, threshold if threshold > 0 else None, var_level=0.05)

        # --- Summary cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean terminal value", f"{summary['mean']:.2f}")
        m2.metric("Median", f"{summary['median']:.2f}")
        m3.metric("Std dev", f"{summary['std']:.2f}")
        m4.metric("5% VaR (vs S0)", f"{var:.2f}" if var is not None else "—")

        # Percentiles table
        st.subheader("Percentiles")
        st.write({
            "5%": f"{summary['5%']:.2f}",
            "25%": f"{summary['25%']:.2f}",
            "75%": f"{summary['75%']:.2f}",
            "90%": f"{summary['90%']:.2f}",
            "95%": f"{summary['95%']:.2f}",
        })

        if threshold > 0:
            st.info(f"Probability(final value < {threshold:.2f}) = **{prob_below:.4f}**")

        # --- Plots: histogram + sample paths side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: histogram
        ax1.hist(results, bins=50)
        ax1.set_title(f"Final Values (T={years} yrs, n={n_simulations})")
        ax1.set_xlabel("Final Portfolio Value")
        ax1.set_ylabel("Frequency")

        # Right: sample paths
        num_paths = 20
        for _ in range(num_paths):
            path = simulate_gbm_path_series(S0, mu, sigma, years, int(steps_per_year))
            ax2.plot(path, alpha=0.6)
        ax2.set_title(f"Sample GBM Paths (steps/yr={int(steps_per_year)})")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Portfolio Value")

        st.pyplot(fig)

elif tool == "CAPM":
    st.header("CAPM Expected Return")
    rf = st.number_input("Risk-free rate (decimal, e.g., 0.03)", value=0.03, step=0.005, format="%.4f")
    mode = st.radio("Provide", ["Expected market return E[Rm]", "Market risk premium (E[Rm]-Rf)"], horizontal=True)
    if mode == "Expected market return E[Rm]":
        er_m = st.number_input("E[Rm] (decimal, e.g., 0.08)", value=0.08, step=0.01, format="%.4f")
        mrp = er_m - rf
    else:
        mrp = st.number_input("Market risk premium (decimal, e.g., 0.05)", value=0.05, step=0.01, format="%.4f")
        er_m = rf + mrp
    beta = st.number_input("Asset beta (e.g., 1.2)", value=1.0, step=0.1, format="%.2f")

    er_i = rf + beta * mrp
    st.write("### Result")
    st.write(f"**Expected asset return E[Ri] = Rf + β·(E[Rm]-Rf) = {er_i:.4f} ({er_i*100:.2f}%)**")
    st.caption(f"Rf={rf:.4f}, E[Rm]={er_m:.4f}, MRP={mrp:.4f}, β={beta:.2f}")
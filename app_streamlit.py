# app_streamlit.py
import math
import streamlit as st
import matplotlib.pyplot as plt

# import your existing functions
from MonteCarlo import (
    monte_carlo_gbm, summarize_results,
    simulate_gbm_path_series,
    npv, irr, payback_period, discounted_payback_period, profitability_index
)

st.set_page_config(page_title="Finance Toolbox", layout="wide", initial_sidebar_state="expanded")

st.title("Finance Toolbox")
tool = st.sidebar.radio("Choose a tool", ("Monte Carlo (GBM)", "CAPM", "DCF Valuation", "Capital Budgeting"), index=0)

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

elif tool == "DCF Valuation":
    st.header("DCF Valuation (with Terminal Value)")

    # Input mode
    mode = st.radio("FCF input mode", ["Manual per year", "Base FCF + constant growth"], horizontal=True)

    # Core parameters
    colA, colB, colC = st.columns(3)
    with colA:
        wacc = st.number_input("Discount rate WACC (decimal)", value=0.09, step=0.005, format="%.4f")
    with colB:
        years = st.number_input("Forecast horizon (years)", min_value=1, value=5, step=1)
    with colC:
        g_term = st.number_input("Terminal growth g (decimal)", value=0.025, step=0.0025, format="%.4f")

    # Gather FCFs
    fcfs = []
    if mode == "Manual per year":
        st.subheader("Enter Free Cash Flow for each year")
        for t in range(1, int(years) + 1):
            f = st.number_input(f"FCF year {t}", value=1_000_000.0, step=50_000.0, key=f"fcf_{t}")
            fcfs.append(float(f))
    else:
        st.subheader("Base FCF + constant growth")
        base_fcf = st.number_input("Base (current) FCF (year 0)", value=1_000_000.0, step=50_000.0)
        g_fore = st.number_input("Annual growth rate for forecast years (decimal)", value=0.06, step=0.005, format="%.4f")
        f = base_fcf
        for _ in range(1, int(years) + 1):
            f = f * (1.0 + g_fore)
            fcfs.append(float(f))

    warn = (g_term >= wacc)
    if warn:
        st.warning("Terminal growth must be **less** than WACC for the Gordon Growth model to be finite.")

    # Compute PVs
    pv_fcfs = 0.0
    for t, f in enumerate(fcfs, start=1):
        pv_fcfs += f / ((1.0 + wacc) ** t)

    fcf_N = fcfs[-1] if fcfs else 0.0
    fcf_N_plus_1 = fcf_N * (1.0 + g_term)
    tv_N = (fcf_N_plus_1 / (wacc - g_term)) if (wacc > g_term) else float('inf')
    pv_tv = (tv_N / ((1.0 + wacc) ** int(years))) if (wacc > g_term) else float('inf')
    enterprise_value = pv_fcfs + pv_tv

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("PV of forecast FCFs", f"{pv_fcfs:,.2f}")
    c2.metric(f"Terminal value at year {int(years)}", "∞" if tv_N == float('inf') else f"{tv_N:,.2f}")
    c3.metric("PV of terminal value", "∞" if pv_tv == float('inf') else f"{pv_tv:,.2f}")
    st.metric("Enterprise Value (EV)", "∞" if enterprise_value == float('inf') else f"{enterprise_value:,.2f}")

    st.caption("All cash flows assumed end-of-period, nominal. Ensure WACC and growth rates are consistent with currency/inflation.")

    # Optional equity adjustments
    with st.expander("Adjust to Equity Value (optional)"):
        do_adj = st.checkbox("Compute Equity Value and Per-share")
        if do_adj and enterprise_value not in (float('inf'), float('nan')):
            cash = st.number_input("Add cash & equivalents", value=0.0, step=10000.0)
            debt = st.number_input("Subtract total debt", value=0.0, step=10000.0)
            shares = st.number_input("Shares outstanding", value=0.0, step=1000.0)
            equity_value = enterprise_value + cash - debt
            st.write(f"**Equity Value:** {equity_value:,.2f}")
            if shares > 0:
                st.write(f"**Implied Value per Share:** {equity_value / shares:,.4f}")

elif tool == "Capital Budgeting":
    st.header("Capital Budgeting (NPV, IRR, Payback, Discounted Payback, PI)")

    mode = st.radio("Cash flow input mode", ["Manual per year", "Base CF + constant growth"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        c0 = st.number_input("Initial investment at t=0 (enter negative)", value=-1_000_000.0, step=50_000.0, format="%.2f")
    with col2:
        years = st.number_input("Number of years", min_value=1, value=5, step=1)
    with col3:
        rate = st.number_input("Discount rate (decimal, e.g., 0.10)", value=0.10, step=0.005, format="%.4f")

    cfs_future = []
    if mode == "Manual per year":
        st.subheader("Enter annual cash flows")
        for t in range(1, int(years) + 1):
            cf = st.number_input(f"CF year {t}", value=250_000.0, step=25_000.0, key=f"cb_cf_{t}")
            cfs_future.append(float(cf))
    else:
        st.subheader("Generate annual cash flows")
        base_cf = st.number_input("Base cash flow (year 1)", value=250_000.0, step=25_000.0)
        g = st.number_input("Annual growth rate for years 2..N (decimal)", value=0.03, step=0.005, format="%.4f")
        cf = base_cf
        for i in range(1, int(years) + 1):
            if i > 1:
                cf = cf * (1.0 + g)
            cfs_future.append(float(cf))

    # Build full cash flow series including t=0
    cashflows = [float(c0)] + cfs_future

    if st.button("Calculate Metrics"):
        # Metrics
        npv_val = npv(rate, cashflows)
        irr_val = irr(cashflows)
        pb = payback_period(abs(c0), cfs_future)
        dpb = discounted_payback_period(abs(c0), cfs_future, rate)
        pi = profitability_index(rate, c0, cfs_future)

        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("NPV", f"{npv_val:,.2f}")
        m2.metric("IRR", "N/A" if irr_val is None else f"{irr_val*100:.2f}%")
        m3.metric("Profitability Index (PI)", f"{pi:.4f}")

        m4, m5 = st.columns(2)
        m4.metric("Payback (undiscounted)", "N/A" if pb is None else f"{pb:.2f} years")
        m5.metric("Payback (discounted)", "N/A" if dpb is None else f"{dpb:.2f} years")

        st.caption("Notes: IRR may be undefined if cash flows do not change sign once. NPV/PI use the discount rate above. Cash flows are assumed end-of-period.")

        # Optional table of cash flows and discounted cash flows
        try:
            import pandas as pd
            rows = []
            for t, cf in enumerate(cashflows):
                if t == 0:
                    pv = cf  # t=0
                else:
                    pv = cf / ((1.0 + rate) ** t)
                rows.append({"Year": t, "Cash Flow": cf, "Present Value": pv})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        except Exception:
            pass
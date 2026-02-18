```mermaid
graph TB
    START([Start])
    START --> MENU[Finance Toolbox (CLI)]

    MENU --> Q{Choose option}
    Q -->|1| MC[Monte Carlo simulation (GBM)]
    Q -->|2| CAPM[CAPM]
    Q -->|3| DCF[DCF valuation]
    Q -->|4| CB[Capital budgeting]

    %% --- Monte Carlo ---
    MC --> MC_in["Inputs:\nS0, μ, σ, years,\nsteps/yr, n_sims, seed, threshold?"]
    MC_in --> MC_run[Simulate GBM paths\n& terminal values]
    MC_run --> MC_out["Outputs:\nmean/median, std,\npercentiles, VaR,\nP(final&lt;threshold)?,\noptional plots"]

    %% --- CAPM ---
    CAPM --> CAPM_in["Inputs:\nRf, E[Rm] or MRP, β"]
    CAPM_in --> CAPM_calc[E[Ri] = Rf + β*(E[Rm]-Rf)]
    CAPM_calc --> CAPM_out["Output:\nExpected return E[Ri]"]

    %% --- DCF ---
    DCF --> DCF_in["Inputs:\nWACC, horizon N,\nFCFs or base+growth,\nterminal g, cash/debt/shares?"]
    DCF_in --> DCF_calc[PV(FCFs) + PV(Terminal Value)]
    DCF_calc --> DCF_out["Outputs:\nEnterprise Value;\noptional Equity Value\n& Per-Share"]

    %% --- Capital Budgeting ---
    CB --> CB_in["Inputs:\nInitial outlay c0,\nCFs (manual or growth),\ndiscount rate"]
    CB_in --> CB_calc[Compute NPV, IRR,\nPayback, Disc. Payback, PI]
    CB_calc --> CB_out["Outputs:\nNPV, IRR, Payback,\nDisc. Payback, PI"]
```

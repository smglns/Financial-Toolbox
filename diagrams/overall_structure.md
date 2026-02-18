```mermaid
flowchart TB

    user([User (CLI)]):::ext
    menu([P0: Finance Toolbox CLI Menu]):::proc
    
    gbm([P2: GBM Monte-Carlo]):::proc
    sum([P2a: Summary and VaR]):::proc
    viz([P3: Visualization (Matplotlib)]):::proc
    capm([P4: CAPM Calculator]):::proc
    dcf([P5a: DCF Valuation]):::proc
    cb([P5b: Capital Budgeting]):::proc
    utils([Utilities: NPV, IRR, Payback, PI]):::store
    mem([In-memory data: lists, dicts, floats]):::store
    
    user -- "parameters: S0, mu, sigma, T, steps, n, seed, threshold; rf/beta; WACC/g; cashflows..." --> menu
    menu --> gbm
    menu --> capm
    menu --> dcf
    menu --> cb
    
    gbm -- "results list[float]; path list[float]" --> sum
    gbm -- "results; paths" --> viz
    sum -- "summary dict; VaR float or None; prob_below float or None" --> user
    viz -- "charts: histogram, sample paths" --> user
    
    capm -- "rf, er_m, mrp, beta, er_i" --> user
    dcf -- "EV; PV_FCFs; TV; PV_TV; optional equity/share" --> user
    cb --> utils
    utils --> cb
    cb -- "NPV; IRR or None; PB; DPB; PI" --> user
    
    gbm --- mem
    sum --- mem
    dcf --- mem
    cb --- mem
    
    classDef proc fill:#eef7ff,stroke:#3b82f6,stroke-width:1px,color:#0b2748;
    classDef store fill:#fef9c3,stroke:#ca8a04,stroke-width:1px,color:#3b2f0b;
    classDef ext fill:#ecfdf5,stroke:#10b981,stroke-width:1px,color:#064e3b;
```
```mermaid
flowchart LR
    user([User]):::ext
    
    subgraph P0 [P0: Collect Inputs]
        in0[/CLI inputs/]:::store
    end
    
    subgraph P2 [P2: GBM Simulation and Summary]
        p2a[standard_normal]:::proc
        p2b[simulate_gbm_one_path or path_series]:::proc
        p2c[monte_carlo_gbm]:::proc
        p2d[summarize_results]:::proc
        results[/results: list float/]:::store
        results_sorted[/results_sorted: list float/]:::store
        summary[/summary: dict/]:::store
        var[/var: float or None/]
        prob[/prob_below: float or None/]
    end
    
    subgraph P3 [P3: Visualization]
        charts[(charts: hist, paths)]:::store
    end
    
    subgraph P4 [P4: CAPM]
        capm_out[/rf, er_m, mrp, beta, er_i/]:::store
    end
    
    subgraph P5 [P5: DCF and Capital Budgeting]
        fcfs[/fcfs: list float/]:::store
        cashflows[/cashflows: list float/]:::store
        p5a[DCF compute EV]:::proc
        p5b[Capital budgeting metrics]:::proc
        p5u[npv, irr, payback, PI]:::proc
        metrics[/NPV, IRR or None, PB, DPB, PI/]:::store
        ev[/EV, PV_FCFs, TV, PV_TV/]:::store
    end
    
    user -->|"S0, mu, sigma, years, steps, n, seed, threshold"| in0
    user -->|"rf, er_m or mrp, beta"| P4
    user -->|"WACC, years, fcfs or base plus g, g_term"| P5
    user -->|"c0, years, rate, cashflows or base plus g"| P5
    
    in0 --> p2a --> p2b --> p2c --> results --> p2d
    p2d --> results_sorted
    p2d --> summary
    p2d --> var
    p2d --> prob
    results --> P3 --> charts --> user
    summary --> user
    var --> user
    prob --> user
    
    fcfs --> p5a --> ev --> user
    cashflows --> p5b --> p5u --> metrics --> user
    
    classDef proc fill:#eef7ff,stroke:#3b82f6,stroke-width:1px,color:#0b2748;
    classDef store fill:#fef9c3,stroke:#ca8a04,stroke-width:1px,color:#3b2f0b;
    classDef ext fill:#ecfdf5,stroke:#10b981,stroke-width:1px,color:#064e3b;
```
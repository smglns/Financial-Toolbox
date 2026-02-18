
flowchart TD
    A([Start]) --> B[Inputs: S0, μ, σ, years, steps_per_year, n_simulations, seed?]
    B --> C{seed is not None?}
    C -- yes --> D[Set random.seed(seed)]
    C -- no --> E[Skip seeding]
    D --> F[results ← []]
    E --> F[results ← []]
    F --> G{i = 1..n_simulations}
    G --> H[final_price ← simulate_gbm_one_path(S0, μ, σ, years, steps_per_year)]
    H --> I[Append final_price to results]
    I --> G
    G -->|done| J[[Return results]]


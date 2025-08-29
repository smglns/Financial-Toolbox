import random, math
from statistics import mean, stdev, median

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

def standard_normal():
    """Return one standard normal sample using Box-Muller."""
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z

def simulate_gbm_one_path(S0, mu, sigma, years, steps_per_year):
    """
    Simulate a single GBM path and return final price after 'years'.
    Uses the exact discretized GBM step:
      S_{t+dt} = S_t * exp((mu - 0.5*sigma^2) dt + sigma * sqrt(dt) * Z)
    """
    dt = 1.0 / steps_per_year
    steps = int(years * steps_per_year)
    S = S0
    for _ in range(steps):
        z = standard_normal()
        S = S * math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
    return S

def simulate_gbm_path_series(S0, mu, sigma, years, steps_per_year):
    """
    Simulate a single GBM path and return the full series [S0, S1, ..., S_T].
    """
    dt = 1.0 / steps_per_year
    steps = int(years * steps_per_year)
    path = [S0]
    S = S0
    for _ in range(steps):
        z = standard_normal()
        S = S * math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
        path.append(S)
    return path

def monte_carlo_gbm(S0, mu, sigma, years, steps_per_year=252, n_simulations=10000, seed=None):
    """
    Run Monte Carlo using GBM for n_simulations and return list of terminal values.
    steps_per_year defaults to 252 (trading days) for finer resolution.
    """
    if seed is not None:
        random.seed(seed)
    results = []
    for i in range(n_simulations):
        final_price = simulate_gbm_one_path(S0, mu, sigma, years, steps_per_year)
        results.append(final_price)
        # optional: print progress every so often
        # if (i+1) % (n_simulations//10 or 1) == 0:
        #     print(f"Sim {i+1}/{n_simulations} done")
    return results

def summarize_results(results, S0, threshold=None, var_level=0.05):
    n = len(results)
    avg = mean(results)
    sd = stdev(results) if n > 1 else 0.0
    med = median(results)
    results_sorted = sorted(results)
    p5 = results_sorted[int(0.05 * n)] if n>0 else None
    p10 = results_sorted[int(0.10 * n)] if n>0 else None
    p25 = results_sorted[int(0.25 * n)] if n>0 else None
    p75 = results_sorted[int(0.75 * n)] if n>0 else None
    p90 = results_sorted[int(0.90 * n)] if n>0 else None
    p95 = results_sorted[int(0.95 * n)] if n>0 else None

    summary = {
        "n": n,
        "mean": avg,
        "std": sd,
        "median": med,
        "5%": p5,
        "10%": p10,
        "25%": p25,
        "75%": p75,
        "90%": p90,
        "95%": p95
    }

    # Value at Risk (loss relative to S0): VaR at var_level
    # VaR = S0 - quantile(var_level)
    quantile_index = int(var_level * n)
    quantile_val = results_sorted[quantile_index] if n>0 else None
    var = None
    if quantile_val is not None:
        var = S0 - quantile_val

    # prob below threshold if provided
    prob_below = None
    if threshold is not None:
        count = sum(1 for x in results if x < threshold)
        prob_below = count / n

    return summary, var, prob_below

def capm_expected_return():
    
    print("\n--- CAPM: Expected Return Calculator ---")
    # Risk-free rate
    rf = float(input("Risk-free rate (decimal, e.g., 0.03 for 3%): ").strip())
    # Choose to enter market expected return or market risk premium
    mode = input("Provide (1) market expected return E[Rm] or (2) market risk premium (E[Rm]-Rf)? Enter 1 or 2: ").strip()
    if mode == "1":
        er_m = float(input("Expected market return E[Rm] (decimal, e.g., 0.08 for 8%): ").strip())
        mrp = er_m - rf
    else:
        mrp = float(input("Market risk premium (E[Rm]-Rf) (decimal, e.g., 0.05 for 5%): ").strip())
        er_m = rf + mrp
    beta = float(input("Asset beta (e.g., 1.2): ").strip())

    er_i = rf + beta * mrp
    print("\n--- CAPM Result ---")
    print(f"Risk-free rate (Rf): {rf:.4f}")
    print(f"Expected market return (E[Rm]): {er_m:.4f}")
    print(f"Market risk premium (E[Rm]-Rf): {mrp:.4f}")
    print(f"Beta (β): {beta:.4f}")
    print(f"\nExpected asset return E[Ri] = Rf + β*(E[Rm]-Rf) = {er_i:.4f} ({er_i*100:.2f}%)")
    return {
        "rf": rf,
        "er_m": er_m,
        "mrp": mrp,
        "beta": beta,
        "er_i": er_i
    }

if __name__ == "__main__":
    while True:
        print("\n=== Finance Toolbox ===")
        print("1) Monte Carlo (GBM) Portfolio Simulator")
        print("2) CAPM Expected Return")
        print("q) Quit")
        choice = input("Choose an option: ").strip().lower()

        if choice == "1":
            # Interactive inputs for Monte Carlo GBM
            S0 = float(input("Initial portfolio value (e.g. 10000): "))
            mu_input_mode = input("Is your expected return (mu) a (1) continuous rate or (2) discrete percent? Enter 1 or 2: ").strip()
            mu_raw = float(input("Expected annual return value (e.g. 0.07 for 7%): "))
            if mu_input_mode == "2":
                # convert discrete g to continuous log-return
                mu = math.log(1.0 + mu_raw)
            else:
                mu = mu_raw
            sigma = float(input("Annual volatility (decimal, e.g. 0.15 for 15%): "))
            years = float(input("Number of years to simulate (e.g. 10): "))
            steps_per_year = int(input("Steps per year (e.g. 252 for daily, 12 for monthly): "))
            n_simulations = int(input("Number of simulations (e.g. 10000): "))
            seed_input = input("Random seed (press Enter to skip): ")
            seed = int(seed_input) if seed_input.strip() else None
            threshold_input = input("Optional threshold to check final value probability (press Enter to skip): ")
            threshold = float(threshold_input) if threshold_input.strip() else None

            results = monte_carlo_gbm(S0, mu, sigma, years, steps_per_year, n_simulations, seed)
            summary, var, prob_below = summarize_results(results, S0, threshold, var_level=0.05)

            print("\n--- Monte Carlo GBM Summary ---")
            print(f"Simulations: {summary['n']}")
            print(f"Mean terminal value: {summary['mean']:.2f}")
            print(f"Median terminal value: {summary['median']:.2f}")
            print(f"Std dev of terminal values: {summary['std']:.2f}")
            print(f"5th percentile: {summary['5%']:.2f}")
            print(f"25th percentile: {summary['25%']:.2f}")
            print(f"75th percentile: {summary['75%']:.2f}")
            print(f"90th percentile: {summary['90%']:.2f}")
            print(f"95th percentile: {summary['95%']:.2f}")

            if var is not None:
                print(f"\nEstimated 5% VaR (loss relative to initial S0): {var:.2f} (i.e. S0 - 5th percentile)")

            if threshold is not None:
                print(f"Probability final value < {threshold}: {prob_below:.4f}")

            # --- Optional Visualization ---
            if _HAS_MPL:
                viz_choice = input("\nShow visuals? (y/n): ").strip().lower()
                if viz_choice == "y":
                    try:
                        # Ask how many sample paths to draw for the right subplot
                        num_paths = input("How many sample paths to plot? (e.g., 20): ").strip()
                        num_paths = int(num_paths) if num_paths else 20
                        num_paths = max(1, min(num_paths, 100))  # cap for readability
                        
                        # Create a single figure with two subplots side-by-side
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Left: histogram of terminal values
                        ax1.hist(results, bins=50, edgecolor="black")
                        ax1.set_title(f"Final Values (T={years} yrs, n={n_simulations})")
                        ax1.set_xlabel("Final Portfolio Value")
                        ax1.set_ylabel("Frequency")
                        
                        # Right: sample GBM paths
                        for _ in range(num_paths):
                            path = simulate_gbm_path_series(S0, mu, sigma, years, steps_per_year)
                            ax2.plot(path, alpha=0.6)
                        ax2.set_title(f"Sample GBM Paths (steps/yr={steps_per_year})")
                        ax2.set_xlabel("Time Steps")
                        ax2.set_ylabel("Portfolio Value")
                        
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        print(f"[Visualization] Combined plot error: {e}")
            else:
                print("\nMatplotlib not available. To enable visualizations, install it with:")
                print("  pip install matplotlib")

        elif choice == "2":
            capm_expected_return()

        elif choice == "q":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")
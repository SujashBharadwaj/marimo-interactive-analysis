# Marimo reactive notebook
# Email: 23f1000914@ds.study.iitm.ac.in   <- required by rubric (present in raw source)

import marimo as mo

app = mo.App()

# --- Cell 1: Imports ---------------------------------------------------------
# Data flow note:
# - Exposes numpy/pandas/matplotlib to downstream cells.
@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    return np, pd, plt, stats

# --- Cell 2: Widget ----------------------------------------------------------
# Data flow note:
# - noise_slider drives the synthetic-data noise level used in Cell 3.
@app.cell
def _():
    noise_slider = mo.ui.slider(start=0, stop=100, value=20, step=5, label="Noise level (%)")
    mo.md("### Controls").show()
    noise_slider
    return noise_slider

# --- Cell 3: Data generation (depends on widget) -----------------------------
# Data flow note:
# - Uses noise_slider.value to generate y = 3x + 5 + ε, where ε ~ N(0, σ).
# - Exposes df for downstream analysis/plots.
@app.cell
def _(np, pd, noise_slider):
    rng = np.random.default_rng(42)
    n = 200
    x = np.linspace(0, 10, n)
    sigma = max(1e-9, (noise_slider.value / 100) * 5.0)  # scale noise by slider
    eps = rng.normal(0, sigma, size=n)
    y = 3 * x + 5 + eps
    df = pd.DataFrame({"x": x, "y": y})
    df
    return df, sigma

# --- Cell 4: Statistics (depends on df) --------------------------------------
# Data flow note:
# - Computes Pearson r and simple linear regression on df from Cell 3.
@app.cell
def _(df, stats, np):
    r, p = stats.pearsonr(df["x"], df["y"])
    slope, intercept, r_val, p_val, stderr = stats.linregress(df["x"], df["y"])
    metrics = {
        "pearson_r": r,
        "p_value": p,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_val**2,
    }
    metrics
    return metrics

# --- Cell 5: Dynamic Markdown (depends on widget + stats) --------------------
# Data flow note:
# - Reactive summary that updates whenever noise_slider.value or metrics change.
@app.cell
def _(mo, noise_slider, metrics, sigma):
    mo.md(
        f"""
### Results (Reactive)
- **Noise:** {noise_slider.value}% (σ ≈ {sigma:.3f})
- **Fit:** y ≈ {metrics['slope']:.3f} x + {metrics['intercept']:.3f}
- **Correlation (Pearson r):** {metrics['pearson_r']:.3f}
- **R²:** {metrics['r_squared']:.3f}

> Increase the slider to inject more noise and watch the correlation and R² drop.
"""
    )
    return

# --- Cell 6: Plot (depends on df + regression) --------------------------------
# Data flow note:
# - Scatter of (x,y) with fitted line using slope/intercept from Cell 4.
@app.cell
def _(plt, df, metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["x"], df["y"], s=12, alpha=0.7)
    xline = df["x"]
    yhat = metrics["slope"] * xline + metrics["intercept"]
    ax.plot(xline, yhat, linewidth=2)
    ax.set_title("Relationship between x and y (reactive)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig
    return fig

if __name__ == "__main__":
    app.run()

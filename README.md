# QARM II – FIRE Planner

A Streamlit web app for a **FIRE (Financial Independence, Retire Early)** client. For a client profile, it shows the probability of reaching the FIRE capital by a target age. The money is invested in a constrained mean-variance portfolio whose risk tolerance γ adapts every year.

- **Universe**: one ETF per asset class from Yahoo Finance (monthly total returns): SPY (US equities), SMICHA.SW (Swiss equities, CHF), EEM (emerging-markets equities), VNQ (US real estate), IEF (sovereign bonds), LQD (corporate bonds), TIP (US inflation-linked bonds) and GLD (gold). Before 2013 Yahoo has no dividends for Zurich-listed ETFs, so SMICHA.SW is backfilled with the SMI total-return index (SMIC.SW).
- **Currency and inflation** (project layer): USD returns are converted to CHF, hedged (default) or unhedged, and deflated with the Swiss CPI. μ and Σ are estimated on these real CHF returns.
- **Risk-free rate**: real rate for the remaining horizon, interpolated on the SNB Confederation spot curve; used only to discount future savings.
- **Optimisation (LN 1)**: the γ-problem `min ½ x'Σx − γ x'μ` subject to `1'x = 1`, `0 ≤ x_i ≤ 40%` and `x_GLD ≥ 10%`. It is solved numerically with `scipy.optimize.minimize`, as in the course labs (Markowitz LN1, Stability LN3). The σ-problem maps a volatility target to γ.
- **Estimation error (LN 3)**: Michaud resampling of the constrained frontier, following slides 11-12 (1'000 draws). A Ledoit–Wolf shrinkage comparison is shown in the technical details.
- **Dynamic risk**: `γ_t = γ_min + (γ_0 − γ_min) · H_t / (W_t + H_t)`, where `H_t` is the present value of the remaining savings. Once `W_t ≥ K`, `γ_t = γ_min`.
- **Probability of FIRE** (project layer, not from the course): Monte Carlo simulation of wealth with γ_t computed on each path and jointly lognormal yearly returns. The probability is the share of paths with `W_T ≥ K` at the target age, shown with its Monte Carlo standard error, a fan chart and the distribution of final wealth.

Assumptions: amounts in today's CHF, currency hedge by covered interest parity (no cross-currency basis), no transaction costs. The *Technical details* tab lists them all.

## Files

| File | Content |
|---|---|
| `app_fire.py` | Streamlit UI only |
| `data.py` | asset configuration, price download, CHF conversion, hedging, deflation, annualised μ and Σ, Ledoit–Wolf δ |
| `market.py` | USD/CHF and US 1-year rate (FRED), Confederation curve and Swiss CPI (SNB), risk-free rate |
| `data/` | local copies used if FRED or the SNB cannot be reached |
| `optimizer.py` | constraints (`CONFIG`), γ-problem, σ-problem, resampling |
| `fire.py` | FIRE capital, dynamic γ, Monte Carlo simulation of wealth, probability of FIRE |
| `tests/` | sanity tests (synthetic data, no network) |
| `app.py`, `download_data.py`, `etf_returns.csv` | first prototype, kept for reference |

## Run locally

Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_fire.py
```

The first run downloads the data and computes the resampled portfolios, which takes about a minute. After that, results are cached for one day.

Tests:

```bash
pip install pytest
python -m pytest
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub. `docs/` (course material) is in `.gitignore` and must not be pushed.
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **Create app** and choose **Deploy a public app from GitHub**.
4. Select the repository, the branch and **Main file path** `app_fire.py`.
5. Under **Advanced settings**, choose the Python version (3.9 or later). No secrets are needed.
6. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app.
7. Optional: in the app settings, set **Who can view this app** to public, and share the URL.

The app downloads market data at runtime, so the deployed app always uses the latest data.

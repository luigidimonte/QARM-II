# QARM II – FIRE Planner

A Streamlit web app for a **FIRE (Financial Independence, Retire Early)** client. For a client profile, it shows the probability of reaching the FIRE capital by a target age. The money is invested in a constrained mean-variance portfolio whose risk tolerance γ adapts every year.

- **Universe**: one ETF per asset class, downloaded fresh from Yahoo Finance (monthly, dividends reinvested). The ETFs are SPY (US equities), EWL (Swiss equities), EEM (emerging-markets equities), IEF (sovereign bonds), LQD (corporate bonds) and GLD (gold).
- **Optimisation (LN 1)**: the γ-problem `min ½ x'Σx − γ x'μ` subject to `1'x = 1`, `0 ≤ x_i ≤ 40%` and `x_GLD ≥ 10%`. It is solved numerically with cvxpy. The σ-problem maps a volatility target to γ.
- **Estimation error (LN 3)**: Michaud resampling of the constrained frontier. A Ledoit–Wolf shrinkage comparison is shown in the technical details.
- **Dynamic risk**: `γ_t = γ_min + (γ_0 − γ_min) · H_t / (W_t + H_t)`, where `H_t` is the present value of the remaining savings. Once `W_t ≥ K`, `γ_t = γ_min`.
- **Probability of FIRE**: the first two moments of wealth are propagated year by year, then a lognormal is fitted. No Monte Carlo.

Assumptions: single currency (FX ignored), real terms (exact Fisher relation), no transaction costs.

## Files

| File | Content |
|---|---|
| `app_fire.py` | Streamlit UI only |
| `data.py` | download, monthly returns, annualised μ and Σ, Ledoit–Wolf δ |
| `optimizer.py` | constraints (`CONFIG`), γ-problem, σ-problem, resampling |
| `fire.py` | FIRE capital, dynamic γ, glide path and trades, probability of FIRE |
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

The first run downloads the data and computes the resampled portfolios, which takes a few seconds. After that, results are cached for one day.

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

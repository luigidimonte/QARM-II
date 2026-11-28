# QARM II – FIRE Planner

A Streamlit web app for a **FIRE (Financial Independence, Retire Early)** client. For a client profile, it shows the probability of reaching the FIRE capital by a target age. The money is invested in a constrained mean-variance portfolio whose risk tolerance γ adapts every year.

The client answers a short questionnaire (age, income and expenses, wealth, investment preferences, assumptions) and gets three tabs:

- **Client profile**: the answers used, the FIRE capital, and a fan chart of simulated wealth with sample paths.
- **Portfolio**: today's allocation, how the portfolio changes over time (animated glide path) and the trades after the first year.
- **Probability of FIRE**: the chance of reaching the FIRE capital by the target age, the distribution of wealth at that age, and a *What if* comparison (savings rate, target age; *Reset to initial answers* restores the questionnaire values).

## Assumptions and methodology

All amounts are in **today's CHF** (real terms). Returns are real CHF returns, so nothing is inflated or deflated by hand elsewhere: inflation enters only through the CPI deflation of past returns and through the real risk-free rate used to discount future savings.

### Data and estimates

- **Universe**: one ETF per asset class, monthly total returns (dividends reinvested) from Yahoo Finance: SPY (US equities), SMICHA.SW (Swiss equities, CHF), EEM (emerging-markets equities), VNQ (US real estate), IEF (sovereign bonds, US Treasuries 7-10y), LQD (corporate bonds, investment grade), TIP (US inflation-linked bonds) and GLD (gold).
- **Swiss equities**: Yahoo has no dividends before 2013 for ETFs listed in Zurich, so before 2013 SMICHA.SW follows the SMI total-return index (SMIC.SW), rescaled to join without a jump.
- **Sample**: month-end prices on the last common trading day of each month, common sample of all assets, without the current (incomplete) month and without months whose CPI is not published yet. At the time of writing: Dec 2004 - Oct 2026, T = 263 months (the app shows the current window).
- **Estimates**: μ = 12 × monthly mean and Σ = 12 × monthly covariance of the real CHF returns.
- **Expected returns are historical estimates** (project assumption). μ is the historical arithmetic mean over the sample window. The window includes a long equity bull market and the gold rally, so μ is optimistic as a forecast. The app labels the expected return "historical estimate".

### Currency hedging and inflation (project assumptions, not from the course)

- **Unhedged** USD assets: `R_CHF = (1 + R_USD) · S_{t+1} / S_t − 1`, with S = CHF per USD (FRED DEXSZUS) on the return dates.
- **Hedged** USD assets (default): `R_CHF = (1 + R_USD) · ((1 + r_CHF,1y) / (1 + r_USD,1y))^(1/12) − 1`, a 1-year forward rolled every month (covered interest parity). r_CHF is the 1-year node of the SNB Confederation curve, r_USD the 1-year Treasury yield (FRED DGS1), both known at the start of the month. CHF-listed assets are never hedged.
- **Hedging is an approximation**: the cross-currency basis is ignored and the hedge covers 100% of the initial value, not the gain or loss of the month.
- **Hedge cost and franc drift**: μ of hedged returns contains the average hedge cost over the sample, not today's value; unhedged μ of USD assets contains the past drift of the franc against the dollar, i.e. assumes that it continues.
- **Currency risk that stays**: only the USD/CHF leg is hedged. Emerging-market equities (EEM) are traded in USD but keep the risk of their local currencies against the dollar.
- **Inflation**: CHF returns are deflated with the actual Swiss CPI (SNB cube `plkopr`), so inflation-linked bonds, gold and real estate get their real characteristics from the data. TIPS protect against US inflation, not Swiss inflation.
- The hedging choice is a questionnaire answer; after the plan is shown it is read-only (change it with *Change my answers*).

### Risk-free rate (SNB Confederation curve)

Spot rates of Swiss Confederation bonds (SNB cube `rendeiduebm`, continuation of `rendoblim`), month-end value of the latest month, interpolated linearly in the discount factor (Fixed Income and Credit Risk course):

```
Z_i = (1 + y_i)^(−T_i),   Z(h) = Z_{i−1} τ/Δ + Z_i (Δ − τ)/Δ,   Δ = T_i − T_{i−1},  τ = T_i − h
r(h) = Z(h)^(−1/h) − 1,   real rate  r_f(h) = (1 + r(h)) / (1 + π) − 1
```

π is the expected inflation (Advanced settings, default 1%); horizons are limited to the nodes (1-30 years). The SNB publishes spot rates, so no par-yield conversion is needed. The rate can also be set manually (flat real rate). It is used only to discount future savings (H_t, below), with the remaining horizon h_t of each year. FRED and SNB data are downloaded at runtime; the copies in `data/` are used if a source cannot be reached (the app then shows a warning).

### Portfolio choice: γ-problem and σ-problem (LN 1)

```
x*(γ) = argmin_x  ½ x'Σx − γ x'μ    subject to  1'x = 1,  l_i ≤ x_i ≤ u_i  (l_i ≥ 0)
```

- γ ≥ 0 is the risk tolerance; γ = 0 is the constrained minimum-variance portfolio. There is no closed form because of the bounds: the problem is solved as a QP with `scipy.optimize.minimize` and `LinearConstraint` (vector bounds), as in the course labs (Markowitz LN1, Stability LN3), with a tight tolerance. A solution is accepted only if every constraint holds within 1e-8; otherwise the solver is restarted once from a different starting point, and a plain-language error is shown if it fails again.
- Using excess returns μ − r_f·1 gives the same x* because 1'x = 1, so r_f does not change the optimal weights.
- **σ-problem**: γ such that σ(x*(γ)) = σ*, solved with `fsolve`. Where tight bounds make σ(γ) flat over a range of γ (the optimum stays at a corner of the constraints), `fsolve` sees no slope, so a bisection on γ is used as a fallback (σ(γ) is continuous and non-decreasing). The end of the frontier is the maximum-return portfolio under the bounds.
- **Default investment rules** (project assumption, not from the course): at most **30% per asset** for diversification; **gold capped at 15%** because its historical return in the sample window (2004-2026) is exceptionally high for an asset without cash flows. There is **no minimum** on any asset, including gold; the gold cap applies only if gold is in the universe. Long-only and full investment always apply and cannot be changed.
- **Client-defined rules**: the questionnaire asks *"Do you have preferences about where you would like to be invested?"*. With *Yes*, the client sets a minimum and a maximum weight for each asset class (0%-100%, steps of 5%; a maximum of 0% excludes it), starting from the defaults. The rules are checked before any computation (each minimum ≤ its maximum, values in [0%, 100%], minima add up to at most 100%, maxima to at least 100%); if they cannot be met, the questionnaire names the asset class and does not continue.

### Estimation error: resampling (LN 3, slides 11-12; Michaud 2007)

1. Draw T monthly returns (T = number of historical observations) from a **joint normal i.i.d.** process with the means, variances and covariances of the data (`np.random.multivariate_normal`); compute the sampled mean vector μ_k and covariance matrix Σ_k (annualised).
2. Compute the constrained frontier (γ-problem) with μ_k and Σ_k on a grid of γ values, the same for all draws (40 γ values whose point-estimate portfolios are evenly spaced in volatility).
3. Repeat for K = 1'000 draws (seed 42).
4. For each γ, average the weights across draws. Averages of feasible portfolios satisfy the same linear constraints. Risk and return of the averaged portfolios are evaluated with the original μ and Σ.

Weights at any γ are interpolated linearly between grid points (convex combinations, hence feasible). The risk targets of the plan are hit on this resampled frontier: σ-problem on the averaged weights, with the original Σ.

The resampling and the γ grid are the heavy step (about 1.5 minutes): they are computed **once, when the questionnaire is submitted**, and cached. They depend only on the data sample, the hedging choice, the investment rules, K and the seed (fixed at 42); changing anything else (age, income, expenses, wealth, FIRE age, withdrawal rate, inflation, risk range, the *What if* sliders) never recomputes them.

### Dynamic risk tolerance γ_t (project layer)

```
H_t = c (1 − (1 + r_f)^(−h_t)) / r_f
γ_t = γ_min + (γ_0 − γ_min) · H_t / (W_t + H_t),     γ_t = γ_min if W_t ≥ K
```

- c = income − expenses (yearly savings), K = expenses / SWR (FIRE capital), h_t = years left to the target age. H_t is the value today of the savings still to come, discounted at the real risk-free rate for the remaining horizon. At and after the target age h_t ≤ 0, so H_t = 0 and γ_t = γ_min.
- γ_0 and γ_min come from the risk range through the σ-problem on the resampled frontier: the portfolio actually held at γ_0 has the highest risk σ_max, the one at γ_min the lowest risk σ_min. By default σ_min = volatility of the minimum-risk portfolio + 0.5 pp and σ_max = 90% of the highest feasible volatility; the client may set the range manually (Advanced settings). Levels outside the feasible range are limited to it, with a note.
- Each year, on each simulated path: (W_t, h_t) → r_f → H_t → γ_t → resampled constrained portfolio → trades.

### Monte Carlo simulation of wealth (project assumption, not from the course)

```
W_{t+1} = W_t (1 + x_t'R_t) + c,    x_t = x(γ_t(W_t, h_t))       before the target age
W_{t+1} = (W_t − e)(1 + x_t'R_t),    x_t = x(γ_min)                from the target age (0 once W_t ≤ e)
p̂ = (1/N) Σ_k 1{W_T^(k) ≥ K},       SE = sqrt(p̂ (1 − p̂) / N)
```

- N = 10'000 paths, yearly steps until age 95. γ_t is computed on each path from that path's own wealth; the portfolio is rebalanced to its target every year and the savings c enter at the end of the year. From the target age the yearly expenses e are withdrawn at the start of each year.
- Yearly gross asset returns are jointly lognormal, with mean and covariance matched exactly to the point estimates μ and Σ of the real CHF returns (not the resampled ones), independent across years. The moments are treated as known: parameter uncertainty enters only through the resampled weights.
- T is the target age: the probability is the share of paths with wealth of at least K at that age, not of having crossed K before. The standard error measures simulation noise only, not model or estimation error.
- Results use a fixed random seed of 42 (resampling and Monte Carlo), so the same answers always give the same simulated markets. The current plan and the *What if* alternative share them (common random numbers), so their difference is not blurred by simulation noise.

### All assumptions: course vs project layer

| Topic | Assumption | Source |
|:---|:---|:---|
| Portfolio choice | γ-problem and σ-problem, fully invested, long-only, minimum and maximum weight per asset | Course (LN 1) |
| Estimation error | Resampled frontier, joint normal i.i.d. draws (Michaud 2007) | Course (LN 3, slides 11-12) |
| Yield curve | Interpolation linear in the discount factor | Course (Fixed Income and Credit Risk) |
| Investment rules | Default 30% cap per asset; gold capped at 15% (exceptionally high historical return for an asset without cash flows); no minimum; client-defined minimum and maximum per asset class | Project layer |
| Expected returns | Historical arithmetic mean over the sample window, optimistic as a forecast | Project layer |
| Spot rates | The SNB publishes spot rates, so no par-yield conversion is needed | Project layer |
| USD 1-year rate | Treasury yield from FRED (DGS1), used as the 1-year spot rate | Project layer |
| Hedging | 1-year forward rolled monthly, no cross-currency basis, 100% of the initial value | Project layer |
| Remaining currency risk | Only USD/CHF is hedged; EEM keeps its local currencies against the dollar | Project layer |
| Franc drift and hedge cost | μ contains the average of the sample, hedged or unhedged, not today's value | Project layer |
| Inflation | Returns deflated with the Swiss CPI; TIPS cover US, not Swiss, inflation | Project layer |
| Expected inflation | Used only in the real risk-free rate | Project layer |
| Dynamic γ_t | Human capital rule computed on each path from its own wealth | Project layer |
| Monte Carlo on wealth | Lognormal yearly returns, point estimates treated as known | Project layer |
| Cash flows | Savings, K and spending constant in today's CHF; savings at the end of the year | Project layer |
| After FIRE | γ = γ_min; the expenses are withdrawn at the start of each year until 95; a path with W_t ≤ e runs out of money (wealth 0) | Project layer |
| Swiss equities | SMICHA.SW, SMI total-return index before 2013 | Project layer |
| Costs | No transaction costs and no taxes | Project layer |

## Files

| File | Content |
|---|---|
| `app_fire.py` | Streamlit UI only |
| `data.py` | asset configuration, price download, CHF conversion, hedging, deflation, annualised μ and Σ |
| `market.py` | USD/CHF and US 1-year rate (FRED), Confederation curve and Swiss CPI (SNB), risk-free rate |
| `data/` | local copies used if FRED or the SNB cannot be reached |
| `optimizer.py` | default rules (`CONFIG`), per-asset bounds and feasibility check, γ-problem, σ-problem, resampling |
| `fire.py` | FIRE capital, dynamic γ, Monte Carlo simulation of wealth, probability of FIRE |
| `ui_style.py` | colours, chart template and CSS |
| `tests/` | sanity tests (synthetic data, no network) |
| `download_data.py`, `etf_returns.csv` | first prototype (not used by the app), kept for reference |

## Run locally

Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_fire.py
```

The questionnaire itself is instant (market data are downloaded once and cached for one day). Showing the plan computes the resampled portfolios (1'000 draws), which takes about 1.5 minutes; it runs again only for a new combination of currency hedging and investment rules. Moving the *What if* sliders takes about a second.

Tests:

```bash
pip install pytest
python -m pytest
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub. Course material (`docs/`, `lab/`, PDFs, notebooks) is in `.gitignore` and must not be pushed.
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **Create app** and choose **Deploy a public app from GitHub**.
4. Select the repository, the branch and **Main file path** `app_fire.py`.
5. Under **Advanced settings**, choose the Python version (3.9 or later). No secrets are needed.
6. Click **Deploy**. Streamlit installs `requirements.txt` and starts the app.
7. Optional: in the app settings, set **Who can view this app** to public, and share the URL.

The app downloads market data at runtime, so the deployed app always uses the latest data.

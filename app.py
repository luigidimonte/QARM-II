import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco

st.set_page_config(page_title="FIRE Portfolio Planner", layout="wide")
st.title("FIRE Portfolio Planner")

# ---------- Asset universe ----------
CORE = ["SPY", "EFA", "EEM", "IEF", "LQD", "TIP", "GLD", "VNQ"]
NAMES = {
    "SPY": "US equities", "EFA": "Developed ex-US equities", "EEM": "Emerging equities",
    "IEF": "US Treasuries 7-10y", "LQD": "Corporate bonds", "TIP": "Inflation-linked bonds",
    "GLD": "Gold", "VNQ": "Real estate (REITs)",
}
GAMMAS = np.linspace(-0.5, 1.5, 100)   # grid of gamma values (as in Lab 3)
k = 100                                # number of resampling simulations

# ---------- 1. Client profile ----------
with st.sidebar:
    st.header("Client profile")
    age = st.number_input("Current age", 18, 70, 30)
    fire_age = st.number_input("Target FIRE age", age + 1, 80, max(50, age + 1))
    capital = st.number_input("Current capital (€)", 0, 10_000_000, 50_000, step=5_000)
    saving = st.number_input("Monthly saving (€)", 0, 50_000, 1_500, step=100)
    expenses = st.number_input("Annual expenses in FIRE (€, today's value)", 5_000, 500_000, 30_000, step=1_000)
    swr = st.slider("Safe withdrawal rate (%)", 2.5, 5.0, 4.0, 0.25) / 100
    inflation = st.slider("Expected inflation (%)", 0.0, 5.0, 2.5, 0.25) / 100

# ---------- 2. Data (Lab 3) ----------
@st.cache_data
def load_data():
    data = pd.read_csv('etf_returns.csv')                 #load csv data
    data['date'] = pd.to_datetime(data['date'])           #transform date column in a datetime object
    data = data.set_index('date')                         #set the date as index of the dataframe
    data_pivot = data.pivot(columns='ticker', values='daily_return')
    data_pivot = np.log(1+data_pivot)
    return data_pivot

data_pivot = load_data()[CORE]

vol = np.array(data_pivot.std()) * np.sqrt(252)
ticker = np.array(data_pivot.columns)
mu = np.array((data_pivot.mean() +1)**252 - 1)
corr_matrix = np.array(data_pivot.corr())
cov_matrix = corr_matrix * (vol.reshape(1,-1).T @ vol.reshape(1,-1))

st.header("Investment universe")
st.caption(f"Daily data from {data_pivot.index[0]:%b %Y} to {data_pivot.index[-1]:%b %Y}.")

stats = pd.DataFrame({"Asset": [NAMES[t] for t in ticker],
                      "Expected return": mu,
                      "Volatility": vol}, index=ticker)
st.dataframe(stats.style.format({"Expected return": "{:.1%}", "Volatility": "{:.1%}"}))

# ---------- 3. Markowitz optimization (Lab 1 / Lab 3) ----------
def QP(x, sigma, mu, gamma ):

    v = 0.5 * x.T @ sigma @ x - gamma * x.T @ mu

    return v

x0 = np.ones(mu.shape) / mu.shape

# Fully invested weight constraint and no short-selling
constraints = [sco.LinearConstraint(np.ones(x0.shape), ub = 1), sco.LinearConstraint(-np.ones(x0.shape), ub = -1), sco.LinearConstraint(np.eye(x0.shape[0]), 0)]


def efficient_frontier2(cov_matrix, mu, gam, constraints):

    res = sco.minimize(QP, x0, args = (cov_matrix, mu, gam) , options={'disp': False}, constraints = constraints)
    optimized_weights = res.x
    mu_optimized = optimized_weights @ mu
    vol_optimized = np.sqrt(optimized_weights @ cov_matrix @ optimized_weights)

    return mu_optimized, vol_optimized, optimized_weights

# Markowitz efficient frontier (original mu and cov_matrix)
@st.cache_data
def markowitz(gammas, mu, cov_matrix):
    weights_efficient_frontier = []
    for gam in gammas:
        weights_efficient_frontier.append(efficient_frontier2(cov_matrix, mu, gam, constraints = constraints)[2])
    return pd.DataFrame(weights_efficient_frontier, columns = ticker, index = gammas)

# ---------- 4. Resampling of the efficient frontier (Lab 3) ----------
@st.cache_data
def resampling(gammas, k, mu, cov_matrix):
    np.random.seed(42)                     # same results at every run
    all_weights = []

    for i in range(k):
        simulated_r = np.random.multivariate_normal(mu, cov_matrix, 100)
        simulated_mu = np.mean(simulated_r, axis = 0)
        simulated_cov_mat = np.cov(simulated_r, rowvar = False)

        weights_efficient_frontier = []

        for gam in gammas:
            weights_efficient_frontier.append(efficient_frontier2(simulated_cov_mat, simulated_mu, gam, constraints = constraints)[2])

        to_append = pd.DataFrame(weights_efficient_frontier, columns = ticker)
        to_append['gamma'] = gammas
        all_weights.append(to_append)

    all_weights = pd.concat(all_weights, ignore_index = True)
    avg_weights = all_weights.groupby(by = 'gamma').mean()
    return avg_weights

markowitz_weights = markowitz(GAMMAS, mu, cov_matrix)
with st.spinner("Resampling the efficient frontier (only the first time)..."):
    avg_weights = resampling(GAMMAS, k, mu, cov_matrix)

# expected return and volatility of each portfolio, computed with the original mu and cov_matrix
def frontier_points(weights_df):
    mu_list = [w @ mu for w in weights_df.values]
    vol_list = [np.sqrt(w @ cov_matrix @ w) for w in weights_df.values]
    return mu_list, vol_list

mu_mk, vol_mk = frontier_points(markowitz_weights)
mu_rs, vol_rs = frontier_points(avg_weights)

# ---------- 5. Client portfolio (resampled) ----------
st.header("Optimal portfolio (resampled)")
gam = st.slider("Risk tolerance γ (temporary, will come from the client profile)", 0.0, float(GAMMAS.max()), 0.5, 0.1)
idx = int(np.argmin(np.abs(GAMMAS - gam)))

weights_p = avg_weights.iloc[idx].values
mu_p = weights_p @ mu
vol_p = np.sqrt(weights_p @ cov_matrix @ weights_p)

col1, col2 = st.columns(2)
with col1:
    points = pd.concat([
        pd.DataFrame({"Volatility": vol_mk, "Expected return": mu_mk, "Type": "Markowitz frontier"}),
        pd.DataFrame({"Volatility": vol_rs, "Expected return": mu_rs, "Type": "Resampled frontier"}),
        pd.DataFrame({"Volatility": vol, "Expected return": mu, "Type": "Single ETFs"}),
        pd.DataFrame({"Volatility": [vol_p], "Expected return": [mu_p], "Type": "Client portfolio"}),
    ])
    st.scatter_chart(points, x="Volatility", y="Expected return", color="Type")
with col2:
    compare = pd.DataFrame({"Markowitz": markowitz_weights.iloc[idx].values,
                            "Resampled": weights_p},
                           index=[NAMES[t] for t in ticker])
    st.bar_chart(compare, stack=False)

st.write(f"Client portfolio: expected return **{mu_p:.1%}** | volatility **{vol_p:.1%}**")

# Composition maps (Lab 3)
st.subheader("Composition of optimal portfolios along the efficient frontier")
col1, col2 = st.columns(2)
with col1:
    st.caption("Markowitz: unstable, concentrated in few assets")
    st.area_chart(markowitz_weights.rename(columns=NAMES))
with col2:
    st.caption("Resampled: smoother and more diversified")
    st.area_chart(avg_weights.rename(columns=NAMES))

# ---------- 6. FIRE projection ----------
ret = mu_p - inflation      # real expected return of the client portfolio
target = expenses / swr

wealth = [capital]
for _ in range(fire_age - age):
    wealth.append(wealth[-1] * (1 + ret) + saving * 12)

w, years = capital, 0
while w < target and years < 60:
    w = w * (1 + ret) + saving * 12
    years += 1

st.header("FIRE projection")
col1, col2, col3 = st.columns(3)
col1.metric("FIRE target", f"€ {target:,.0f}")
col2.metric(f"Wealth at {fire_age}", f"€ {wealth[-1]:,.0f}",
            delta=f"€ {wealth[-1] - target:,.0f} vs target")
col3.metric("Age FIRE is reached", f"{age + years}" if w >= target else "Not within 60 years")

df = pd.DataFrame({"Wealth": wealth, "FIRE target": target},
                  index=range(age, fire_age + 1))
df.index.name = "Age"
st.line_chart(df)
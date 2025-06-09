"""
kelly_sims_complete.py
─────────────
Tomás Korenblit, 8 Jun 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
from datetime import datetime
import os

# config de la página, che
st.set_page_config(
    page_title="ATP Tennis Betting Simulator",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# funciones helper
def calculate_kelly_fraction(p: float, odds: float) -> float:
    """fracción de Kelly, entre 0 y 0.99"""
    if p <= 0 or p >= 1 or odds <= 1:
        return 0.0
    b = odds - 1
    q = 1 - p

    def obj(f):
        return -(p * np.log1p(b * f) + q * np.log1p(-f))

    res = minimize_scalar(obj, bounds=(0, 0.99), method="bounded")
    return res.x if res.success else 0.0


def calculate_expected_value(p: float, odds: float) -> float:
    # valor esperado básico
    return p * (odds - 1) - (1 - p)


def simulate_strategy(
    data: pd.DataFrame,
    strategy: dict,
    bankroll: float,
    min_bet: float,
    max_bet_pct: float = None,
    daily_limit: int = None,
):
    """simula cómo va el bankroll con una estrategia"""
    results = {
        "bankroll": [bankroll],
        "log_bankroll": [np.log(bankroll)],
        "bets": [],
        "wins": 0,
        "losses": 0,
        "peak": bankroll,
        "max_drawdown": 0,
        "busted_match": None,
        "daily_bets": 0,
        "current_date": None,
    }

    for _, row in data.iterrows():
        # límite diario de apuestas
        cur_date = row["date"].date()
        if results["current_date"] != cur_date:
            results["current_date"] = cur_date
            results["daily_bets"] = 0
        if daily_limit is not None and results["daily_bets"] >= daily_limit:
            continue

        # elegí a quién apostar
        bet_on_winner = row["p_win"] > 0.5
        p = row["p_win"] if bet_on_winner else 1 - row["p_win"]
        odds = row["b_winner"] if bet_on_winner else row["b_loser"]
        if pd.isna(odds) or odds <= 1:
            continue

        # calculo stake
        if strategy["type"] == "kelly":
            stake = bankroll * calculate_kelly_fraction(p, odds) * strategy.get(
                "fraction", 1.0
            )
        elif strategy["type"] == "flat":
            stake = strategy["amount"]
        elif strategy["type"] == "percentage":
            stake = bankroll * strategy["pct"]
        elif strategy["type"] == "edge":
            edge = p - 1 / odds
            stake = bankroll * max(edge, 0) * strategy["multiplier"]
        else:
            stake = 0

        if max_bet_pct is not None:
            stake = min(stake, bankroll * max_bet_pct)
        stake = max(min_bet, min(stake, bankroll))
        if bankroll < min_bet or stake < min_bet:
            break

        # hago la apuesta
        hit = (row["a_won"] == 1) == bet_on_winner
        if hit:
            bankroll += stake * (odds - 1)
            results["wins"] += 1
        else:
            bankroll -= stake
            results["losses"] += 1

        # registro todo
        results["bankroll"].append(bankroll)
        results["log_bankroll"].append(np.log(max(bankroll, 1e-12)))
        results["bets"].append(
            {
                "date": cur_date,
                "player": row["winner"] if bet_on_winner else row["loser"],
                "stake": stake,
                "odds": odds,
                "outcome": "win" if hit else "loss",
            }
        )
        results["daily_bets"] += 1

        # actualizo peak y drawdown
        if bankroll > results["peak"]:
            results["peak"] = bankroll
        dd = (results["peak"] - bankroll) / results["peak"]
        results["max_drawdown"] = max(results["max_drawdown"], dd)
        if bankroll <= 0 and not results["busted_match"]:
            results["busted_match"] = {
                "date": cur_date,
                "match": f"{row['player_a']} vs {row['player_b']}",
                "stake": stake,
            }
            break

    # métricas finales
    results["final_bankroll"] = bankroll
    results["roi"] = (bankroll - results["bankroll"][0]) / results["bankroll"][0]
    total = results["wins"] + results["losses"]
    results["win_rate"] = results["wins"] / total if total else 0

    yrs = max(
        1e-9, (data["date"].iloc[-1] - data["date"].iloc[0]).days / 365.25
    )
    results["geom_cagr"] = np.exp(
        (results["log_bankroll"][-1] - results["log_bankroll"][0]) / yrs
    ) - 1
    return results


# sidebar parámetros
with st.sidebar:
    st.header("Simulation Parameters")
    initial_bankroll = st.number_input("Starting Bankroll ($)", 100, 100000, 1000, 100)
    min_bet = st.number_input("Minimum Bet ($)", 1, 1000, 10, 1)
    
    # Optional daily bet limit
    use_daily_limit = st.checkbox("Limit daily bets", value=True)
    max_daily_bets = st.slider("Max Bets Per Day", 1, 10, 3, disabled=not use_daily_limit)
    max_daily_bets = max_daily_bets if use_daily_limit else None
    
    # Optional max bet percentage
    use_max_bet_pct = st.checkbox("Limit max bet size", value=True)
    max_bet_pct = st.slider("Max Bet (% of Bankroll)", 1, 20, 5, disabled=not use_max_bet_pct) / 100
    max_bet_pct = max_bet_pct if use_max_bet_pct else None
    
    # Optional minimum win probability
    use_min_prob = st.checkbox("Filter by minimum win probability", value=False)
    min_win_prob = st.slider("Minimum win probability", 0.5, 0.95, 0.55, 0.05, disabled=not use_min_prob)
    min_win_prob = min_win_prob if use_min_prob else 0.0
    
    log_scale = st.checkbox("Plot bankroll on log scale", value=False)

    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded, na_values=["", "NaN"])
        st.success(f"Loaded {len(df_raw):,} rows")
    else:
        df_raw = pd.read_csv("inferencia/predicciones_atp_pwin_reordered.csv", na_values=["", "NaN"])
        st.info("Using bundled sample data")

# ETL nuevo esquema - arreglar con tiempo.
df = df_raw.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
# armamos columnas winner/loser y odds
df["winner"] = np.where(df["a_won"] == 1, df["player_a"], df["player_b"])
df["loser"] = np.where(df["a_won"] == 1, df["player_b"], df["player_a"])
df["b_winner"] = np.where(df["a_won"] == 1, df["b_a"], df["b_b"])
df["b_loser"] = np.where(df["a_won"] == 1, df["b_b"], df["b_a"])
df["p_win"] = np.where(df["a_won"] == 1, df["p_a"], df["p_b"])
df = df.dropna(subset=["date", "p_win"]).sort_values("date").reset_index(drop=True)

# defino strategies
strategies = [ # puedo hacer algo con el p_win, después veo.
    dict(name="Kelly Full", type="kelly", fraction=1.0, filter=lambda r: r["p_win"] >= min_win_prob),
    dict(name="Kelly ½",   type="kelly", fraction=0.5, filter=lambda r: r["p_win"] >= min_win_prob),
    dict(name="Kelly ¼",   type="kelly", fraction=0.25, filter=lambda r: r["p_win"] >= min_win_prob),
    dict(name="Flat $20",  type="flat",  amount=20,     filter=lambda r: r["p_win"] >= min_win_prob),
    dict(name="Fixed 2 %", type="percentage", pct=0.02, filter=lambda r: r["p_win"] >= min_win_prob),
    dict(name="Edge x5",   type="edge", multiplier=5,   filter=lambda r: r["p_win"] >= min_win_prob),
    dict(name="Conservative", type="kelly", fraction=0.10, filter=lambda r: r["p_win"] >= min_win_prob),
]

# interfaz
st.title("🎾 ATP Tennis Betting Strategy Simulator")

if st.button("▶️ Run Simulation", use_container_width=True):
    results = {}
    prog = st.progress(0.0)

    for i, strat in enumerate(strategies):
        subset = df[df.apply(strat["filter"], axis=1)]
        results[strat["name"]] = simulate_strategy(
            subset, strat, initial_bankroll, min_bet, max_bet_pct, max_daily_bets
        )
        prog.progress((i + 1) / len(strategies))

    # bankroll evolution
    st.subheader("Bankroll Evolution")
    fig = go.Figure()
    for name, res in results.items():
        y_vals = res["bankroll"] if not log_scale else res["log_bankroll"]
        fig.add_trace(go.Scatter(x=list(range(len(y_vals))), y=y_vals, mode="lines", name=name))

    y_label = "Bankroll ($)" if not log_scale else "log(Bankroll)"
    fig.update_layout(
        xaxis_title="Bet count",
        yaxis_title=y_label,
        legend=dict(orientation="h", y=1.02, x=1, yanchor="bottom", xanchor="right"),
        height=600,
    )
    if log_scale:
        fig.update_yaxes(type="linear")
    st.plotly_chart(fig, use_container_width=True)

    # performance metrics
    st.subheader("Performance Metrics")
    rows = []
    for name, res in results.items():
        bust = (
            f"{res['busted_match']['date']} – {res['busted_match']['match']}"
            if res["busted_match"]
            else "Never"
        )
        rows.append(
            dict(
                Strategy=name,
                Final=f"${res['final_bankroll']:,.0f}",
                ROI=f"{res['roi']*100:.1f} %",
                CAGR=f"{res['geom_cagr']*100:.1f} %",
                WinRate=f"{res['win_rate']*100:.1f} %",
                MaxDD=f"{res['max_drawdown']*100:.1f} %",
                Bets=len(res["bets"]),
                Bankruptcy=bust,
            )
        )
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

    # bet log
    with st.expander("Bet Log"):
        strat_sel = st.selectbox("Strategy", list(results.keys()))
        log_df = pd.DataFrame(results[strat_sel]["bets"])
        if log_df.empty:
            st.info("No bets placed.")
        else:
            log_df["profit"] = np.where(
                log_df["outcome"] == "win",
                log_df["stake"] * (log_df["odds"] - 1),
                -log_df["stake"],
            )
            st.dataframe(log_df, use_container_width=True)

else:
    st.info("Adjust parameters and press **Run Simulation** to start.")
    st.image(
        "https://www.usatoday.com/gcdn/media/USATODAY/gameon/2012/10/14/10-14-smash-djoker-16_9.jpg?width=2902&height=1642&fit=crop&format=pjpg&auto=webp"
    )

    

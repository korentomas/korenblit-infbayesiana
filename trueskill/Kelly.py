import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
from collections import defaultdict
import matplotlib.pyplot as plt
from TrueSkillThroughTime import History, Gaussian  # Your teacher's package

# Configuration
MU = 0.0
SIGMA = 2.5
BETA = 1.0
GAMMA = 0.03
INITIAL_BANKROLL = 1000
FRACTIONAL_KELLY = 0.5  # Risk management

def load_data(file_path):
    """Load and preprocess tennis match data"""
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.dropna(subset=['Date', 'Winner', 'Loser', 'B365W', 'B365L'])
    
    # Convert dates to numeric (days since 1900-01-01)
    ref_date = datetime(1900, 1, 1)
    df['days'] = (df['Date'] - ref_date).dt.days
    
    return df

def kelly_fraction(p, odds):
    """Calculate fractional Kelly bet size"""
    if odds < 1.01:  # Sanity check
        return 0.0
    b = odds - 1.0
    edge = p * (b + 1) - 1
    return max(0.0, FRACTIONAL_KELLY * edge / b)

def predict_win_prob(mu1, sigma1, mu2, sigma2, beta=BETA):
    """Predict win probability for player1"""
    mu_diff = mu1 - mu2
    sigma_diff = np.sqrt(sigma1**2 + sigma2**2 + 2 * beta**2)
    return norm.cdf(mu_diff / sigma_diff)

def simulate_betting(df, skills_history):
    """Run Kelly betting simulation"""
    bankroll = INITIAL_BANKROLL
    bankroll_history = []
    skills = defaultdict(lambda: Gaussian(MU, SIGMA))
    
    for idx, row in df.iterrows():
        # Get current skills (from previous matches)
        winner_skill = skills[row['Winner']]
        loser_skill = skills[row['Loser']]
        
        # Predict win probability
        p_win = predict_win_prob(
            winner_skill.mu,
            winner_skill.sigma,
            loser_skill.mu,
            loser_skill.sigma
        )
        
        # Calculate Kelly fraction
        odds = row['B365W']
        f = kelly_fraction(p_win, odds)
        
        # Place bet (assuming we always bet on the winner)
        stake = bankroll * f
        bankroll -= stake  # Place bet
        bankroll += stake * odds  # Winner wins - we get stake * odds
        
        # Update bankroll history
        bankroll_history.append(bankroll)
        
        # Update skills from history
        winner_skill = skills_history[row['Winner']][row['days']]
        loser_skill = skills_history[row['Loser']][row['days']]
        skills[row['Winner']] = winner_skill
        skills[row['Loser']] = loser_skill
        
        # Exit if bankrupt
        if bankroll < 1:
            print(f"Bankrupt at match {idx}")
            break
    
    return bankroll_history

def main():
    # Load data
    df = load_data("data/df_atp.csv")
    
    # Prepare composition and times
    composition = []
    times = []
    for _, row in df.iterrows():
        composition.append([[row['Winner']], [row['Loser']]])
        times.append(row['days'])
    
    # Model skills over time
    history = History(
        composition=composition,
        times=times,
        mu=MU,
        sigma=SIGMA,
        beta=BETA,
        gamma=GAMMA,
        online=True
    )
    history.convergence(iterations=1, verbose=True)
    
    # Get learning curves
    lc = history.learning_curves()
    
    # Reformat skills history
    skills_history = defaultdict(dict)
    for player, curve in lc.items():
        for time, skill in curve:
            skills_history[player][time] = skill
    
    # Run betting simulation
    bankroll_history = simulate_betting(df, skills_history)
    
    # Save predictions and results
    results = []
    for idx, row in df.iterrows():
        winner_skill = skills_history[row['Winner']][row['days']]
        loser_skill = skills_history[row['Loser']][row['days']]
        
        results.append({
            'date': row['Date'],
            'winner': row['Winner'],
            'loser': row['Loser'],
            'm_winner': winner_skill.mu,
            's_winner': winner_skill.sigma,
            'm_loser': loser_skill.mu,
            's_loser': loser_skill.sigma,
            'b_winner': row['B365W'],
            'b_loser': row['B365L'],
            'bankroll': bankroll_history[idx] if idx < len(bankroll_history) else None
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("predictions_with_bankroll.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['date'], results_df['bankroll'])
    plt.title("Bankroll Growth Over Time")
    plt.xlabel("Date")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.savefig("bankroll_growth.png")
    plt.show()

if __name__ == "__main__":
    main()
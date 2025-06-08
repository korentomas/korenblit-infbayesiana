# %% [markdown]
# # Tennis Match Prediction with TrueSkill and Kelly Criterion
# ## Analyzing Tennis Matches with TrueSkill Through Time and Kelly Betting

# %%
import os
import sys
import io
import itertools
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm
import math
from TrueskillThroughTime import *

class TennisDataPreprocessor:
    """Handles tennis data preprocessing and preparation."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.team_dict = None
    
    def prepare_data(self) -> pd.DataFrame:
        """Prepare tennis data for analysis."""
        # Add required columns
        self.df['home'] = self.df['Winner']
        self.df['away'] = self.df['Loser']
        self.df['result'] = 'hwin'  # Winner always wins in tennis
        self.df['date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Remove rows with missing dates
        self.df = self.df.dropna(subset=['date'])
        
        # Create player dictionary
        all_players = pd.unique(self.df[['Winner', 'Loser']].values.ravel('K'))
        self.team_dict = {name: idx for idx, name in enumerate(all_players)}
        
        return self.df
    
    def prepare_ttt_data(self) -> Tuple[List, List, List, List]:
        """Prepare data for TTT model."""
        composition = []
        results = []
        times = []
        weights = []
        
        for _, row in self.df.iterrows():
            try:
                match_composition = [[row['home']], [row['away']]]
                composition.append(match_composition)
                
                match_result = [1, 0]  # Home player (Winner) always wins
                results.append(match_result)
                
                match_date = row['date'].to_pydatetime()
                times.append(match_date.toordinal())
                
                match_weights = [[1.0], [1.0]]
                weights.append(match_weights)
            except (AttributeError, ValueError) as e:
                print(f"Skipping row due to error: {e}")
                continue
        
        if not times:
            raise ValueError("No valid matches found in the dataset")
        
        print(f"Processed {len(times)} valid matches")
        return composition, results, times, weights

class KellyBettingAnalyzer:
    """Handles Kelly Criterion betting analysis."""
    
    @staticmethod
    def calculate_kelly_fraction(p: float, odds: float) -> float:
        """Calculate optimal Kelly fraction."""
        try:
            p = float(p)
            odds = float(odds)
            
            if p <= 0 or odds <= 1:
                return 0.0
            
            q = 1 - p
            b = odds - 1
            f = (p * b - q) / b
            
            # Limit Kelly fraction to reasonable values
            return max(0, min(0.25, f))  # Cap at 25% of bankroll
        except (ValueError, TypeError):
            return 0.0
    
    def analyze_betting_opportunities(self, df: pd.DataFrame, final_ratings: Dict[str, float], 
                                   beta: float = 0.5) -> pd.DataFrame:
        """Analyze betting opportunities using TrueSkill ratings and odds."""
        results = []
        
        # Convert odds columns to numeric, replacing invalid values with NaN
        df['B365W'] = pd.to_numeric(df['B365W'], errors='coerce')
        df['B365L'] = pd.to_numeric(df['B365L'], errors='coerce')
        
        # Only analyze matches with valid odds
        df_valid = df.dropna(subset=['B365W', 'B365L'])
        
        print(f"Analyzing {len(df_valid)} matches with valid odds out of {len(df)} total matches")
        
        opportunities_found = 0
        total_ev = 0
        
        for _, match in df_valid.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            date = match['date']
            
            # Get TrueSkill ratings and calculate probability
            winner_rating = final_ratings.get(winner, 0)
            loser_rating = final_ratings.get(loser, 0)
            delta = winner_rating - loser_rating
            win_prob = norm.cdf(delta / math.sqrt(2 * beta**2))
            
            # Get odds and calculate Kelly fractions
            winner_odds = float(match['B365W'])
            loser_odds = float(match['B365L'])
            
            # Skip matches with invalid odds
            if winner_odds <= 1 or loser_odds <= 1:
                continue
                
            # Calculate expected values
            winner_ev = win_prob * (winner_odds - 1) - (1 - win_prob)
            loser_ev = (1 - win_prob) * (loser_odds - 1) - win_prob
            
            kelly_winner = self.calculate_kelly_fraction(win_prob, winner_odds)
            kelly_loser = self.calculate_kelly_fraction(1 - win_prob, loser_odds)
            
            # Only bet if there's positive expected value
            if max(winner_ev, loser_ev) > 0:
                opportunities_found += 1
                total_ev += max(winner_ev, loser_ev)
            
            results.append({
                'date': date,
                'winner': winner,
                'loser': loser,
                'winner_rating': winner_rating,
                'loser_rating': loser_rating,
                'predicted_win_prob': win_prob,
                'actual_winner': winner,
                'winner_odds': winner_odds,
                'loser_odds': loser_odds,
                'winner_ev': winner_ev,
                'loser_ev': loser_ev,
                'kelly_winner': kelly_winner,
                'kelly_loser': kelly_loser,
                'bet_on_winner': kelly_winner > kelly_loser,
                'optimal_kelly': max(kelly_winner, kelly_loser)
            })
        
        if not results:
            raise ValueError("No valid betting opportunities found in the dataset")
            
        print(f"\nBetting opportunities analysis:")
        print(f"Found {opportunities_found} positive EV opportunities")
        print(f"Average EV per opportunity: {total_ev/len(results) if results else 0:.4f}")
        
        return pd.DataFrame(results)
    
    def simulate_betting_strategy(self, betting_analysis: pd.DataFrame, 
                                initial_bankroll: float = 1000, 
                                kelly_fraction: float = 0.5) -> pd.DataFrame:
        """Simulate betting results using Kelly Criterion."""
        bankroll = initial_bankroll
        results = []
        
        # Track betting statistics
        total_bets = 0
        winning_bets = 0
        total_profit = 0
        max_bankroll = initial_bankroll
        
        for _, bet in betting_analysis.iterrows():
            # Only bet if we have positive expected value
            if bet['optimal_kelly'] <= 0 or (bet['winner_ev'] <= 0 and bet['loser_ev'] <= 0):
                continue
            
            kelly_bet = bet['optimal_kelly'] * kelly_fraction
            bet_size = bankroll * kelly_bet
            
            # Ensure minimum bet size
            if bet_size < 1.0:  # Skip tiny bets
                continue
            
            if bet['bet_on_winner']:
                profit = bet_size * (bet['winner_odds'] - 1) if bet['winner'] == bet['actual_winner'] else -bet_size
            else:
                profit = bet_size * (bet['loser_odds'] - 1) if bet['winner'] != bet['actual_winner'] else -bet_size
            
            total_bets += 1
            if profit > 0:
                winning_bets += 1
            total_profit += profit
            bankroll += profit
            max_bankroll = max(max_bankroll, bankroll)
            
            results.append({
                'date': bet['date'],
                'bankroll': bankroll,
                'bet_size': bet_size,
                'profit': profit,
                'bet_on': bet['winner'] if bet['bet_on_winner'] else bet['loser'],
                'odds_taken': bet['winner_odds'] if bet['bet_on_winner'] else bet['loser_odds'],
                'predicted_prob': bet['predicted_win_prob'] if bet['bet_on_winner'] else 1 - bet['predicted_win_prob'],
                'ev': bet['winner_ev'] if bet['bet_on_winner'] else bet['loser_ev']
            })
        
        # Print betting statistics
        print(f"\nBetting simulation results:")
        print(f"Total bets placed: {total_bets}")
        print(f"Winning bets: {winning_bets} ({winning_bets/total_bets*100:.1f}% win rate)")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Return on investment: {(total_profit/initial_bankroll)*100:.1f}%")
        print(f"Maximum bankroll: ${max_bankroll:.2f}")
        print(f"Final bankroll: ${bankroll:.2f}")
        
        return pd.DataFrame(results)

class PerformanceAnalyzer:
    """Handles performance analysis and visualization."""
    
    @staticmethod
    def analyze_metrics(betting_results: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        betting_results['daily_return'] = betting_results['bankroll'].pct_change()
        monthly_returns = betting_results.set_index('date').resample('M')['bankroll'].last().pct_change()
        
        total_return = (betting_results['bankroll'].iloc[-1] / betting_results['bankroll'].iloc[0]) - 1
        avg_daily_return = betting_results['daily_return'].mean()
        daily_volatility = betting_results['daily_return'].std()
        sharpe_ratio = np.sqrt(252) * (avg_daily_return / daily_volatility) if daily_volatility != 0 else 0
        
        rolling_max = betting_results['bankroll'].cummax()
        drawdown = (betting_results['bankroll'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        wins = (betting_results['profit'] > 0).sum()
        total_bets = len(betting_results)
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365 / len(betting_results)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_bets': total_bets,
            'avg_bet_size': betting_results['bet_size'].mean(),
            'avg_profit': betting_results['profit'].mean(),
            'monthly_returns': monthly_returns
        }
    
    @staticmethod
    def plot_detailed_results(betting_results: pd.DataFrame) -> go.Figure:
        """Create detailed performance plot."""
        fig = go.Figure()
        
        # Cumulative Returns
        fig.add_trace(go.Scatter(
            x=betting_results['date'],
            y=betting_results['bankroll'],
            mode='lines',
            name='Cumulative Returns',
            line=dict(width=2, color='blue'),
            hovertemplate="<b>Date</b>: %{x}<br><b>Bankroll</b>: $%{y:.2f}<br><extra></extra>"
        ))
        
        # Trend line
        x_numeric = (betting_results['date'] - betting_results['date'].min()).dt.days
        z = np.polyfit(x_numeric, betting_results['bankroll'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=betting_results['date'],
            y=p(x_numeric),
            mode='lines',
            name='Trend',
            line=dict(dash='dash', color='red'),
            hovertemplate="<b>Trend Value</b>: $%{y:.2f}<br><extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': 'Betting Performance Over Time',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title="Date",
            yaxis_title="Bankroll ($)",
            template="plotly_white",
            hovermode='x unified',
            width=1200,
            height=800
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns(monthly_returns: pd.Series) -> go.Figure:
        """Create monthly returns plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_returns.index,
            y=monthly_returns * 100,
            name='Monthly Returns',
            marker_color=np.where(monthly_returns >= 0, 'green', 'red'),
            hovertemplate="<b>Month</b>: %{x}<br><b>Return</b>: %{y:.2f}%<br><extra></extra>"
        ))
        
        fig.update_layout(
            title='Monthly Returns',
            xaxis_title='Month',
            yaxis_title='Return (%)',
            template='plotly_white',
            showlegend=False,
            width=1200,
            height=400
        )
        
        return fig

class TrueSkillAnalyzer:
    """Handles TrueSkill analysis and visualization."""
    
    @staticmethod
    def analyze_ttt_results(history, team_dict):
        """Analyze and visualize TTT results."""
        # Get learning curves
        learning_curves = history.learning_curves()
        
        # Create a reverse mapping from team ID to name
        reverse_team_dict = {v: k for k, v in team_dict.items()}
        
        # Extract the final ratings for each player
        final_ratings = {}
        for team_id, curve in learning_curves.items():
            if curve:  # Check if the player has any rating data
                # Safely get player name, use team_id as fallback if not found
                player_name = reverse_team_dict.get(team_id, f"Player_{team_id}")
                final_ratings[player_name] = curve[-1][1].mu
        
        # Sort players by their final rating
        sorted_players = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)
        
        return learning_curves, final_ratings, sorted_players
    
    @staticmethod
    def print_player_rankings(sorted_players, top_n=None):
        """Print player rankings with their ratings."""
        players_to_show = sorted_players[:top_n] if top_n else sorted_players
        
        print("Player Rankings:")
        print("-" * 40)
        print(f"{'Rank':<5}{'Player':<30}{'Rating':<10}")
        print("-" * 40)
        for i, (player, rating) in enumerate(players_to_show, 1):
            print(f"{i:<5}{player:<30}{rating:<10.3f}")
    
    @staticmethod
    def predict_match_outcome(player1, player2, final_ratings, beta=0.5):
        """Predict the outcome of a match between two players."""
        player1_rating = final_ratings.get(player1, 0)
        player2_rating = final_ratings.get(player2, 0)
        
        # Calculate win probability using TrueSkill formula
        delta = player1_rating - player2_rating
        
        player1_win_prob = norm.cdf(delta / math.sqrt(2 * beta**2))
        player2_win_prob = 1 - player1_win_prob  # No draws in tennis
        
        return {
            f"{player1} Win": player1_win_prob,
            f"{player2} Win": player2_win_prob
        }
    
    @staticmethod
    def plot_player_ratings(learning_curves, team_dict, top_n=20):
        """Plot player ratings over time."""
        reverse_team_dict = {v: k for k, v in team_dict.items()}
        
        # Calculate player performance metrics
        player_performances = []
        for team_id, curve in learning_curves.items():
            if curve:
                # Get player name from ID
                player_name = reverse_team_dict.get(team_id, f"Player_{team_id}")
                final_rating = curve[-1][1].mu
                uncertainty = curve[-1][1].sigma
                player_performances.append((player_name, team_id, final_rating, uncertainty))
        
        # Sort by final rating and select top players
        player_performances.sort(key=lambda x: x[2], reverse=True)
        top_players = player_performances[:top_n]
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot learning curves for top players
        for player_name, team_id, _, _ in top_players:
            curve = learning_curves[team_id]
            times = [datetime.fromordinal(t[0]) for t in curve]
            ratings = [t[1].mu for t in curve]
            plt.plot(times, ratings, label=player_name, linewidth=2)
        
        plt.title('Player Skill Ratings Over Time', fontsize=18, pad=20)
        plt.xlabel('Date', fontsize=14, labelpad=10)
        plt.ylabel('Skill Rating', fontsize=14, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  borderaxespad=0., fontsize=10, framealpha=0.8)
        plt.tight_layout()
        
        return plt
    
    @staticmethod
    def plot_player_ratings_interactive(learning_curves, team_dict, top_n=20):
        """Create an interactive Plotly visualization of player ratings over time."""
        reverse_team_dict = {v: k for k, v in team_dict.items()}
        
        # Calculate player performance metrics
        player_performances = []
        for team_id, curve in learning_curves.items():
            if curve:
                player_name = reverse_team_dict.get(team_id, f"Player_{team_id}")
                final_rating = curve[-1][1].mu
                uncertainty = curve[-1][1].sigma
                player_performances.append((player_name, team_id, final_rating, uncertainty))
        
        # Sort and select top players
        player_performances.sort(key=lambda x: x[2], reverse=True)
        top_players = player_performances[:top_n]
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each player
        for player_name, team_id, _, _ in top_players:
            curve = learning_curves[team_id]
            times = [datetime.fromordinal(t[0]) for t in curve]
            ratings = [t[1].mu for t in curve]
            
            fig.add_trace(go.Scatter(
                x=times,
                y=ratings,
                name=player_name,
                mode='lines',
                hovertemplate=(
                    f"<b>{player_name}</b><br>" +
                    "Date: %{x}<br>" +
                    "Rating: %{y:.2f}<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Player Skill Ratings Over Time',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title="Date",
            yaxis_title="Skill Rating",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            template="plotly_white",
            width=1200,
            height=800,
            margin=dict(r=150)
        )
        
        fig.update_xaxes(tickangle=45, gridcolor='lightgrey', showgrid=True)
        fig.update_yaxes(gridcolor='lightgrey', showgrid=True)
        
        return fig

def main():
    """Main execution function."""
    # Load data
    print("Loading data...")
    df_wta = pd.read_csv('df_wta.csv')
    print(f"Loaded WTA dataset with shape: {df_wta.shape}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = TennisDataPreprocessor(df_wta)
    df_processed = preprocessor.prepare_data()
    composition, results, times, weights = preprocessor.prepare_ttt_data()
    
    # Run TrueSkill Through Time
    print("\nRunning TrueSkill Through Time analysis...")
    history = History(
        composition=composition,
        results=results,
        times=times,
        weights=weights,
        mu=0.0,
        sigma=8.0,
        beta=0.5,
        gamma=0.02,
        p_draw=0.0,
        online=False
    )
    
    history.convergence(iterations=10, epsilon=0.001, verbose=True)
    
    # Analyze TrueSkill results
    trueskill_analyzer = TrueSkillAnalyzer()
    learning_curves, final_ratings, sorted_players = trueskill_analyzer.analyze_ttt_results(
        history, preprocessor.team_dict)
    
    # Print top players and save ratings plots
    print("\nTop 20 Players by Rating:")
    trueskill_analyzer.print_player_rankings(sorted_players, top_n=20)
    
    print("\nGenerating player rating plots...")
    plt.ioff()
    ratings_plot = trueskill_analyzer.plot_player_ratings(
        learning_curves, preprocessor.team_dict, top_n=20)
    ratings_plot.savefig('tennis_player_ratings.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    ratings_plot_interactive = trueskill_analyzer.plot_player_ratings_interactive(
        learning_curves, preprocessor.team_dict, top_n=20)
    ratings_plot_interactive.write_html('tennis_player_ratings.html')
    
    # Analyze betting opportunities
    print("\nAnalyzing betting opportunities...")
    kelly_analyzer = KellyBettingAnalyzer()
    betting_analysis = kelly_analyzer.analyze_betting_opportunities(df_processed, final_ratings)
    betting_results = kelly_analyzer.simulate_betting_strategy(betting_analysis)
    
    # Analyze performance
    print("\nAnalyzing performance...")
    performance_analyzer = PerformanceAnalyzer()
    metrics = performance_analyzer.analyze_metrics(betting_results)
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Total Bets: {metrics['total_bets']}")
    print(f"Average Bet Size: ${metrics['avg_bet_size']:.2f}")
    print(f"Average Profit per Bet: ${metrics['avg_profit']:.2f}")
    
    # Generate and save visualizations
    print("\nGenerating performance visualizations...")
    detailed_plot = performance_analyzer.plot_detailed_results(betting_results)
    monthly_plot = performance_analyzer.plot_monthly_returns(metrics['monthly_returns'])
    
    detailed_plot.write_html('tennis_detailed_performance.html')
    monthly_plot.write_html('tennis_monthly_returns.html')
    betting_results.to_csv('tennis_betting_results.csv', index=False)
    
    print("\nAnalysis complete! Generated files:")
    print("- tennis_player_ratings.png (Static player ratings plot)")
    print("- tennis_player_ratings.html (Interactive player ratings)")
    print("- tennis_detailed_performance.html (Cumulative returns and trend)")
    print("- tennis_monthly_returns.html (Monthly return distribution)")
    print("- tennis_betting_results.csv (Detailed betting history)")

if __name__ == "__main__":
    main()
# %%

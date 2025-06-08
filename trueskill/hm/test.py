import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import io
from datetime import datetime
from collections import defaultdict
import math
from scipy.stats import norm
from TrueskillThroughTime import *  # Import the TTT module from the shared code
from helpers import *
import pickle
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def format_season_df(df_orig):
    df = df_orig.copy()
    # Assign integer indices to alpha sorted team names
    teams_dict = {team_name: team_num for team_name, team_num in zip(sorted(df['Home'].unique().tolist()), range(df['Home'].nunique()))}
    for col in ['Home', 'Away']:
        df[col.lower()[0] + 'g'] = df[col].map(teams_dict)
    # Rename column names for consistency with original paper
    df['yg1'] = df['h_ftgoals'] ; df['yg2'] = df['a_ftgoals']
    # Sort the values to get an approximate match to the sorting order used in the paper
    df = df.sort_values(by=['Date', 'Home'], ascending=[True, False]).reset_index(drop=True)
    df['g'] = df.index
    # lowercase col names and select columns to go forward
    df.columns = [col.lower() for col in df.columns]
    df = df[['g', 'home', 'away', 'hg', 'ag', 'yg1', 'yg2', 'date']]
    # Put match outcome into df for later analysis
    condlist = [df['yg1']>df['yg2'], df['yg1']==df['yg2'], df['yg2']>df['yg1']]
    choicelist = ['hwin', 'draw', 'awin']
    df['result'] = np.select(condlist, choicelist)
    return df, teams_dict

def scrape_fbref_data(url):
    """
    Scrape tables from a FBref URL and return their IDs.
    
    Args:
        url (str): URL of the FBref page to scrape
        
    Returns:
        list: List of table IDs found on the page
    """
    # Send a GET request to the URL
    response = requests.get(url)

    table_ids = []
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all tables in page
        tables = soup.find_all("table")
        
        # Get IDs for each table
        for table in tables:
            table_id = table.get("id")
            if table_id:
                table_ids.append(table_id)
                
    return table_ids, soup

seasons = ["https://fbref.com/en/comps/21/2015/schedule/2015-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2016/schedule/2016-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2016-2017/schedule/2016-2017-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2017-2018/schedule/2017-2018-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2018-2019/schedule/2018-2019-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2019-2020/schedule/2019-2020-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2021/schedule/2021-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2022/schedule/2022-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2023/schedule/2023-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/2024/schedule/2024-Liga-Profesional-Argentina-Scores-and-Fixtures",
           "https://fbref.com/en/comps/21/schedule/Liga-Profesional-Argentina-Scores-and-Fixtures"]

data = []

for season in seasons:
    ids, soup = scrape_fbref_data(season)
    print(ids)
    
    table = soup.select("#" +ids[0]) # using '#' to indicate id we want
    if not table:
        continue
        
    table = table[0] # soup creates lists, so get first element
    df = pd.read_html(io.StringIO(str(table)))[0]
    df.dropna(subset=['Score'], inplace=True)
    try:
        df = df.loc[df['Round'] == 'Torneo Apertura — Regular season']
    except:
        print(f"No round found for {season}")
        pass
    
    # Extract goals, handling potential errors
    try:
        df['h_ftgoals'] = df['Score'].str.split('–').str[0].astype(int)
        df['a_ftgoals'] = df['Score'].str.split('–').str[1].astype(int)
    except:
        print(f"Error processing scores for season {season}")
        continue
        
    df = df[['Wk', 'Day', 'Date', 'Home', 'Away', 'h_ftgoals', 'a_ftgoals']]
    df, team_dict = format_season_df(df)
    print(team_dict)
    data.append(df)
        
# Combine all seasons
df = pd.concat(data, ignore_index=True)
# Process each row to create composition, results, and times
composition = []
results = []
times = []
for idx, row in df.iterrows():
    home_team = [row['home']]
    away_team = [row['away']]
    composition.append([home_team, away_team])
    
    # Determine result based on match outcome
    if row['result'] == 'hwin':
        game_result = [1, 0]
    elif row['result'] == 'awin':
        game_result = [0, 1]
    else:
        game_result = [0, 0]
    results.append(game_result)
    times.append(row['date'])  # Using index as time

# Calculate draw probability
p_draw = len(df[df['result'] == 'draw']) / len(df)
mu = 0.0
sigma = 6.0  # Initial skill uncertainty
beta = 1.0   # Performance variance
gamma = 0.03 # Skill dynamics (how much skills can change over time)

history = History(
    composition=composition,
    results=results,
    times=times,
    mu=mu,
    sigma=sigma,
    beta=beta,
    gamma=gamma,
    p_draw=p_draw,
    online=False
)
# TTT preparation functions
def prepare_ttt_data(df):
    """Prepare data for TTT model."""
    composition = []
    results = []
    times = []
    weights = []
    
    for _, row in df.iterrows():
        # Each team is in its own list (1-person team)
        match_composition = [[row['home']], [row['away']]]
        composition.append(match_composition)
        
        # Results are ordered by who won (higher = better)
        if row['result'] == 'hwin':
            match_result = [1, 0]  # Home team wins
        elif row['result'] == 'awin':
            match_result = [0, 1]  # Away team wins
        else:
            match_result = [0.5, 0.5]  # Draw
        results.append(match_result)
        
        # Convert date to timestamp
        if isinstance(row['date'], str):
            match_date = datetime.strptime(row['date'], '%Y-%m-%d')
        else:
            match_date = row['date']
        times.append(match_date.toordinal())
        
        # Equal weights for all players
        match_weights = [[1.0], [1.0]]
        weights.append(match_weights)
    
    return composition, results, times, weights

# Analysis functions
def analyze_ttt_results(history, team_dict):
    """Analyze and visualize TTT results."""
    # Get learning curves
    learning_curves = history.learning_curves()
    
    # Create a reverse mapping from team ID to name
    reverse_team_dict = {v: k for k, v in team_dict.items()}
    
    # Extract the final ratings for each team
    final_ratings = {}
    for team_id, curve in learning_curves.items():
        if curve:  # Check if the team has any rating data
            # Safely get team name, use team_id as fallback if not found
            team_name = reverse_team_dict.get(team_id, f"Team_{team_id}")
            final_ratings[team_name] = curve[-1][1].mu
    
    # Sort teams by their final rating
    sorted_teams = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)
    
    return learning_curves, final_ratings, sorted_teams

# Visualization functions
def plot_team_ratings(learning_curves, team_dict, top_n=30):
    """Plot team ratings over time."""
    reverse_team_dict = {v: k for k, v in team_dict.items()}
    
    # Calculate team performance metrics
    team_performances = []
    for team_id, curve in learning_curves.items():
        if curve:
            # Get team name from ID
            team_name = reverse_team_dict.get(team_id, f"Team_{team_id}")
            final_rating = curve[-1][1].mu
            uncertainty = curve[-1][1].sigma
            team_performances.append((team_name, team_id, final_rating, uncertainty))
    
    # Sort by final rating
    team_performances.sort(key=lambda x: x[2], reverse=True)
    
    # Select top N teams
    top_teams = team_performances[:top_n]
    
    # Create figure with larger size and better aspect ratio
    plt.figure(figsize=(15, 10))
    
    # Plot learning curves for top teams
    for team_name, team_id, _, _ in top_teams:
        curve = learning_curves[team_id]
        
        # Extract times and ratings
        times = [datetime.fromordinal(t[0]) for t in curve]
        ratings = [t[1].mu for t in curve]
        
        plt.plot(times, ratings, label=team_name, linewidth=2)
    
    plt.title('Team Skill Ratings Over Time', fontsize=18, pad=20)
    plt.xlabel('Date', fontsize=14, labelpad=10)
    plt.ylabel('Skill Rating', fontsize=14, labelpad=10)
    
    # Improve grid appearance
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust legend placement and appearance
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., fontsize=10, framealpha=0.8)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return plt

def print_team_rankings(sorted_teams):
    """Print team rankings with their ratings."""
    print("Team Rankings:")
    print("-" * 40)
    print(f"{'Rank':<5}{'Team':<30}{'Rating':<10}")
    print("-" * 40)
    for i, (team, rating) in enumerate(sorted_teams, 1):
        print(f"{i:<5}{team:<30}{rating:<10.3f}")

def predict_match_outcome(team1, team2, final_ratings, beta=1.0):
    """Predict the outcome of a match between two teams."""
    team1_rating = final_ratings.get(team1, 0)
    team2_rating = final_ratings.get(team2, 0)
    
    # Calculate win probability using the TrueSkill formula
    delta = team1_rating - team2_rating
    draw_margin = 0.0  # You can adjust this for draw probability
    
    team1_win_prob = norm.cdf(delta / math.sqrt(2 * beta**2))
    team2_win_prob = norm.cdf(-delta / math.sqrt(2 * beta**2))
    draw_prob = 1 - team1_win_prob - team2_win_prob
    
    return {
        f"{team1} Win": team1_win_prob,
        "Draw": draw_prob,
        f"{team2} Win": team2_win_prob
    }

def plot_team_ratings_plotly(learning_curves, team_dict, top_n=30):
    """Create an interactive Plotly visualization of team ratings over time."""
    reverse_team_dict = {v: k for k, v in team_dict.items()}
    
    # Calculate team performance metrics
    team_performances = []
    for team_id, curve in learning_curves.items():
        if curve:
            team_name = reverse_team_dict.get(team_id, f"Team_{team_id}")
            final_rating = curve[-1][1].mu
            uncertainty = curve[-1][1].sigma
            team_performances.append((team_name, team_id, final_rating, uncertainty))
    
    # Sort by final rating
    team_performances.sort(key=lambda x: x[2], reverse=True)
    
    # Select top N teams
    top_teams = team_performances[:top_n]
    
    # Create the figure
    fig = go.Figure()
    
    # Add traces for each team
    for team_name, team_id, _, _ in top_teams:
        curve = learning_curves[team_id]
        times = [datetime.fromordinal(t[0]) for t in curve]
        ratings = [t[1].mu for t in curve]
        
        fig.add_trace(go.Scatter(
            x=times,
            y=ratings,
            name=team_name,
            mode='lines',
            hovertemplate=(
                f"<b>{team_name}</b><br>" +
                "Date: %{x}<br>" +
                "Rating: %{y:.2f}<br>" +
                "<extra></extra>"
            )
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Team Skill Ratings Over Time',
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
        margin=dict(r=150)  # Add margin for legend
    )
    
    # Update axes
    fig.update_xaxes(
        tickangle=45,
        gridcolor='lightgrey',
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor='lightgrey',
        showgrid=True
    )
    
    return fig

# Prepare data for TTT
composition, results, times, weights = prepare_ttt_data(df)

# Create TTT history
history = History(
    composition=composition,
    results=results,
    times=times,
    weights=weights,
    mu=0.0,        # Initial mu
    sigma=6.0,     # Initial sigma
    beta=1.0,      # Performance SD
    gamma=0.03,    # Skill drift SD
    p_draw=0.1,    # Draw probability
    online=False   # Process all at once, not incrementally
)

# Run the algorithm
history.convergence(iterations=10, epsilon=0.001, verbose=True)

# Analyze results
learning_curves, final_ratings, sorted_teams = analyze_ttt_results(history, team_dict)

# Print team rankings
print_team_rankings(sorted_teams)

# Plot ratings over time
plt.ioff()  # Turn off interactive mode
plot = plot_team_ratings(learning_curves, team_dict)
plt.savefig('team_ratings_2025.png')
plt.close()

# Example usage (add this where you want to create the interactive plot):
fig = plot_team_ratings_plotly(learning_curves, team_dict)
fig.show()  # This will open in your default browser
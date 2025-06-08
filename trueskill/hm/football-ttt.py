import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import io
from datetime import datetime
import matplotlib.pyplot as plt
from TrueskillThroughTime import *

# --- 1. Data Scraping & Loading ---
def scrape_fbref_schedule(url):
    """Scrape match schedule table from FBref."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.select_one('table')
    df = pd.read_html(io.StringIO(str(table)))[0]
    return df

def load_data(filepath=None, url=None):
    if filepath:
        df = pd.read_csv(filepath)
    elif url:
        df = scrape_fbref_schedule(url)
    else:
        raise ValueError('Provide either filepath or url')
    return df

def load_multiple_seasons(urls):
    """Load and combine multiple seasons from a list of FBref URLs."""
    data = []
    for url in urls:
        print(f"Loading {url}")
        df_raw = scrape_fbref_schedule(url)
        if 'Score' in df_raw.columns:
            df_raw = df_raw.dropna(subset=['Score'])
            df_raw['h_ftgoals'] = df_raw['Score'].str.split('–').str[0].astype(int)
            df_raw['a_ftgoals'] = df_raw['Score'].str.split('–').str[1].astype(int)
        data.append(df_raw)
    df_all = pd.concat(data, ignore_index=True)
    return df_all

# --- 2. Data Formatting ---
def format_season_df(df_orig):
    df = df_orig.copy()
    teams_dict = {team_name: team_num for team_num, team_name in enumerate(sorted(df['Home'].unique()))}
    for col in ['Home', 'Away']:
        df[col.lower()[0] + 'g'] = df[col].map(teams_dict)
    df['yg1'] = df['h_ftgoals'] ; df['yg2'] = df['a_ftgoals']
    df = df.sort_values(by=['Date', 'Home'], ascending=[True, False]).reset_index(drop=True)
    df['g'] = df.index
    df.columns = [col.lower() for col in df.columns]
    df = df[['g', 'home', 'away', 'hg', 'ag', 'yg1', 'yg2', 'date']]
    condlist = [df['yg1']>df['yg2'], df['yg1']==df['yg2'], df['yg2']>df['yg1']]
    choicelist = ['hwin', 'draw', 'awin']
    df['result'] = np.select(condlist, choicelist, default='other')
    return df, teams_dict

# --- 3. TTT Preparation ---
def prepare_ttt_data(df):
    composition, results, times, weights = [], [], [], []
    for _, row in df.iterrows():
        match_composition = [[row['home']], [row['away']]]
        composition.append(match_composition)
        if row['result'] == 'hwin':
            match_result = [1, 0]
        elif row['result'] == 'awin':
            match_result = [0, 1]
        else:
            match_result = [0.5, 0.5]
        results.append(match_result)
        if isinstance(row['date'], str):
            match_date = datetime.strptime(row['date'], '%Y-%m-%d')
        else:
            match_date = row['date']
        times.append(match_date.toordinal())
        match_weights = [[1.0], [1.0]]
        weights.append(match_weights)
    return composition, results, times, weights

# --- 4. Run TTT Model ---
def run_ttt(composition, results, times, weights, mu=0.0, sigma=6.0, beta=1.0, gamma=0.03, p_draw=0.1, online=False):
    history = History(
        composition=composition,
        results=results,
        times=times,
        weights=weights,
        mu=mu,
        sigma=sigma,
        beta=beta,
        gamma=gamma,
        p_draw=p_draw,
        online=online
    )
    history.convergence(iterations=10, epsilon=0.001, verbose=True)
    return history

# --- 5. Analysis & Visualization ---
def analyze_ttt_results(history, team_dict):
    learning_curves = history.learning_curves()
    reverse_team_dict = {v: k for k, v in team_dict.items()}
    final_ratings = {}
    for team_id, curve in learning_curves.items():
        if curve:
            team_name = reverse_team_dict.get(team_id, f"Team_{team_id}")
            final_ratings[team_name] = curve[-1][1].mu
    sorted_teams = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)
    return learning_curves, final_ratings, sorted_teams

def plot_team_ratings(learning_curves, team_dict, top_n=20):
    reverse_team_dict = {v: k for k, v in team_dict.items()}
    team_performances = []
    for team_id, curve in learning_curves.items():
        if curve:
            team_name = reverse_team_dict.get(team_id, f"Team_{team_id}")
            final_rating = curve[-1][1].mu
            team_performances.append((team_name, team_id, final_rating))
    team_performances.sort(key=lambda x: x[2], reverse=True)
    top_teams = team_performances[:top_n]
    plt.figure(figsize=(15, 8))
    for team_name, team_id, _ in top_teams:
        curve = learning_curves[team_id]
        times = [datetime.fromordinal(t[0]) for t in curve]
        ratings = [t[1].mu for t in curve]
        plt.plot(times, ratings, label=team_name)
    plt.title('Team Skill Ratings Over Time')
    plt.xlabel('Date')
    plt.ylabel('Skill Rating')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 6. Main Execution ---
def main():
    # --- Config ---
    # List of season URLs
    urls = [
        "https://fbref.com/en/comps/21/2015/schedule/2015-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2016/schedule/2016-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2016-2017/schedule/2016-2017-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2017-2018/schedule/2017-2018-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2018-2019/schedule/2018-2019-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2019-2020/schedule/2019-2020-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2021/schedule/2021-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2022/schedule/2022-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2023/schedule/2023-Liga-Profesional-Argentina-Scores-and-Fixtures",
        "https://fbref.com/en/comps/21/2024/schedule/2024-Liga-Profesional-Argentina-Scores-and-Fixtures"
    ]
    df_raw = load_multiple_seasons(urls)
    # --- Preprocess ---
    # TODO: Adjust column names if needed
    if 'Score' in df_raw.columns:
        df_raw = df_raw.dropna(subset=['Score'])
        df_raw['h_ftgoals'] = df_raw['Score'].str.split('–').str[0].astype(int)
        df_raw['a_ftgoals'] = df_raw['Score'].str.split('–').str[1].astype(int)
    df, team_dict = format_season_df(df_raw)
    composition, results, times, weights = prepare_ttt_data(df)
    # --- Run TTT ---
    history = run_ttt(composition, results, times, weights)
    # --- Analyze ---
    learning_curves, final_ratings, sorted_teams = analyze_ttt_results(history, team_dict)
    print('Top Teams:')
    for i, (team, rating) in enumerate(sorted_teams[:10], 1):
        print(f'{i:2d}. {team:25s} {rating:8.3f}')
    plot_team_ratings(learning_curves, team_dict)
    # TODO: Export results for Kelly or other scripts if needed

if __name__ == '__main__':
    main()

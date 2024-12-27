import pandas as pd
import kagglehub

# Download dataset
path = kagglehub.dataset_download("ajaxianazarenka/premier-league")

# Load the data
df = pd.read_csv(f"{path}/PremierLeague.csv")

# Examine what we have
season_df = df[df['Season'] == '2023-2024']

def get_team_stats(team_name):
    home_games = season_df[season_df['HomeTeam'] == team_name]
    away_games = season_df[season_df['AwayTeam'] == team_name]
    seasonStats = {
        'team': team_name,
        'matches played': len(home_games) + len(away_games),
        'goals scored': home_games['FullTimeHomeTeamGoals'].sum() + away_games['FullTimeAwayTeamGoals'].sum(),
        'goals conceded': home_games['FullTimeAwayTeamGoals'].sum() + away_games['FullTimeHomeTeamGoals'].sum(),
        'points': home_games['HomeTeamPoints'].sum() + away_games['AwayTeamPoints'].sum(),
        'win': sum(home_games['HomeTeamPoints'] == 3),
        'draw': sum(home_games['HomeTeamPoints'] == 1),
        'loss': sum(home_games['HomeTeamPoints'] == 0)
    }
    return seasonStats

def get_alltimeStat():
    alltimeStats = []
    for season in df['Season'].unique():
        season_df = df[df['Season'] == season]
        teams = season_df['HomeTeam'].unique()
        seasonStats = [get_team_stats(team) for team in teams]
        seasonStats = pd.DataFrame(seasonStats)
        seasonStats = seasonStats.sort_values(by='points', ascending=False)
        
    return alltimeStats
# Get all time statsS
alltimeStats = get_alltimeStat()
print(alltimeStats)
'''
teams = season_df['HomeTeam'].unique()
seasonStats = [get_team_stats(team) for team in teams]
seasonStats = pd.DataFrame(seasonStats)
seasonStats = seasonStats.sort_values(by='points', ascending=False)
print(seasonStats)'''
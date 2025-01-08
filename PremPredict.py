import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Download dataset
path = kagglehub.dataset_download("ajaxianazarenka/premier-league")


def seasonHomeStats(group):
    stats = pd.Series({
        'matches_played': len(group),
        'goals_scored': group['FullTimeHomeTeamGoals'].sum(),
        'goals_conceded': group['FullTimeAwayTeamGoals'].sum(),
        'shots': group['HomeTeamShots'].sum(),
        'shots_on_target': group['HomeTeamShotsOnTarget'].sum(),
        'corners': group['HomeTeamCorners'].sum(),
        'fouls': group['HomeTeamFouls'].sum(),
        'yellow_cards': group['HomeTeamYellowCards'].sum(),
        'red_cards': group['HomeTeamRedCards'].sum(),
        'points': group['HomeTeamPoints'].sum(),
        'goals_difference': (group['FullTimeHomeTeamGoals'] - group['FullTimeAwayTeamGoals']).sum(),
        'clean_sheets': (group['FullTimeAwayTeamGoals'] == 0).sum(),
        'wins': (group['FullTimeResult'] == 'H').sum(),
        'draws': (group['FullTimeResult'] == 'D').sum(),
        'losses': (group['FullTimeResult'] == 'A').sum(),
        'avg_odds': group['B365HomeTeam'].mean(),
        'market_value': group['MarketMaxHomeTeam'].mean()
    })
    return stats

def seasonAwayStats(group):
    stats = pd.Series({
        'matches_played': len(group),
        'goals_scored': group['FullTimeAwayTeamGoals'].sum(),
        'goals_conceded': group['FullTimeHomeTeamGoals'].sum(),
        'shots': group['AwayTeamShots'].sum(),
        'shots_on_target': group['AwayTeamShotsOnTarget'].sum(),
        'corners': group['AwayTeamCorners'].sum(),
        'fouls': group['AwayTeamFouls'].sum(),
        'yellow_cards': group['AwayTeamYellowCards'].sum(),
        'red_cards': group['AwayTeamRedCards'].sum(),
        'points': group['AwayTeamPoints'].sum(),
        'goals_difference': (group['FullTimeAwayTeamGoals'] - group['FullTimeHomeTeamGoals']).sum(),
        'clean_sheets': (group['FullTimeHomeTeamGoals'] == 0).sum(),
        'wins': (group['FullTimeResult'] == 'A').sum(),
        'draws': (group['FullTimeResult'] == 'D').sum(),
        'losses': (group['FullTimeResult'] == 'H').sum(),
        'avg_odds': group['B365AwayTeam'].mean(),
        'market_value': group['MarketMaxAwayTeam'].mean()
    })
    return stats

def historicalData(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    homeStats = df.groupby(['Season', 'HomeTeam'], group_keys=False).apply(seasonHomeStats, include_groups=False)
    
    homeStats = homeStats.reset_index()
    homeStats.columns = ['Season', 'Team'] + list(homeStats.columns[2:])

    awayStats = df.groupby(['Season', 'AwayTeam'], group_keys=False).apply(seasonAwayStats, include_groups=False)
    awayStats = awayStats.reset_index()
    awayStats.columns = ['Season', 'Team'] + list(awayStats.columns[2:])

    seasonStats = pd.DataFrame()
    for season in homeStats['Season'].unique():
        for team in homeStats[homeStats['Season'] == season]['Team'].unique():
            home_data = homeStats[(homeStats['Season'] == season) & (homeStats['Team'] == team)].iloc[0]
            away_data = awayStats[(awayStats['Season'] == season) & (awayStats['Team'] == team)].iloc[0]
            
            combined = pd.Series({
                'Season': season,
                'Team': team,
                'matches_played': home_data['matches_played'] + away_data['matches_played'],
                'goals_scored': home_data['goals_scored'] + away_data['goals_scored'],
                'goals_conceded': home_data['goals_conceded'] + away_data['goals_conceded'],
                'shots': home_data['shots'] + away_data['shots'],
                'shots_on_target': home_data['shots_on_target'] + away_data['shots_on_target'],
                'corners': home_data['corners'] + away_data['corners'],
                'fouls': home_data['fouls'] + away_data['fouls'],
                'yellow_cards': home_data['yellow_cards'] + away_data['yellow_cards'],
                'red_cards': home_data['red_cards'] + away_data['red_cards'],
                'points': home_data['points'] + away_data['points'],
                'goals_difference': home_data['goals_difference'] + away_data['goals_difference'],
                'clean_sheets': home_data['clean_sheets'] + away_data['clean_sheets'],
                'wins': home_data['wins'] + away_data['wins'],
                'draws': home_data['draws'] + away_data['draws'],
                'losses': home_data['losses'] + away_data['losses'],
                'avg_odds': (home_data['avg_odds'] + away_data['avg_odds']) / 2,
                'market_value': (home_data['market_value'] + away_data['market_value']) / 2
            })
            seasonStats = pd.concat([seasonStats, combined.to_frame().T], ignore_index=True)

    matches = seasonStats['matches_played'].replace(0, 1)  
    shots = seasonStats['shots'].replace(0, 1)  
    seasonStats['win_ratio'] = seasonStats['wins'] / matches
    seasonStats['goals_per_game'] = seasonStats['goals_scored'] / matches
    seasonStats['goals_conceded_per_game'] = seasonStats['goals_conceded'] / matches
    seasonStats['shots_conversion_rate'] = seasonStats['goals_scored'] / shots
    seasonStats['shots_on_target_ratio'] = seasonStats['shots_on_target'] / shots
    seasonStats = seasonStats.replace([np.inf, -np.inf], 0)
    seasonStats = seasonStats.fillna(0)

    seasonStats['is_champion'] = seasonStats.groupby('Season')['points'].transform('max') == seasonStats['points']    
    return seasonStats

def trainModel(featuresDf):
    featureCol = [
        'points', 'goals_scored', 'goals_conceded', 'shots', 'shots_on_target',
        'corners', 'fouls', 'yellow_cards', 'red_cards', 'goals_difference',
        'clean_sheets', 'wins', 'draws', 'losses', 'win_ratio',
        'goals_per_game', 'goals_conceded_per_game', 'shots_conversion_rate',
        'shots_on_target_ratio', 'avg_odds', 'market_value'
    ]
    X = featuresDf[featureCol]
    y = featuresDf['is_champion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report: \n", classification_report(y_test, y_pred))
    featureImportance = pd.DataFrame({
        'feature': featureCol,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance: \n", featureImportance)

    return model, scaler, featureCol

def create_matchup_data(team1, team2, seasonStats):
    # Get stats for both teams
    team1_stats = seasonStats[seasonStats['Team'] == team1].iloc[-1]
    team2_stats = seasonStats[seasonStats['Team'] == team2].iloc[-1]

    if team1_stats.empty:
        raise ValueError(f"Stats for team '{team1}' not found.")
    if team2_stats.empty:
        raise ValueError(f"Stats for team '{team2}' not found.")

    # Create comparative features
    matchup_features = pd.Series({
        'points': team1_stats['points'] - team2_stats['points'],  # Relative strength
        'goals_scored': team1_stats['goals_scored'] - team2_stats['goals_scored'],
        'goals_conceded': team1_stats['goals_conceded'] - team2_stats['goals_conceded'],
        'shots': team1_stats['shots'] - team2_stats['shots'],
        'shots_on_target': team1_stats['shots_on_target'] - team2_stats['shots_on_target'],
        'corners': team1_stats['corners'] - team2_stats['corners'],
        'fouls': team1_stats['fouls'] - team2_stats['fouls'],
        'yellow_cards': team1_stats['yellow_cards'] - team2_stats['yellow_cards'],
        'red_cards': team1_stats['red_cards'] - team2_stats['red_cards'],
        'goals_difference': team1_stats['goals_difference'] - team2_stats['goals_difference'],
        'clean_sheets': team1_stats['clean_sheets'] - team2_stats['clean_sheets'],
        'wins': team1_stats['wins'] - team2_stats['wins'],
        'draws': team1_stats['draws'] - team2_stats['draws'],
        'losses': team1_stats['losses'] - team2_stats['losses'],
        'win_ratio': team1_stats['win_ratio'] - team2_stats['win_ratio'],
        'goals_per_game': team1_stats['goals_per_game'] - team2_stats['goals_per_game'],
        'goals_conceded_per_game': team1_stats['goals_conceded_per_game'] - team2_stats['goals_conceded_per_game'],
        'shots_conversion_rate': team1_stats['shots_conversion_rate'] - team2_stats['shots_conversion_rate'],
        'shots_on_target_ratio': team1_stats['shots_on_target_ratio'] - team2_stats['shots_on_target_ratio'],
        'avg_odds': team1_stats['avg_odds'],  # Keep home team odds
        'market_value': team1_stats['market_value'] - team2_stats['market_value']
    })

    return matchup_features

def prepare_match_dataset(seasonStats, results_df):
    match_data = []
    for _, match in results_df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Get season for this match
        season = seasonStats[seasonStats['Team'] == home_team]['Season'].iloc[-1]
        
        # Get team stats from that season
        home_stats = seasonStats[(seasonStats['Team'] == home_team) & (seasonStats['Season'] == season)]
        away_stats = seasonStats[(seasonStats['Team'] == away_team) & (seasonStats['Season'] == season)]
        
        if len(home_stats) == 0 or len(away_stats) == 0:
            continue

        features = create_matchup_data(home_team, away_team, seasonStats[seasonStats['Season'] == season])
        features['result'] = 1 if match['FullTimeResult'] == 'H' else 0
        match_data.append(features)

    return pd.DataFrame(match_data)

def predict_match(team1, team2, model, scaler, feature_cols, seasonStats):
    """
    Predict the outcome of a match between two teams.
    """
    features = create_matchup_data(team1, team2, seasonStats)
    features_scaled = scaler.transform([features[feature_cols]])
    prob = model.predict_proba(features_scaled)[0]
    
    return {
        'home_team': team1,
        'away_team': team2,
        'home_win_probability': round(prob[1] * 100, 2),
        'away_win_probability': round(prob[0] * 100, 2)
    }

def evaluate_model_on_matches(model, scaler, feature_cols, match_df):
    X = match_df[feature_cols]
    y = match_df['result']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    print(f"Model Accuracy on Match Dataset: {accuracy}")
    print("\nClassification Report:\n", classification_report(y, y_pred))

    return accuracy

def predict_season_winner(model, scaler, feature_cols, seasonStats, current_season):
    """
    Predict the likely winner of the current season based on current stats
    """
    current_teams = seasonStats[seasonStats['Season'] == current_season]
    if current_teams.empty:
        raise ValueError(f"No data found for season {current_season}")
    
    X = current_teams[feature_cols]
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)
    
    # Create predictions dataframe
    predictions = pd.DataFrame({
        'Team': current_teams['Team'],
        'Championship_Probability': [round(prob[1] * 100, 2) for prob in probabilities]
    }).sort_values('Championship_Probability', ascending=False)
    
    return predictions

df = pd.read_csv(f"{path}/PremierLeague.csv")
current_year = datetime.now().year

current_season = f"{current_year-1}-{current_year}"
last_ten_seasons = [
    f"{current_year-i-1}-{current_year-i}" for i in range(1, 11)
]

histData = df[df['Season'].isin(last_ten_seasons + [current_season])]

results_df = histData[[
    'HomeTeam',
    'AwayTeam',
    'FullTimeResult',
    'FullTimeHomeTeamGoals',
    'FullTimeAwayTeamGoals'
]]

seasonStats = historicalData(histData)

model, scaler, feature_cols = trainModel(seasonStats)

print("\n=== Premier League Predictor ===")
print("1. Predict match outcome")
print("2. Predict season winner")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    print("\nAvailable teams:", seasonStats['Team'].unique())
    team1 = input("Enter the name of the first team (Home Team): ").strip()
    team2 = input("Enter the name of the second team (Away Team): ").strip()
    
    try:
        match_outcome = predict_match(team1, team2, model, scaler, feature_cols, seasonStats)
        print("\nMatch Prediction:")
        print(f"{match_outcome['home_team']} vs {match_outcome['away_team']}")
        print(f"Home Win Probability: {match_outcome['home_win_probability']}%")
        print(f"Away Win Probability: {match_outcome['away_win_probability']}%")
    except ValueError as e:
        print(f"Error: {e}")

elif choice == "2":
    try:
        season_predictions = predict_season_winner(model, scaler, feature_cols, seasonStats, current_season)
        print("\nSeason Winner Predictions:")
        print("Top 5 Contenders:")
        print(season_predictions.head().to_string(index=False))
    except ValueError as e:
        print(f"Error: {e}")

else:
    print("Invalid choice. Please run again and select 1 or 2.")

'''
data set keys: 
    ['MatchID', 'Season', 'MatchWeek', 'Date', 'Time', 'HomeTeam',
       'AwayTeam', 'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals',
       'FullTimeResult', 'HalfTimeHomeTeamGoals', 'HalfTimeAwayTeamGoals',
       'HalfTimeResult', 'Referee', 'HomeTeamShots', 'AwayTeamShots',
       'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget', 'HomeTeamCorners',
       'AwayTeamCorners', 'HomeTeamFouls', 'AwayTeamFouls',
       'HomeTeamYellowCards', 'AwayTeamYellowCards', 'HomeTeamRedCards',
       'AwayTeamRedCards', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam',
       'B365Over2.5Goals', 'B365Under2.5Goals', 'MarketMaxHomeTeam',
       'MarketMaxDraw', 'MarketMaxAwayTeam', 'MarketAvgHomeTeam',
       'MarketAvgDraw', 'MarketAvgAwayTeam', 'MarketMaxOver2.5Goals',
       'MarketMaxUnder2.5Goals', 'MarketAvgOver2.5Goals',
       'MarketAvgUnder2.5Goals', 'HomeTeamPoints', 'AwayTeamPoints']
'''

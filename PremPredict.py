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
    team1_stats = seasonStats[seasonStats['Team'] == team1]
    team2_stats = seasonStats[seasonStats['Team'] == team2]

    if team1_stats.empty:
        raise ValueError(f"Stats for team '{team1}' not found.")
    if team2_stats.empty:
        raise ValueError(f"Stats for team '{team2}' not found.")

    team1_stats = team1_stats.iloc[0]
    team2_stats = team2_stats.iloc[0]

    matchup_features = pd.Series({
        'goals_difference': team1_stats.get('goals_scored', 0) - team2_stats.get('goals_scored', 0),
        'shots': team1_stats.get('shots', 0) - team2_stats.get('shots', 0),
        'points': team1_stats.get('points', 0) - team2_stats.get('points', 0),
        'home_advantage': 1,
    })

    return matchup_features


def prepare_match_dataset(seasonStats, results_df):
    match_data = []
    for _, match in results_df.iterrows():
        team1 = match['HomeTeam']
        team2 = match['AwayTeam']
        result = 1 if match['FullTimeResult'] == 'H' else 0  

        features = create_matchup_data(team1, team2, seasonStats)
        features['result'] = result
        match_data.append(features)

    return pd.DataFrame(match_data)

def predict_match(team1, team2, model, seasonStats):
    features = create_matchup_data(team1, team2, seasonStats)
    prob = model.predict_proba([features])[0]
    return {
        'team1': team1,
        'team2': team2,
        'team1_win_prob': prob[1],
        'team2_win_prob': prob[0],
    }


df = pd.read_csv(f"{path}/PremierLeague.csv")
current_year = datetime.now().year

current_season = f"{current_year-1}-{current_year}"
last_ten_seasons = [
    f"{current_year-2}-{current_year-1}",
    f"{current_year-3}-{current_year-2}",
    f"{current_year-4}-{current_year-3}",
    f"{current_year-5}-{current_year-4}",
    f"{current_year-6}-{current_year-5}",
    f"{current_year-7}-{current_year-6}",
    f"{current_year-8}-{current_year-7}",
    f"{current_year-9}-{current_year-8}",
    f"{current_year-10}-{current_year-9}",
]

histData = df[df['Season'].isin(last_ten_seasons + [current_season])]
lastten = df[df['Season'].isin(last_ten_seasons + [current_season])]
resultsdf = lastten[[
    'HomeTeam',
    'AwayTeam',
    'FullTimeResult',
    'FullTimeHomeTeamGoals',
    'FullTimeAwayTeamGoals'
]]
seasonStats = historicalData(histData)
model, scaler, featureCol = trainModel(seasonStats)
print(seasonStats['Team'].unique())
team1 = input("Enter the name of the first team (Home Team): ")
team2 = input("Enter the name of the second team (Away Team): ")
matchset = prepare_match_dataset(seasonStats, resultsdf)
print(matchset)
match_outcome = predict_match(model, team1.strip(), team2.strip(), seasonStats)
print(match_outcome)

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

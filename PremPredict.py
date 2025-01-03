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
    # Fetch stats for both teams
    team1_stats = seasonStats['Team'] == team1
    team2_stats = seasonStats['Team'] == team2

    # Calculate matchup-specific features
    matchup_features = pd.Series({
        'goals_difference': team1_stats['goals_scored'] - team2_stats['goals_scored'],
        'shots': team1_stats['shots'] - team2_stats['shots'],
        'points': team1_stats['points'] - team2_stats['points'],
        'home_advantage': 1
    })

    return matchup_features

def prepare_match_dataset(seasonStats, results_df):
    match_data = []
    for _, match in results_df.iterrows():
        team1 = match['HomeTeam']
        team2 = match['AwayTeam']
        result = 1 if match['FullTimeResult'] == 'H' else 0  # 1 for home win

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

seasonStats = historicalData(histData)
model, scaler, featureCol = trainModel(seasonStats)

print(seasonStats['Team'].unique())
team1 = input("Enter the name of the first team (Home Team): ")
team2 = input("Enter the name of the second team (Away Team): ")

match_outcome = predict_match(model, team1, team2, seasonStats)
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

Cumulative Stats: 
                team  matches played  goals scored  goals conceded  points  wins  draws  losses
0         Man United            1183          2220            1140    2399   410    110      71
3            Arsenal            1182          2156            1177    2259   390    129      72
5          Liverpool            1179          2122            1166    2201   378    141      71
11           Chelsea            1180          2038            1176    2184   359    148      83
16         Tottenham            1180          1830            1483    1851   326    127     138
19          Man City             992          1794            1078    1755   305     93      98
15           Everton            1182          1510            1533    1596   273    153     165
2          Newcastle            1106          1517            1505    1535   274    137     141
8        Aston Villa            1067          1341            1410    1418   219    152     164
14          West Ham            1065          1324            1562    1334   229    133     169
18       Southampton             876          1039            1289    1017   167    122     150
1          Blackburn             640           842             845     878   153     79      89
24         Leicester             653           879             944     823   129     89     108
33            Fulham             656           755             958     754   140     71     117
4              Leeds             525           704             712     735   117     68      76
25     Middlesbrough             532           594             719     620   102     79      85
28        Sunderland             608           612             904     618    98     87     119
23    Crystal Palace             537           595             771     608    86     66     117
26            Bolton             494           575             745     575    93     75      79
35         West Brom             494           510             772     490    76     66     105
40             Stoke             380           398             525     457    81     54      55
37            Wolves             390           423             612     438    72     43      80
30          Charlton             304           342             442     361    58     40      54
42           Burnley             342           341             533     349    54     42      75
12          Coventry             299           320             422     335    55     50      45
47          Brighton             276           334             387     335    47     49      42
38             Wigan             304           316             482     331    48     45      59
10           Norwich             336           342             583     323    53     50      65
6          Wimbledon             260           311             401     318    50     38      41
46       Bournemouth             276           345             480     313    50     36      52
44           Swansea             266           306             383     312    51     37      45
9     Sheffield Weds             260           331             392     309    49     40      41
36        Portsmouth             266           292             380     302    54     34      45
34        Birmingham             266           273             360     301    50     46      37
32           Watford             304           310             518     285    45     40      67
22     Nott'm Forest             236           280             363     278    42     40      36
27             Derby             266           271             420     274    48     35      50
7                QPR             217           256             348     222    37     30      41
41              Hull             190           181             323     171    29     24      42
13           Ipswich             155           171             248     171    26     21      31
20  Sheffield United             187           158             313     162    29     23      42
49         Brentford             124           181             187     157    26     18      18
39           Reading             114           136             186     119    23     12      22
45           Cardiff              76            66             143      64    11      7      20
31          Bradford              76            68             138      62    10     15      13
48      Huddersfield              76            50             134      53     8      8      22
43         Blackpool              38            55              78      39     5      5       9
29          Barnsley              38            37              82      35     7      4       8
17            Oldham              33            31              52      34     4      6       6
21           Swindon              36            41              86      26     4      7       7
50             Luton              38            52              85      26     4      4      11


'''

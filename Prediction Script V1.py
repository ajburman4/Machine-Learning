"Import Data"

import pandas as pd

stats_2037 = pd.read_excel("C:/Users/ajbur/OneDrive/Documents/Sim League/Sim League Exports - 2037-2038 Stats.xlsx")
ratings_2037 = pd.read_excel("C:/Users/ajbur/OneDrive/Documents/Sim League/Sim League Exports - 2037-2038 Ratings.xlsx")
wins_2037 = pd.read_excel("C:/Users/ajbur/OneDrive/Documents/Sim League/2037-2038 Wins.xlsx")

stats_2037["Full_Name"] = stats_2037["FName"] + " " + stats_2037["LName"]
ratings_2037["Full_Name"] = ratings_2037["FName"] + " " + ratings_2037["LName"]

players_2037a = pd.merge(stats_2037, ratings_2037,on = "Full_Name", how = "inner")

"Create New Metrics"
players_2037a['FG%'] = players_2037a['FGM'] / players_2037a['FGA']
players_2037a['FT%'] = players_2037a['FTM'] / players_2037a['FTA']
players_2037a['3P%'] = players_2037a['3PM'] / players_2037a['3PA']
players_2037a['DReb'] = players_2037a['Reb'] - players_2037a['OReb']
players_2037 = players_2037a.fillna(players_2037a.mean())
print(players_2037.loc[[262]])

"Get Team Stats"
team_stats = players_2037.groupby('Team_x').agg(
    FGM = ('FGM', sum),
    FGA = ('FGA', sum),
    FTM = ('FTM', sum),
    FTA = ('FTA', sum),
    TPM = ('3PM', sum),
    TPA = ('3PA', sum),
    OREB = ('OReb', sum),
    DREB = ('DReb', sum),
    AST = ('Ast', sum),
    STL = ('Stl_x', sum),
    BLK = ('Blk_x', sum),
    TO = ('TO', sum),
    )

team_stats['FG%'] = team_stats['FGM'] / team_stats['FGA']
team_stats['FT%'] = team_stats['FTM'] / team_stats['FTA']
team_stats['3P%'] = team_stats['TPM'] / team_stats['TPA']

team_results = pd.merge(team_stats, wins_2037, on = "Team_x", how = "inner")

print(team_results)

"Prep for model - performance metric and data split"
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

features_raw = team_results.drop(columns = ['FGM', 'FTM', 'TPM', 'Wins', 'Team_x'])
player_features_raw = pd.DataFrame(players_2037, columns = ['FGA', 'FTA', '3PA', 'OReb', 'DReb', 'Ast', 'Stl_x', 'Blk_x', 'TO', 'FG%', 'FT%', '3P%'])
print(player_features_raw)
player_features = player_features_raw.rename(columns = {'3PA': 'TPA',
                                                        'OReb': 'OREB',
                                                        'DReb': 'DREB',
                                                        'Ast': 'AST',
                                                        'Stl_x': 'STL',
                                                        'Blk_x': 'BLK',
                                                        })

wins = team_results['Wins']

scorer = make_scorer(r2_score)

def fit_model(X, y, r, p):
    cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20)
    regressor = r
    params = p
    grid = GridSearchCV(regressor, params, scoring = scorer, cv = cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_

"Normalize data to prevent feature domination"
'from sklearn.preprocessing import MinMaxScaler'
'scaler = MinMaxScaler()'
'features = scaler.fit_transform(features_raw)'
'player_features_scaled = scaler.fit_transform(player_features)'
"player_features_df = pd.DataFrame(player_features_scaled, columns = ['FGA', 'FTA', 'TPA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TO', 'FG%', 'FT%', '3P%'])"
'print(player_features_df)'

"Random Forest"
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()
rf_params = {'max_depth':range(1,21)}
rf_model = fit_model(features_raw, wins, rf_regressor, rf_params)
rf_predict = rf_model.predict(player_features)
print(rf_predict)

players_2037['Proj_Wins'] = rf_predict
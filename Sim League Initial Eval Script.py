
"Import Data"

import pandas as pd

stats_2037 = pd.read_excel("C:/Users/ajbur/OneDrive/Documents/Sim League/Sim League Exports - 2037-2038 Stats.xlsx")
ratings_2037 = pd.read_excel("C:/Users/ajbur/OneDrive/Documents/Sim League/Sim League Exports - 2037-2038 Ratings.xlsx")
wins_2037 = pd.read_excel("C:/Users/ajbur/OneDrive/Documents/Sim League/2037-2038 Wins.xlsx")

print(stats_2037[1:3])
print(ratings_2037[1:3])
print(len(stats_2037))
print(len(ratings_2037))

stats_2037["Full_Name"] = stats_2037["FName"] + " " + stats_2037["LName"]
ratings_2037["Full_Name"] = ratings_2037["FName"] + " " + ratings_2037["LName"]

players_2037 = pd.merge(stats_2037, ratings_2037,on = "Full_Name", how = "inner")
players_2037.drop(columns = ["FName_y", "LName_y", "Pos_y"], inplace=True)
print(players_2037[0:0])
print(players_2037[0:5])
print(len(players_2037))

"Create New Metrics"
players_2037['FG%'] = players_2037['FGM'] / players_2037['FGA']
players_2037['FT%'] = players_2037['FTM'] / players_2037['FTA']
players_2037['3P%'] = players_2037['3PM'] / players_2037['3PA']
players_2037['DReb'] = players_2037['Reb'] - players_2037['OReb']

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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

features_raw = team_results.drop(columns = ['FGM', 'FTM', 'TPM', 'Wins', 'Team_x'])
wins = team_results['Wins']

'x_train, x_test, y_train, y_test = train_test_split(features, wins, test_size = .2)'

scorer = make_scorer(r2_score)

def fit_model(X, y, r, p):
    cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20)
    regressor = r
    params = p
    grid = GridSearchCV(regressor, params, scoring = scorer, cv = cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_

"Normalize data to prevent feature domination"
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features_raw)

"Linear Regression"
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
'linear_model.fit(x_train, y_train)'
'linear_predict = linear_model.predict(x_test)'
'linear_r2 = r2_score(y_test, linear_predict)'
'print(linear_r2)'
'print(linear_predict)'
"-.6894 -- this is literal trash"

"Decision Tree"
from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor()
dt_params = {'max_depth':range(1,11)}
'dt_model = fit_model(x_train, y_train, dt_regressor, dt_params)'
'dt_predict = dt_model.predict(x_test)'
'dt_r2 = r2_score(y_test, dt_predict)'
'print(dt_r2)'
'print(dt_model)'

"Random Forest"
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()
rf_params = {'max_depth':range(1,11)}
'rf_model = fit_model(x_train, y_train, rf_regressor, rf_params)'
'rf_predict = rf_model.predict(x_test)'
'rf_r2 = r2_score(y_test, rf_predict)'
'print(rf_r2)'
'print(rf_model)'

"Support Vector Machine"
from sklearn.svm import SVR

svr_regressor = SVR()

svr_params = {'kernel':['rbf', 'linear', 'poly'], 'C':[.1, 1, 10]}
'svr_model = fit_model(x_train, y_train, svr_regressor, svr_params)'
'svr_predict = svr_model.predict(x_test)'
'svr_r2 = r2_score(y_test, svr_predict)'
'print(svr_r2)'
'print(svr_model)'


"Neural Network"
from sklearn.neural_network import MLPRegressor

mlp_regressor = MLPRegressor()
mlp_params = {'solver':['lbfgs', 'sgd', 'adam'], 'learning_rate':['constant', 'invscaling', 'adaptive']}
'mlp_model = fit_model(x_train, y_train, mlp_regressor, mlp_params)'
'mlp_predict = mlp_model.predict(x_test)'
'mlp_r2 = r2_score(y_test, mlp_predict)'
'print(mlp_r2)'
'print(mlp_model)'

"Model Selection"
model_selection_df = pd.DataFrame()
model_selection_df['Linear_Regression'] = range(1,101)
model_selection_df['Linear_Regression'] = model_selection_df['Linear_Regression'].astype(float)
model_selection_df['Decision_Tree'] = range(1,101)
model_selection_df['Decision_Tree'] = model_selection_df['Decision_Tree'].astype(float)
model_selection_df['Random_Forest'] = range(1,101)
model_selection_df['Random_Forest'] = model_selection_df['Random_Forest'].astype(float)
model_selection_df['SVR'] = range(1,101)
model_selection_df['SVR'] = model_selection_df['SVR'].astype(float)
model_selection_df['MLP'] = range(1,101)
model_selection_df['MLP'] = model_selection_df['MLP'].astype(float)
print(model_selection_df)

def model_sim(column, regressor, parameters):
    for i in range(0,100):
        x_train, x_test, y_train, y_test = train_test_split(features, wins, test_size = .2)
        model = fit_model(x_train, y_train, regressor, parameters)
        prediction = model.predict(x_test)
        r2 = r2_score(y_test, prediction)
        model_selection_df[column][i] = r2

for i in range(0,100):
    x_train, x_test, y_train, y_test = train_test_split(features, wins, test_size = .2)
    linear_model.fit(x_train, y_train)
    linear_predict = linear_model.predict(x_test)
    r2 = r2_score(y_test, linear_predict)
    model_selection_df['Linear_Regression'][i] = r2

model_sim('Decision_Tree', dt_regressor, dt_params)
model_sim('Random_Forest', rf_regressor, rf_params)
model_sim('SVR', svr_regressor, svr_params)
model_sim('MLP', mlp_regressor, mlp_params)

'print(model_selection_df)'

model_performance = pd.DataFrame()
model_performance['LR_R2'] = model_selection_df['Linear_Regression'].mean()
model_performance['LR_R2'] = model_selection_df['Linear_Regression'].std()
model_performance['DT_R2'] = model_selection_df['Decision_Tree'].mean()
model_performance['DT_R2'] = model_selection_df['Decision_Tree'].std()
model_performance['RF_R2'] = model_selection_df['Random_Forest'].mean()
model_performance['RF_R2'] = model_selection_df['Random_Forest'].std()
model_performance['SVR_R2'] = model_selection_df['SVR'].mean()
model_performance['SVR_R2'] = model_selection_df['SVR'].std()
model_performance['MLP'] = model_selection_df['MLP'].mean()
model_performance['MLP'] = model_selection_df['MLP'].std()

print(model_performance)

""


"GRAVEYARD"
'print(dt_model.get_params)'

'print(dt_model.predict(x_test))'

'dt_predict = dt_model.predict(x_test)'
'print(dt_model.score(dt_predict, y_test))'



"team_series = pd.Series(players_2037['Team_x'])"
'team_series = team_series.drop_duplicates()'
'print(team_series)'

'team_df = pd.DataFrame()'

'print(team_df)'
'def performance_metric(y_true, y_predict):'
'score = r2_score(y_true, y_predict)'
'return score'
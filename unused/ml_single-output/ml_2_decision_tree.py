## Import packages
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV



## Import data
pv_train = pd.read_csv('YMCA_train.csv')
pv_test = pd.read_csv('YMCA_test.csv')

input_cols = ['TempOut', 'OutHum', 'WindSpeed', 'Bar', 'SolarRad']
output_cols = ['NRM_P_GEN_MIN', 'NRM_P_GEN_MAX']

# Bin separation
# Block (0-3) => BIN 1
cut_blocks = [i for i in range(1, 9)]

pv_train['Bin'] = pd.cut(pv_train['Block'], 8, labels = cut_blocks)
pv_train.style



## Setup models
# Decision Tree regression
dt = DecisionTreeRegressor(random_state=0)



## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train.NRM_P_GEN_MIN
# y_train = pv_train.NRM_P_GEN_MAX

# Parameters to search through
criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
splitter = ['best', 'random']
min_samples_split = [x for x in range(2,15,1)]
min_samples_leaf = [x for x in range(1,50,1)]
max_features = ['sqrt', 'log2', None]

# Put all hyperparameter into a dict
random_grid = {
    'criterion': criterion,
    'splitter': splitter,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features
}

# Search thoroughly for optimised hyperparameter
dt_gcv = GridSearchCV(estimator=dt,
                        param_grid=random_grid,
                        scoring=['neg_root_mean_squared_error','neg_mean_absolute_error'],
                        refit='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
dt_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(dt_gcv.best_params_)
print('\n\n')



dt.fit(x_train, y_train)
dt_opt = DecisionTreeRegressor(criterion='squared_error',
                               max_features=None,
                               min_samples_leaf=28,
                               min_samples_split=2,
                               splitter='random',
                               random_state= 0)
dt_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test.NRM_P_GEN_MIN
# y_test = pv_test.NRM_P_GEN_MAX

y_pred_dt = dt.predict(x_test)
y_pred_dt_opt = dt_opt.predict(x_test)

print(f'RMSE for Test Data (Decision Tree): {mean_squared_error(y_test, y_pred_dt, squared=False)}')
print(f'RMSE for Test Data (Decision Tree Optimised): {mean_squared_error(y_test, y_pred_dt_opt, squared=False)}')

## Code reference:
## https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de
## https://stackoverflow.com/questions/43532811/gridsearch-over-multioutputregressor

## Import packages
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
# Linear regression
ln = MultiOutputRegressor(LinearRegression(), n_jobs=-1)
# Lasso regression
lasso = MultiOutputRegressor(Lasso(random_state= 0), n_jobs=-1)
ls = MultiTaskLasso(random_state=0)



## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train[output_cols]

# Parameters to search through
max_iter = [500, 1000, 2000, 5000]
selection = ['cyclic', 'random']

# Put all hyperparameter into a dict
random_grid = {
    'estimator__max_iter': max_iter,
    'estimator__selection': selection,
}
rg = {
    'max_iter': max_iter,
    'selection': selection
}

# Search thoroughly for optimised hyperparameter
lasso_gcv = GridSearchCV(estimator=lasso,
                        param_grid=random_grid,
                        scoring=['neg_root_mean_squared_error','neg_mean_absolute_error'],
                        refit='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
lasso_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(lasso_gcv.best_params_)
print(lasso_gcv.best_estimator_)
print('\n\n')

# Search thoroughly for optimised hyperparameter
ls_gcv = GridSearchCV(estimator=ls,
                        param_grid=rg,
                        scoring=['neg_root_mean_squared_error','neg_mean_absolute_error'],
                        refit='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
ls_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(ls_gcv.best_params_)
print(ls_gcv.best_estimator_)
print('\n\n')



ln.fit(x_train, y_train)
lasso.fit(x_train, y_train)
lasso_opt = MultiOutputRegressor(Lasso(random_state= 0, max_iter=500, selection='cyclic'),
                                 n_jobs=-1)
lasso_opt.fit(x_train, y_train)
ls.fit(x_train, y_train)
ls_opt = MultiTaskLasso(max_iter=500,random_state=0,selection='cyclic')
ls_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test[output_cols]

y_pred_ln = ln.predict(x_test)
y_pred_lasso = lasso.predict(x_test)
y_pred_ls = ls.predict(x_test)
y_pred_lasso_opt = lasso_opt.predict(x_test)
y_pred_ls_opt = ls_opt.predict(x_test)

print(f'Root Mean Squared Error for Test Data (Linear): {mean_squared_error(y_test, y_pred_ln, squared=False)}')
print(f'Root Mean Squared Error for Test Data (Lasso): {mean_squared_error(y_test, y_pred_lasso, squared=False)}')
print(f'Root Mean Squared Error for Test Data (Lasso Optimised): {mean_squared_error(y_test, y_pred_lasso_opt, squared=False)}')
print(f'Root Mean Squared Error for Test Data (MultiLasso): {mean_squared_error(y_test, y_pred_ls, squared=False)}')
print(f'Root Mean Squared Error for Test Data (MultiLasso Optimised): {mean_squared_error(y_test, y_pred_ls_opt, squared=False)}')

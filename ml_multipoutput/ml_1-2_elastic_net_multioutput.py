## Code reference:
## https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de
## https://stackoverflow.com/questions/43532811/gridsearch-over-multioutputregressor

## Import packages
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, MultiTaskElasticNet
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
# Linear regression
ln = MultiOutputRegressor(LinearRegression(), n_jobs=-1)
# Lasso regression
eln = MultiOutputRegressor(ElasticNet(random_state= 0), n_jobs=-1)
meln = MultiTaskElasticNet(random_state=0)

## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train[output_cols]

# Parameters to search through
l1_ratio = [x for x in np.linspace(start=0, stop=1, num=101)]
max_iter = [500, 1000, 2000, 5000]
selection = ['cyclic', 'random']

# Put all hyperparameter into a dict
random_grid = {
    'estimator__l1_ratio': l1_ratio,
    'estimator__max_iter': max_iter,
    'estimator__selection': selection,
}
rg = {
    'l1_ratio': l1_ratio,
    'max_iter': max_iter,
    'selection': selection
}

# Search thoroughly for optimised hyperparameter
eln_gcv = GridSearchCV(estimator=eln,
                        param_grid=random_grid,
                        scoring=['neg_root_mean_squared_error','neg_mean_absolute_error'],
                        refit='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
eln_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(eln_gcv.best_params_)
print(eln_gcv.best_estimator_)
print('\n\n')

# Search thoroughly for optimised hyperparameter
meln_gcv = GridSearchCV(estimator=meln,
                        param_grid=rg,
                        scoring=['neg_root_mean_squared_error','neg_mean_absolute_error'],
                        refit='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
meln_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(meln_gcv.best_params_)
print(meln_gcv.best_estimator_)
print('\n\n')



ln.fit(x_train, y_train)
eln.fit(x_train, y_train)
eln_opt = MultiOutputRegressor(ElasticNet(random_state= 0, l1_ratio=0.2,
                                          max_iter=500, selection='cyclic'),
                               n_jobs=-1)
eln_opt.fit(x_train, y_train)
meln.fit(x_train, y_train)
meln_opt = MultiTaskElasticNet(random_state=0, l1_ratio=0.23, max_iter=500, selection='cyclic')
meln_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test[output_cols]

y_pred_ln = ln.predict(x_test)
y_pred_eln = eln.predict(x_test)
y_pred_eln_opt = eln_opt.predict(x_test)
y_pred_meln = meln.predict(x_test)
y_pred_meln_opt = meln_opt.predict(x_test)

print(f'RMSE for Test Data (Linear): {mean_squared_error(y_test, y_pred_ln, squared=False)}')
print(f'RMSE for Test Data (ElasticNet): {mean_squared_error(y_test, y_pred_eln, squared=False)}')
print(f'RMSE for Test Data (ElasticNet Optimised): {mean_squared_error(y_test, y_pred_eln_opt, squared=False)}')
print(f'RMSE for Test Data (MultiEN): {mean_squared_error(y_test, y_pred_meln, squared=False)}')
print(f'RMSE for Test Data (MultiEN Optimised): {mean_squared_error(y_test, y_pred_meln_opt, squared=False)}')

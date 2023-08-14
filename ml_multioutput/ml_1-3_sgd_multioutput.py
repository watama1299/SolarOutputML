## Code reference:
## https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de
## https://stackoverflow.com/questions/43532811/gridsearch-over-multioutputregressor

## Import packages
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV



## Import data
pv_train = pd.read_csv('YMCA_data_train.csv')
pv_test = pd.read_csv('YMCA_data_test.csv')

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
sgd = MultiOutputRegressor(SGDRegressor(random_state= 0, early_stopping=True,
                                        n_iter_no_change=10, shuffle=True),
                           n_jobs=-1)

## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train[output_cols]

# Parameters to search through
loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
penalty = ['l2', 'l1', 'elasticnet', None]
l1_ratio = [x for x in np.linspace(start=0, stop=1, num=101)]
max_iter = [500, 1000, 2000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']

# Put all hyperparameter into a dict
grid = {
    'estimator__loss': loss,
    'estimator__penalty': penalty,
    'estimator__l1_ratio': l1_ratio,
    'estimator__max_iter': max_iter,
    'estimator__learning_rate': learning_rate,
}

# Search thoroughly for optimised hyperparameter
sgd_gcv = GridSearchCV(estimator=sgd,
                        param_grid=grid,
                        scoring=['neg_root_mean_squared_error','neg_mean_absolute_error'],
                        refit='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
sgd_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(sgd_gcv.best_params_)
print(sgd_gcv.best_estimator_)
print('\n\n')



ln.fit(x_train, y_train)
sgd.fit(x_train, y_train)
sgd_opt = MultiOutputRegressor(SGDRegressor(loss='huber', penalty='elasticnet',
                                            l1_ratio=0.72, max_iter=500,
                                            shuffle=True, random_state=0,
                                            learning_rate='adaptive',
                                            early_stopping=True,
                                            n_iter_no_change=10,),
                               n_jobs=-1)
sgd_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test[output_cols]

y_pred_ln = ln.predict(x_test)
y_pred_sgd = sgd.predict(x_test)
y_pred_sgd_opt = sgd_opt.predict(x_test)

print(f'RMSE for Test Data (Linear): {mean_squared_error(y_test, y_pred_ln, squared=False)}')
print(f'RMSE for Test Data (ElasticNet): {mean_squared_error(y_test, y_pred_sgd, squared=False)}')
print(f'RMSE for Test Data (ElasticNet Optimised): {mean_squared_error(y_test, y_pred_sgd_opt, squared=False)}')

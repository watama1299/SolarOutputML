## Code reference: https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de

## Import packages
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR, NuSVR
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
svm = MultiOutputRegressor(LinearSVR(random_state=0), n_jobs=-1)
svm_nu = MultiOutputRegressor(NuSVR(cache_size=1000, max_iter=-1), n_jobs=-1)



## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train[output_cols]

# Parameters to search through
loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
dual = [True, False]
max_iter = [1000, 2000, 3000]

nu = [0.43, 0.44, 0.45]
kernel = ['linear', 'rbf', 'sigmoid']
gamma = ['scale', 'auto']
cache_size = [200, 500, 1000]

# Put all hyperparameter into a dict
rg_svm = {
    'estimator__loss': loss,
    'estimator__dual': dual,
    'estimator__max_iter': max_iter
}

rg_svm_nu = {
    'estimator__nu': nu,
    'estimator__kernel': kernel,
    'estimator__gamma': gamma,
    'estimator__cache_size': cache_size
}

# Search thoroughly for optimised hyperparameter
svm_gcv = GridSearchCV(estimator=svm,
                        param_grid=rg_svm,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
svm_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(svm_gcv.best_params_)
print('\n\n')

svm_nu_gcv = GridSearchCV(estimator=svm_nu,
                        param_grid=rg_svm_nu,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
svm_nu_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(svm_nu_gcv.best_params_)
print('\n\n')



## Training
svm.fit(x_train, y_train)
svm_nu.fit(x_train, y_train)

svm_opt = MultiOutputRegressor(LinearSVR(random_state=0,
                                         loss='squared_epsilon_insensitive',
                                         dual=False, max_iter=1000),
                               n_jobs=-1)
svm_opt.fit(x_train, y_train)

svm_nu_opt = MultiOutputRegressor(NuSVR(nu=0.45, kernel='rbf',
                                        gamma='scale', cache_size=200),
                                  n_jobs=-1)
svm_nu_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test[output_cols]

y_pred_svm = svm.predict(x_test)
y_pred_svm_opt = svm_opt.predict(x_test)
y_pred_svm_nu = svm_nu.predict(x_test)
y_pred_svm_nu_opt = svm_nu_opt.predict(x_test)

print(f'RMSE for Test Data (Support Vector Regression): {mean_squared_error(y_test, y_pred_svm, squared=False)}')
print(f'RMSE for Test Data (SVR Optimised): {mean_squared_error(y_test, y_pred_svm_opt, squared=False)}')
print(f'RMSE for Test Data (NuSVR): {mean_squared_error(y_test, y_pred_svm_nu, squared=False)}')
print(f'RMSE for Test Data (NuSVR Optimised): {mean_squared_error(y_test, y_pred_svm_nu_opt, squared=False)}')
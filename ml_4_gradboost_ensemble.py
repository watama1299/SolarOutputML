## Code reference: https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de

## Import packages
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
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
gb = GradientBoostingRegressor(random_state=0, verbose=1)



## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train.NRM_P_GEN_MAX

# Parameters to search through
loss = ['squared_error', 'absolute_error', 'huber', 'quantile']
n_estimators = [x for x in np.linspace(start=100, stop=1000, num=11)]
criterion = ['friedman_mse', 'squared_error']

# Put all hyperparameter into a dict
random_grid = {
    'loss': loss,
    'n_estimators': n_estimators,
    'criterion': criterion,

}

# Search thoroughly for optimised hyperparameter
gb_gcv = GridSearchCV(estimator=gb,
                        param_grid=random_grid,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
gb_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(gb_gcv.best_params_)
print('\n\n')



gb.fit(x_train, y_train)
gb_opt = GradientBoostingRegressor(random_state=0, verbose=1)
gb_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test.NRM_P_GEN_MAX

y_pred_gb = gb.predict(x_test)
y_pred_gb_opt = gb_opt.predict(x_test)

print(f'RMSE for Test Data (Support Vector Regression): {mean_squared_error(y_test, y_pred_gb, squared=False)}')
print(f'RMSE for Test Data (NuSVR Optimised): {mean_squared_error(y_test, y_pred_gb_opt, squared=False)}')
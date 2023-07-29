## Import packages
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
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
# ANN (BPNN)
ann_sk = MLPRegressor(solver='adam', random_state=0, 
                      shuffle=True, early_stopping=True, 
                      verbose=False)



## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train[output_cols]

# Parameters to search through
hidden_layer_sizes = [(100,), (50,), (10,), (5,), (100,100,), (50,50,), (10,10,), (5,5,)]
# hidden_layer_sizes = [(110, 110,), (100, 100,), (90, 90,)]
activation = ['identity', 'logistic', 'tanh', 'relu']
max_iter = [200, 500, 1000]


# Put all hyperparameter into a dict
random_grid = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'max_iter': max_iter,
    'activation': activation
}

# Search thoroughly for optimised hyperparameter
ann_sk_gcv = GridSearchCV(estimator=ann_sk,
                        param_grid=random_grid,
                        scoring=['neg_root_mean_squared_error','neg_mean_absolute_error'],
                        refit='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
ann_sk_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(ann_sk_gcv.best_params_)
print(ann_sk_gcv.best_estimator_)
print('\n\n')



ann_sk.fit(x_train, y_train)
ann_sk_opt = MLPRegressor(hidden_layer_sizes=(100,100,), activation='logistic', 
                          solver='adam', max_iter=200, shuffle=True, 
                          random_state=0, early_stopping=True)
ann_sk_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test[output_cols]

y_pred_ann_sk = ann_sk.predict(x_test)
y_pred_ann_sk_opt = ann_sk_opt.predict(x_test)

print(f'RMSE for Test Data (MLP): {mean_squared_error(y_test, y_pred_ann_sk, squared=False)}')
print(f'RMSE for Test Data (MLP Optimised): {mean_squared_error(y_test, y_pred_ann_sk_opt, squared=False)}')
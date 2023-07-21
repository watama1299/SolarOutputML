## Code reference: https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de

## Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, NuSVR, SVR
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
svm = LinearSVR(random_state=0)
svm_nu = NuSVR(verbose=True)



## Hyperparameter Optimisation
x_train = pv_train[input_cols]
y_train = pv_train.NRM_P_GEN_MAX

# Parameters to search through
nu = [x for x in np.linspace(start=0, stop=1, num=101)]
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
gamma = ['scale', 'auto']

# Put all hyperparameter into a dict
rg_svm = {
    'nu': nu,
    'kernel': kernel,
    'gamma': gamma
}

# Search thoroughly for optimised hyperparameter
svm_gcv = GridSearchCV(estimator=svm_nu,
                        param_grid=rg_svm,
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        verbose=3)
svm_gcv.fit(x_train, y_train)

# Print best hyperparameter
print(svm_gcv.best_params_)
print('\n\n')



svm.fit(x_train, y_train)
svm_nu_opt = NuSVR()
svm_nu_opt.fit(x_train, y_train)



## Predicting
x_test = pv_test[input_cols]
y_test = pv_test.NRM_P_GEN_MAX

y_pred_svm = svm.predict(x_test)
y_pred_svm_nu = svm_nu.predict(x_test)
y_pred_svm_nu_opt = svm_nu_opt.predict(x_test)

print(f'RMSE for Test Data (Support Vector Regression): {mean_squared_error(y_test, y_pred_svm, squared=False)}')
print(f'RMSE for Test Data (NuSVR): {mean_squared_error(y_test, y_pred_svm_nu, squared=False)}')
print(f'RMSE for Test Data (NuSVR Optimised): {mean_squared_error(y_test, y_pred_svm_nu_opt, squared=False)}')
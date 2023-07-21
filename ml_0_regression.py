## Code reference: https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de

## Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_validate



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



## Setup Pipeline
## Without hyperparameter optimisation
# Linear regression
model_ln = Pipeline([('lin_regression', LinearRegression())])
# Tree regression
model_dt = Pipeline([('dt_regression', DecisionTreeRegressor(random_state=0))])
# Linear Support Vector regression
model_svr = Pipeline([('svm_regression', SVR())])
# Ensemble regression
model_grb = Pipeline([('gradboost_regression', GradientBoostingRegressor(random_state=0))])
models = [
    model_ln,
    model_dt,
    model_svr,
    model_grb
]
reg_dict = {
    0: 'Linear',
    1: 'Decision Tree',
    2: 'SVM',
    3: 'Grad Boost'
}

## With hyperparameter optimisation
# Lasso regression
model_lasso_opt = Pipeline([('lin_regression', Lasso(random_state= 0, max_iter=500, selection='cyclic'))])
# ElasticNet regression
model_eln_opt = Pipeline([('lin_regression', ElasticNet(random_state= 0, l1_ratio=0.2, max_iter=500, selection='cyclic'))])
# Tree regression
model_dt_opt = Pipeline([('dt_regression', DecisionTreeRegressor(criterion='absolute_error', min_samples_leaf=28, min_samples_split=2, splitter='random', random_state= 0))])
# Linear Support Vector regression
model_svr_opt = Pipeline([('svm_regression', LinearSVR(random_state= 0, loss='squared_epsilon_insensitive', max_iter=500))])
# Ensemble regression
model_grb_opt = Pipeline([('gradboost_regression', GradientBoostingRegressor(random_state=0))])
models_opt = [
    model_ln,
    model_lasso_opt,
    model_eln_opt,
    model_dt_opt,
    model_svr_opt,
    model_grb_opt
]
reg_opt_dict = {
    0: 'Linear',
    1: 'Lasso',
    2: 'ElasticNet',
    3: 'Decision Tree',
    4: 'SVM',
    5: 'Grad Boost'
}



## k-Fold Cross Validation
import warnings
warnings.filterwarnings("ignore")

skf = StratifiedKFold(n_splits=10, random_state=23, shuffle=True)
pv_train['kfold'] = -1

for fold,(train_indices, valid_indices) in enumerate(skf.split(X=pv_train.iloc[:,:-1], y=pv_train['Bin'])):
    pv_train.loc[valid_indices, 'kfold'] = fold

def model_comparison(model_list, model_dict):
    best_rmse = 100.0
    best_regressor = 0
    best_pipeline = ""

    for j, model in enumerate(model_list):
        print(model_dict[j])
        RMSE = list()
        for i in range(8):
            # copy training data that doesnt match the fold value into xtrain
            xtrain = pv_train[pv_train['kfold'] != i]
            # copy validation data that match the fold value into xvalid
            xvalid = pv_train[pv_train['kfold'] == i]

            ytrain = xtrain.P_GEN_MAX
            yvalid = xvalid.P_GEN_MAX

            xtrain = xtrain[input_cols]
            xvalid = xvalid[input_cols]

            scaler = StandardScaler()
            scaler.fit_transform(xtrain)
            scaler.transform(xvalid)

            model.fit(xtrain, ytrain)
            # rmse = np.sqrt(mean_squared_error(yvalid, model.predict(xvalid)))
            rmse = mean_squared_error(yvalid, model.predict(xvalid), squared=False)
            RMSE.append(rmse)


        folds_mean_rmse = np.mean(RMSE)
        print('Mean Validation RMSE: {}\n'.format(folds_mean_rmse))


        if folds_mean_rmse < best_rmse:
            best_rmse = folds_mean_rmse
            best_pipeline = model
            best_regressor = j

    print('\nRegressor with least RMSE: {}'.format(model_dict[best_regressor]))
    print(best_pipeline)

## Initial comparison
model_comparison(models, reg_dict)

## Optimised comparison
model_comparison(models_opt, reg_opt_dict)
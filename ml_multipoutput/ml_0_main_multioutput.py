## Code reference: https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de

## Import packages
import pandas as pd
import numpy as np
import pickle as p
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold



## Import data
pv_train = pd.read_csv('YMCA_train.csv')
pv_test = pd.read_csv('YMCA_test.csv')
pv_test2 = pd.read_csv('FR_test.csv')

input_cols = ['TempOut', 'OutHum', 'WindSpeed', 'Bar', 'SolarRad']
output_cols = ['NRM_P_GEN_MIN', 'NRM_P_GEN_MAX']



# Bin separation
# Block (0-3) => BIN 1
cut_blocks = [i for i in range(1, 9)]

pv_train['Bin'] = pd.cut(pv_train['Block'], 8, labels=cut_blocks)
pv_train.style
pv_test['Bin'] = pd.cut(pv_test['Block'], 8, labels=cut_blocks)
pv_test.style
pv_test2['Bin'] = pd.cut(pv_test2['Block'], 8, labels=cut_blocks)
pv_test2.style



## Setup Pipeline
## Without hyperparameter optimisation
# Linear regression
model_ln = Pipeline([('lin_regression', MultiOutputRegressor(LinearRegression(), n_jobs=-1))])
# Tree regression
model_dt = Pipeline([('dt_regression', MultiOutputRegressor(DecisionTreeRegressor(random_state=0), n_jobs=-1))])
# Linear Support Vector regression
model_svr = Pipeline([('svm_regression', MultiOutputRegressor(LinearSVR(random_state=0), n_jobs=-1))])
# Ensemble regression
model_grb = Pipeline([('gradboost_regression', MultiOutputRegressor(GradientBoostingRegressor(random_state=0), n_jobs=-1))])
# ANN model
model_ann = Pipeline([('ann_sk', MLPRegressor(solver='adam', random_state=0,
                                              shuffle=True, early_stopping=True))])
models = [
    model_ln,
    model_dt,
    model_svr,
    model_grb,
    model_ann
]
reg_dict = {
    0: 'Linear',
    1: 'Decision Tree',
    2: 'SVM',
    3: 'Grad Boost',
    4: 'ANN'
}

## With hyperparameter optimisation
# Lasso regression
model_lasso_opt = Pipeline([('lasso_opt', MultiOutputRegressor(Lasso(random_state= 0,
                                                                     max_iter=500,
                                                                     selection='cyclic'),
                                                               n_jobs=-1))])
# ElasticNet regression
model_eln_opt = Pipeline([('eln_opt', MultiOutputRegressor(ElasticNet(random_state= 0,
                                                                      l1_ratio=0.2,
                                                                      max_iter=500,
                                                                      selection='cyclic'),
                                                           n_jobs=-1))])
# Tree regression
model_dt_opt = Pipeline([('dt_opt', MultiOutputRegressor(DecisionTreeRegressor(criterion='squared_error',
                                                                               min_samples_leaf=38,
                                                                               min_samples_split=2,
                                                                               splitter='random',
                                                                               random_state= 0),
                                                         n_jobs=-1))])
# Linear Support Vector regression
model_svr_opt = Pipeline([('svm_opt', MultiOutputRegressor(LinearSVR(random_state= 0,
                                                                     loss='squared_epsilon_insensitive',
                                                                     max_iter=1000),
                                                           n_jobs=-1))])
# Nu Support Vector regression
model_nusvr_opt = Pipeline([('nusvr_opt', MultiOutputRegressor(NuSVR(nu=0.45,
                                                                     kernel='rbf',
                                                                     gamma='scale',
                                                                     cache_size=200),
                                                               n_jobs=-1))])
# Ensemble regression
model_grb_opt = Pipeline([('gradboost_opt', MultiOutputRegressor(GradientBoostingRegressor(random_state=0,
                                                                                           loss='squared_error',
                                                                                           criterion='squared_error',
                                                                                           min_samples_split=2,
                                                                                           min_samples_leaf=14,
                                                                                           n_estimators=100),
                                                                 n_jobs=-1))])
# ANN model
model_ann_opt = Pipeline([('ann_sk_opt', MLPRegressor(hidden_layer_sizes=(100,100,), activation='logistic',
                                                      solver='adam', random_state=0, max_iter=200,
                                                      shuffle=True, early_stopping=True))])
models_opt = [
    model_ln,
    model_lasso_opt,
    model_eln_opt,
    model_dt_opt,
    model_svr_opt,
    model_nusvr_opt,
    model_grb_opt,
    model_ann_opt
]
reg_opt_dict = {
    0: 'Linear',
    1: 'Lasso',
    2: 'ElasticNet',
    3: 'Decision Tree',
    4: 'Linear SVM',
    5: 'Linear NuSVM',
    6: 'Grad Boost',
    7: 'ANN'
}



## Stratified k-Fold Cross Validation
def stratified_kfcv(input_data):
    import warnings
    warnings.filterwarnings("ignore")

    skf = StratifiedKFold(n_splits=10, random_state=23, shuffle=True)
    input_data['kfold'] = -1

    for fold,(train_indices, valid_indices) in enumerate(skf.split(X=input_data.iloc[:,:-1], y=input_data['Bin'])):
        input_data.loc[valid_indices, 'kfold'] = fold

    return input_data

pv_train = stratified_kfcv(pv_train)
pv_test = stratified_kfcv(pv_test)
pv_test2 = stratified_kfcv(pv_test2)



## Function for comparing ML models
def model_comparison(model_list, model_dict, data_train, data_test):
    best_rmse = 100.0
    best_rmse_pipeline = ""
    best_rmse_regressor = 0

    best_mae = 100.0
    best_mae_pipeline = ""
    best_mae_regressor = 0

    best_regressor = 0
    best_pipeline = ""

    for j, model in enumerate(model_list):
        print(model_dict[j])
        RMSE_train, MAE_train = list(), list()
        min_rmse, min_mae = 100.0, 100.0
        best_model = b''
        print('Training results')
        # RMSE_test, MAE_test = list(), list()
        for i in range(10):
            ## Training dataset
            # k fold block for training
            xtrain = data_train[data_train['kfold'] != i]
            # k-1 fold blocks for validation
            xvalid = data_train[data_train['kfold'] == i]

            # getting ML output values
            ytrain = xtrain[output_cols]
            yvalid = xvalid[output_cols]

            # getting ML input values
            xtrain = xtrain[input_cols]
            xvalid = xvalid[input_cols]

            # scaling ML input values
            scaler = StandardScaler()
            scaler.fit_transform(xtrain)
            scaler.transform(xvalid)

            # training model
            model.fit(xtrain, ytrain)
            ypred = model.predict(xvalid)

            # scoring using training dataset
            rmse = mean_squared_error(yvalid, ypred, squared=False)
            RMSE_train.append(rmse)
            mae = mean_absolute_error(yvalid, ypred)
            MAE_train.append(mae)
            print('Fold {}: {}, {}'.format(i+1, rmse, mae))

            # capture best model found during CV
            if min(RMSE_train) < min_rmse and min(MAE_train) < min_mae:
                min_rmse = min(RMSE_train)
                min_mae = min(MAE_train)
                best_model = p.dumps(model)
                print('New best model fit found: Fold {}'.format(i+1))

            if min(RMSE_train) < min_rmse:
                min_rmse = min(RMSE_train)
                # best_model = p.dumps(model)
                # print(i)

            if min(MAE_train) < min_mae:
                min_mae = min(MAE_train)
                # best_model = p.dumps(model)
                # print(i)

            """
            ## Testing dataset
            xtrain = data_test[data_test['kfold'] != i]
            xvalid = data_test[data_test['kfold'] == i]

            ytrain = xtrain[output_cols]
            yvalid = xvalid[output_cols]

            xtrain = xtrain[input_cols]
            xvalid = xvalid[input_cols]

            scaler = StandardScaler()
            scaler.fit_transform(xtrain)
            scaler.transform(xvalid)

            # predict testing dataset (no need training since done previously)
            ypred = model.predict(xvalid)
            rmse = mean_squared_error(yvalid, ypred, squared=False)
            RMSE_test.append(rmse)
            mae = mean_absolute_error(yvalid, ypred)
            MAE_test.append(mae)
            """


        ## Printing training and testing scoring for each model
        train_folds_mean_rmse = np.mean(RMSE_train)
        print('Mean Validation RMSE: {}'.format(train_folds_mean_rmse))
        train_folds_mean_mae = np.mean(MAE_train)
        print('Mean Validation MAE: {}'.format(train_folds_mean_mae))
        """
        print('Testing results')
        test_folds_mean_rmse = np.mean(RMSE_test)
        print('Mean Validation RMSE: {}'.format(test_folds_mean_rmse))
        test_folds_mean_mae = np.mean(MAE_test)
        print('Mean Validation MAE: {}\n'.format(test_folds_mean_mae))
        """


        ## Testing score
        x_test = data_test[input_cols]
        y_test = data_test[output_cols]

        best = p.loads(best_model)
        y_pred = best.predict(x_test)
        print('Test data scoring using best model fit')
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        print('RMSE: {}'.format(test_rmse))
        test_mae = mean_absolute_error(y_test, y_pred)
        print('MAE: {}\n'.format(test_mae))


        ## Determining best model according to testing scoring
        if test_rmse < best_rmse and test_mae < best_mae:
            best_rmse = test_rmse
            best_rmse_pipeline = model
            best_rmse_regressor = j
            
            best_mae = test_mae
            best_mae_pipeline = model
            best_mae_regressor = j

            best_pipeline = model
            best_regressor = j

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_rmse_pipeline = model
            best_rmse_regressor = j

        if test_mae < best_mae:
            best_mae = test_mae
            best_mae_pipeline = model
            best_mae_regressor = j

    print('-----------------------------------------------------')
    print('Regressor with least RMSE: {}'.format(model_dict[best_rmse_regressor]))
    print('Regressor with least MAE: {}'.format(model_dict[best_mae_regressor]))
    print('Best regressor: {}'.format(model_dict[best_regressor]))
    print(best_pipeline)
    print('-----------------------------------------------------')
    print('\n')

## Initial comparison
model_comparison(models, reg_dict, pv_train, pv_test)

## Optimised comparison
model_comparison(models_opt, reg_opt_dict, pv_train, pv_test)

## Predicting FR site using models trained on YMCA site
model_comparison(models_opt, reg_opt_dict, pv_train, pv_test2)
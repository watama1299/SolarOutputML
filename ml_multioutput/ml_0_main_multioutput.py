## Code reference:
## https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de
## https://www.geeksforgeeks.org/how-to-use-pickle-to-save-and-load-variables-in-python/
## https://datatofish.com/numpy-array-to-pandas-dataframe/

## Import packages
import pandas as pd
import numpy as np
import pickle as p
import warnings as w
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, MultiTaskLasso, MultiTaskElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR, NuSVR
from sklearn.neural_network import MLPRegressor



## Import data
pv_train = pd.read_csv('YMCA_data_train.csv')
pv_test = pd.read_csv('YMCA_data_test.csv')
pv_test2 = pd.read_csv('FR_data_test.csv')

input_cols = ['TempOut', 'OutHum', 'WindSpeed', 'Bar', 'SolarRad']
output_cols = ['NRM_P_GEN_MIN', 'NRM_P_GEN_MAX']



# Bin separation
# Block (1-3) => Bin 1, Block (4-6) => Bin 2, ... Block (22-24) => Bin 8
bin_nums = [i for i in range(1, 9)]

pv_train['Bin'] = pd.cut(pv_train['Block'], 8, labels=bin_nums)
pv_train.style
pv_test['Bin'] = pd.cut(pv_test['Block'], 8, labels=bin_nums)
pv_test.style
pv_test2['Bin'] = pd.cut(pv_test2['Block'], 8, labels=bin_nums)
pv_test2.style



## Stratified k-Fold Cross Validation
def stratified_kfcv(input_data):
    # Split data into 10 folds
    skf = StratifiedKFold(n_splits=10, random_state=23, shuffle=True)
    # Making new column for kfold labels
    input_data['kfold'] = -1

    # Splitting the 8 bins fairly into the each of the folds to get balanced representation
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
    best_model_dict = {}
    all_models = {}

    ## Go through the model_list and train and test each model
    for j, model in enumerate(model_list):
        print(model_dict[j])
        RMSE_train, MAE_train = list(), list()
        model_rmse, model_mae = list(), list()
        min_rmse, min_mae = 100.0, 100.0

        # variables to store/save/persist models
        best_model_overall = b''
        best_model_rmse = b''
        best_model_mae = b''
        # print('Training results: RMSE, MAE')
        for i in range(10):
            ## Training dataset
            # Blocks for training => all other folds
            xtrain = data_train[data_train['kfold'] != i]
            # Block for validation => kth fold
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
            rmse_cv = mean_squared_error(yvalid, ypred, squared=False)
            RMSE_train.append(rmse_cv)
            mae_cv = mean_absolute_error(yvalid, ypred)
            MAE_train.append(mae_cv)
            # print('Fold {}: {}, {}'.format(i+1, rmse_cv, mae_cv))

            # capture best model found during CV
            if rmse_cv < min_rmse and mae_cv < min_mae:
                # if everything improves, then update all models
                min_rmse = rmse_cv
                best_model_rmse = p.dumps(model)
                min_mae = mae_cv
                best_model_mae = p.dumps(model)

                best_model_overall = p.dumps(model)
                print('New best overall model fit found: Fold {}'.format(i+1))

            elif rmse_cv < min_rmse:
                # if only rmse improve, then update rmse model
                min_rmse = rmse_cv
                best_model_rmse = p.dumps(model)
                print('New best RMSE model fit found: Fold {}'.format(i+1))
                # print(i)

            elif mae_cv < min_mae:
                # if only mae improve, then update mae model
                min_mae = mae_cv
                best_model_mae = p.dumps(model)
                print('New best MAE model fit found: Fold {}'.format(i+1))
                # print(i)
            


        ## Printing training and testing scoring for each model
        train_folds_mean_rmse = np.mean(RMSE_train)
        print('Mean Validation RMSE: {}'.format(train_folds_mean_rmse))
        train_folds_mean_mae = np.mean(MAE_train)
        print('Mean Validation MAE: {}'.format(train_folds_mean_mae))
        


        ## Testing score
        x_train = data_train[input_cols]
        y_train = data_train[output_cols]
        x_test = data_test[input_cols]
        y_test = data_test[output_cols]

        # Scoring of normal model
        normal = model.fit(x_train, y_train)
        yn_pred = normal.predict(x_test)
        print('\nTest data scoring using normal model')
        yn_rmse = mean_squared_error(y_test, yn_pred, squared=False)
        model_rmse.append(yn_rmse)
        print('RMSE: {}'.format(yn_rmse))
        yn_mae = mean_absolute_error(y_test, yn_pred)
        model_mae.append(yn_mae)
        print('MAE: {}'.format(yn_mae))

        # Scoring using best overall model from CV
        overall_best = p.loads(best_model_overall)
        ybo_pred = overall_best.predict(x_test)
        print('Test data scoring using best fit model from CV')
        ybo_rmse = mean_squared_error(y_test, ybo_pred, squared=False)
        model_rmse.append(ybo_rmse)
        print('RMSE: {}'.format(ybo_rmse))
        ybo_mae = mean_absolute_error(y_test, ybo_pred)
        model_mae.append(ybo_mae)
        print('MAE: {}'.format(ybo_mae))

        # Scoring using best RMSE model from CV
        rmse_best = p.loads(best_model_rmse)
        ybr_pred = rmse_best.predict(x_test)
        print('Test data scoring using best RMSE model from CV')
        ybr_rmse = mean_squared_error(y_test, ybr_pred, squared=False)
        model_rmse.append(ybr_rmse)
        print('RMSE: {}'.format(ybr_rmse))
        ybr_mae = mean_absolute_error(y_test, ybr_pred)
        model_mae.append(ybr_mae)
        print('MAE: {}'.format(ybr_mae))

        # Scoring using best MAE model from CV
        mae_best = p.loads(best_model_mae)
        ybm_pred = mae_best.predict(x_test)
        print('Test data scoring using best MAE model from CV')
        ybm_rmse = mean_squared_error(y_test, ybm_pred, squared=False)
        model_rmse.append(ybm_rmse)
        print('RMSE: {}'.format(ybm_rmse))
        ybm_mae = mean_absolute_error(y_test, ybm_pred)
        model_mae.append(ybm_mae)
        print('MAE: {}\n\n'.format(ybm_mae))



        ## Save the 4 trained models for output
        model_trained = {
            'name': model_dict[j],
            'normal': p.dumps(normal),
            'overall_best': p.dumps(overall_best),
            'rmse_best': p.dumps(rmse_best),
            'mae_best': p.dumps(mae_best)
        }



        ## Determining best model according to testing scoring
        # test_rmse = min(model_rmse)
        # test_mae = min(model_mae)
        test_rmse = yn_rmse
        test_mae = yn_mae

        if test_rmse < best_rmse and test_mae < best_mae:
            best_rmse = test_rmse
            best_rmse_pipeline = model
            best_rmse_regressor = j
            
            best_mae = test_mae
            best_mae_pipeline = model
            best_mae_regressor = j

            best_pipeline = model
            best_regressor = j
            best_model_dict = model_trained

        elif test_rmse < best_rmse:
            best_rmse = test_rmse
            best_rmse_pipeline = model
            best_rmse_regressor = j

        elif test_mae < best_mae:
            best_mae = test_mae
            best_mae_pipeline = model
            best_mae_regressor = j

        ## Add trained model to list of all models trained
        # key: model, value: model_trained
        all_models[model_dict[j]] = model_trained

    print('-----------------------------------------------------')
    print('Best regressor: {}'.format(model_dict[best_regressor]))
    print('Regressor with best RMSE: {}'.format(model_dict[best_rmse_regressor]))
    print('Regressor with best MAE: {}'.format(model_dict[best_mae_regressor]))
    # print(best_pipeline)
    print('-----------------------------------------------------')
    print('\n\n\n')

    # return best_model_dict
    return all_models



## Setup Pipeline
## Without hyperparameter optimisation
# Linear regression
model_ln = MultiOutputRegressor(LinearRegression(), n_jobs=-1)
# Tree regression
model_dt = MultiOutputRegressor(DecisionTreeRegressor(random_state=0),
                                n_jobs=-1)
# Linear Support Vector regression
model_svr = MultiOutputRegressor(LinearSVR(random_state=0),
                                 n_jobs=-1)
# Ensemble regression
model_grb = MultiOutputRegressor(GradientBoostingRegressor(random_state=0),
                                 n_jobs=-1)
# ANN model
model_ann = MLPRegressor(solver='adam', random_state=0, 
                         shuffle=True, early_stopping=True)
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
    2: 'Linear SVM',
    3: 'Gradient Boost',
    4: 'ANN'
}

## With hyperparameter optimisation
# Lasso regression
model_lasso_opt = MultiOutputRegressor(Lasso(max_iter=500,
                                            random_state=0,
                                            selection='cyclic'),
                                       n_jobs=-1)
model_mlasso_opt = MultiTaskLasso(max_iter=500,
                                  random_state=0,
                                  selection='cyclic')
# ElasticNet regression
model_eln_opt = MultiOutputRegressor(ElasticNet(l1_ratio=0.2,
                                                max_iter=500,
                                                random_state=0,
                                                selection='cyclic'),
                                     n_jobs=-1)
model_meln_opt = MultiTaskElasticNet(l1_ratio=0.23,
                                     max_iter=500,
                                     random_state=0,
                                     selection='cyclic')
# Tree regression
model_dt_opt = MultiOutputRegressor(DecisionTreeRegressor(criterion='squared_error',
                                                          splitter='random',
                                                          min_samples_split=2,
                                                          min_samples_leaf=38,
                                                          max_features=None,
                                                          random_state=0),
                                    n_jobs=-1)
# Linear Support Vector regression
model_svr_opt = MultiOutputRegressor(LinearSVR(loss='squared_epsilon_insensitive',
                                               dual=False,
                                               random_state=0,
                                               max_iter=1000),
                                     n_jobs=-1)
# Nu Support Vector regression
model_nusvr_opt = MultiOutputRegressor(NuSVR(nu=0.42,
                                             kernel='poly',
                                             gamma='scale',
                                             cache_size=1000,
                                             max_iter=-1),
                                       n_jobs=-1)
# Ensemble regression
model_grb_opt = MultiOutputRegressor(GradientBoostingRegressor(loss='squared_error',
                                                               n_estimators=100,
                                                               criterion='friedman_mse',
                                                               min_samples_split=2,
                                                               min_samples_leaf=190,
                                                               max_depth=None,
                                                               random_state=0,
                                                               max_features=None,
                                                               n_iter_no_change=10),
                                     n_jobs=-1)
# ANN model
model_ann_opt = MLPRegressor(hidden_layer_sizes=(100,100,),
                             activation='logistic',
                             solver='adam',
                             max_iter=200,
                             shuffle=True,
                             random_state=0,
                             early_stopping=True)
models_opt = [
    model_ln,
    model_lasso_opt,
    model_mlasso_opt,
    model_eln_opt,
    model_meln_opt,
    model_dt_opt,
    model_svr_opt,
    model_nusvr_opt,
    model_grb_opt,
    model_ann_opt
]
reg_opt_dict = {
    0: 'Linear',
    1: 'Lasso',
    2: 'MultiLasso',
    3: 'ElasticNet',
    4: 'MultiElasticNet',
    5: 'Decision Tree',
    6: 'Linear SVM',
    7: 'NuSVM',
    8: 'Gradient Boost',
    9: 'ANN'
}



w.filterwarnings('ignore')
## Initial comparison
print('-----------------------------------------------------')
print('Initial comparison, tested on YMCA testing dataset')
print('-----------------------------------------------------')
out_models_init = model_comparison(models, reg_dict, pv_train, pv_test)
# for trained_models in out_models_init.values():
#     best_model_init = p.loads(trained_models.get('normal'))
#     y_init = best_model_init.predict(pv_test[input_cols])
#     y_init = pd.DataFrame(y_init, columns=['PRED_NRM_P_GEN_MIN', 'PRED_NRM_P_GEN_MAX'])
#     # y_init.to_csv('y_init_mo_{}.csv'.format(trained_models.get('name')), index=False)
#     # print('Model type {} has successfully exported its initial output'.format(trained_models.get('name')))

## Optimised comparison
print('-----------------------------------------------------')
print('Optimised comparison, tested on YMCA testing dataset')
print('-----------------------------------------------------')
out_models_opt = model_comparison(models_opt, reg_opt_dict, pv_train, pv_test)
# for trained_models in out_models_opt.values():
#     best_model_opt = p.loads(trained_models.get('normal'))
#     y_opt = best_model_opt.predict(pv_test[input_cols])
#     y_opt = pd.DataFrame(y_opt, columns=['PRED_NRM_P_GEN_MIN', 'PRED_NRM_P_GEN_MAX'])
#     # y_opt.to_csv('y_opt_mo_{}.csv'.format(trained_models.get('name')), index=False)
#     # print('Model type {} has successfully exported its optimised output'.format(trained_models.get('name')))

## Predicting FR site using models trained on YMCA site
print('-----------------------------------------------------')
print('Optimised models, tested on FR testing dataset')
print('-----------------------------------------------------')
out_models_fr = model_comparison(models_opt, reg_opt_dict, pv_train, pv_test2)
# for trained_models in out_models_fr.values():
#     best_model_fr = p.loads(trained_models.get('normal'))
#     y_fr = best_model_fr.predict(pv_test2[input_cols])
#     y_fr = pd.DataFrame(y_fr, columns=['PRED_NRM_P_GEN_MIN', 'PRED_NRM_P_GEN_MAX'])
#     # y_fr.to_csv('y_fr_mo_{}.csv'.format(trained_models.get('name')), index=False)
#     # print('Model type {} has successfully exported its Forest Road output'.format(trained_models.get('name')))
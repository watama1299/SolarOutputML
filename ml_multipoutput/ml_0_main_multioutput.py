## Code reference:
## https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de
## https://www.geeksforgeeks.org/how-to-use-pickle-to-save-and-load-variables-in-python/
## https://datatofish.com/numpy-array-to-pandas-dataframe/

## Import packages
import pandas as pd
import numpy as np
import pickle as p
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
pv_train = pd.read_csv('YMCA_train.csv')
pv_test = pd.read_csv('YMCA_test.csv')
pv_test2 = pd.read_csv('FR_test.csv')

input_cols = ['TempOut', 'OutHum', 'WindSpeed', 'Bar', 'SolarRad']
output_cols = ['NRM_P_GEN_MIN', 'NRM_P_GEN_MAX']



# Bin separation
# Block (0-8) => BIN 1
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
    3: 'Gradient Boost Decision Tree',
    4: 'ANN'
}

## With hyperparameter optimisation
# Lasso regression
model_lasso_opt = Pipeline([('lasso_opt', MultiOutputRegressor(Lasso(max_iter=500,
                                                                     random_state=0,
                                                                     selection='cyclic'),
                                                               n_jobs=-1))])
model_mlasso_opt = Pipeline([('mlasso_opt', MultiTaskLasso(max_iter=500,
                                                           random_state=0,
                                                           selection='cyclic'))])
# ElasticNet regression
model_eln_opt = Pipeline([('eln_opt', MultiOutputRegressor(ElasticNet(l1_ratio=0.2,
                                                                      max_iter=500,
                                                                      random_state=0,
                                                                      selection='cyclic'),
                                                           n_jobs=-1))])
model_meln_opt = Pipeline([('meln_opt', MultiTaskElasticNet(l1_ratio=0.23,
                                                            max_iter=500,
                                                            random_state=0,
                                                            selection='cyclic'))])
# Tree regression
model_dt_opt = Pipeline([('dt_opt', MultiOutputRegressor(DecisionTreeRegressor(criterion='squared_error',
                                                                               splitter='random',
                                                                               min_samples_split=2,
                                                                               min_samples_leaf=38,
                                                                               max_features='auto',
                                                                               random_state=0),
                                                         n_jobs=-1))])
# Linear Support Vector regression
model_svr_opt = Pipeline([('svm_opt', MultiOutputRegressor(LinearSVR(loss='squared_epsilon_insensitive',
                                                                     dual=False,
                                                                     random_state=0,
                                                                     max_iter=1000),
                                                           n_jobs=-1))])
# Nu Support Vector regression
model_nusvr_opt = Pipeline([('nusvr_opt', MultiOutputRegressor(NuSVR(nu=0.5,
                                                                     kernel='rbf',
                                                                     gamma='scale',
                                                                     cache_size=1000,
                                                                     max_iter=-1),
                                                               n_jobs=-1))])
# Ensemble regression
model_grb_opt = Pipeline([('gradboost_opt', MultiOutputRegressor(GradientBoostingRegressor(loss='squared_error',
                                                                                           n_estimators=100,
                                                                                           criterion='friedman_mse',
                                                                                           min_samples_split=2,
                                                                                           min_samples_leaf=100,
                                                                                           max_depth=None,
                                                                                           random_state=0,
                                                                                           max_features='sqrt',
                                                                                           n_iter_no_change=10),
                                                                 n_jobs=-1))])
# ANN model
model_ann_opt = Pipeline([('ann_sk_opt', MLPRegressor(hidden_layer_sizes=(100,100,),
                                                      activation='logistic',
                                                      solver='adam',
                                                      max_iter=200,
                                                      shuffle=True,
                                                      random_state=0,
                                                      early_stopping=True))])
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
    1: 'Linear - Lasso',
    2: 'Linear - MultiLasso',
    3: 'Linear - ElasticNet',
    4: 'Linear - MultiElasticNet',
    5: 'Decision Tree',
    6: 'Linear SVM',
    7: 'NuSVM',
    8: 'Gradient Boost Decision Tree',
    9: 'ANN'
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
        print('Training results: RMSE, MAE')
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
            rmse_cv = mean_squared_error(yvalid, ypred, squared=False)
            RMSE_train.append(rmse_cv)
            mae_cv = mean_absolute_error(yvalid, ypred)
            MAE_train.append(mae_cv)
            print('Fold {}: {}, {}'.format(i+1, rmse_cv, mae_cv))

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
        test_rmse = min(model_rmse)
        test_mae = min(model_mae)

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
        all_models[model_dict[j]] = model_trained

    print('-----------------------------------------------------')
    print('Regressor with least RMSE: {}'.format(model_dict[best_rmse_regressor]))
    print('Regressor with least MAE: {}'.format(model_dict[best_mae_regressor]))
    print('Best regressor: {}'.format(model_dict[best_regressor]))
    # print(best_pipeline)
    print('-----------------------------------------------------')
    print('\n\n\n')

    # return best_model_dict
    return all_models



## Initial comparison
print('-----------------------------------------------------')
print('Initial comparison')
print('-----------------------------------------------------')
out_models_init = model_comparison(models, reg_dict, pv_train, pv_test)
best_model_init = p.loads(out_models_init.get('Gradient Boost Decision Tree').get('normal'))
# best_model_init = p.loads(out_models_init.get('Gradient Boost Decision Tree').get('normal'))
# best_model_init = p.loads(out_models_init.get('Gradient Boost Decision Tree').get('normal'))
y_init = best_model_init.predict(pv_test[input_cols])
y_init = pd.DataFrame(y_init, columns=['PRED_NRM_P_GEN_MIN', 'PRED_NRM_P_GEN_MAX'])
# y_init.to_csv('y_init_mo.csv', index=False)

## Optimised comparison
print('-----------------------------------------------------')
print('Optimised comparison')
print('-----------------------------------------------------')
out_models_opt = model_comparison(models_opt, reg_opt_dict, pv_train, pv_test)
best_model_opt = p.loads(out_models_opt.get('Decision Tree').get('normal'))
# best_model_opt = p.loads(out_models_opt.get('Decision Tree').get('normal'))
# best_model_opt = p.loads(out_models_opt.get('Decision Tree').get('normal'))
y_opt = best_model_opt.predict(pv_test[input_cols])
y_opt = pd.DataFrame(y_opt, columns=['PRED_NRM_P_GEN_MIN', 'PRED_NRM_P_GEN_MAX'])
# y_opt.to_csv('y_opt_mo.csv', index=False)

## Predicting FR site using models trained on YMCA site
print('-----------------------------------------------------')
print('Optimised models trained on YMCA, tested on FR')
print('-----------------------------------------------------')
out_models_fr = model_comparison(models_opt, reg_opt_dict, pv_train, pv_test2)
best_model_fr = p.loads(out_models_fr.get('Linear').get('normal'))
# best_model_fr = p.loads(out_models_fr.get('Linear').get('normal'))
# best_model_fr = p.loads(out_models_fr.get('Linear').get('normal'))
y_fr = best_model_fr.predict(pv_test2[input_cols])
y_fr = pd.DataFrame(y_fr, columns=['PRED_NRM_P_GEN_MIN', 'PRED_NRM_P_GEN_MAX'])
# y_fr.to_csv('y_fr_mo.csv', index=False)
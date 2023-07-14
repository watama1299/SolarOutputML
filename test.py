## Code reference: https://medium.com/@akashprabhakar427/solar-power-forecasting-using-machine-learning-and-deep-learning-61d6292693de

## Import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import zipfile


## Read csv files
pv_combined = pd.read_csv('YMCA_combined.csv')
pv_train = pd.read_csv('YMCA_train.csv')
pv_test = pd.read_csv('YMCA_test.csv')
# print(pv_train.describe())
# print(pv_test.describe())


# ## Data analysis
# ## Using scatter plot and regression line to find correlation
# ## Regression line reference: https://towardsdatascience.com/seaborn-pairplot-enhance-your-data-understanding-with-a-single-plot-bf2f44524b22
# p1 = sns.pairplot(pv_combined,
#                   y_vars=['TempOut','OutHum','DewPt'], 
#                   x_vars=['P_GEN_MIN','P_GEN_MAX'], 
#                   kind='reg',
#                   plot_kws={'line_kws':{'color':'red'}})

# p2 = sns.pairplot(pv_combined,
#                   y_vars=['WindSpeed','WindRun','WindChill'], 
#                   x_vars=['P_GEN_MIN','P_GEN_MAX'],  
#                   kind='reg',
#                   plot_kws={'line_kws':{'color':'red'}})

# p3 = sns.pairplot(pv_combined,
#                   y_vars=['HeatIndex','THWIndex'], 
#                   x_vars=['P_GEN_MIN','P_GEN_MAX'],  
#                   kind='reg',
#                   plot_kws={'line_kws':{'color':'red'}})

# p4 = sns.pairplot(pv_combined,
#                   y_vars=['Bar','Rain','RainRate'], 
#                   x_vars=['P_GEN_MIN','P_GEN_MAX'],  
#                   kind='reg',
#                   plot_kws={'line_kws':{'color':'red'}})

# p5 = sns.pairplot(pv_combined,
#                   y_vars=['SolarRad','SolarEnergy','HiSolarRad'], 
#                   x_vars=['P_GEN_MIN','P_GEN_MAX'],  
#                   kind='reg',
#                   plot_kws={'line_kws':{'color':'red'}})
# plt.show()

# ## Using heat map to find correlation
# h1 = sns.heatmap(pv_combined.corr(), annot=True)
# plt.show()


## Handling outlier
percentile_dict = {}
for i in pv_train.columns[3:7]:
    a_list = []
    for j in [1, 10, 25, 50, 75, 90, 99, 100]:
        a_list.append(round(np.percentile(pv_train[i], j), 2))
    percentile_dict[i] = a_list
outlier = pd.DataFrame(pd.concat([pd.DataFrame({'Percentiles':[1,10,25,50,75,90,99,100]}),pd.DataFrame(percentile_dict)],axis=1))
outlier.style

outlier_imputer_dict = {}

for var in pv_train.columns[3:]:
    percentile_dict = {}
    
    NinetyNine_percentile = np.percentile(pv_train[var],99)  
       
    First_percentile = np.percentile(pv_train[var],1)

    percentile_dict['99th'] =  NinetyNine_percentile
    percentile_dict['1st'] =  First_percentile  
    # Saving as dictionary for each column
    outlier_imputer_dict[var] = percentile_dict
      
#Saving the final dictionary         
np.save('outlier_imputer_dict',outlier_imputer_dict)    
def outlier_imputer(df):
    #Loading Outlier Imputer dictionary
    outlier_dict = np.load('outlier_imputer_dict.npy',allow_pickle='TRUE').item()
    
    for var in df.columns[3:]:
        
        df.loc[df[df[var] > outlier_dict[var]['99th']].index,var] = outlier_dict[var]['99th']  
       
        df.loc[df[df[var] < outlier_dict[var]['1st']].index,var] = outlier_dict[var]['1st']
    
    return df

print(outlier_imputer_dict)

#Applying imputation on Train & Test 
pv_train_out = outlier_imputer(pv_train)
pv_test_out = outlier_imputer(pv_test)
# pv_train_out.style
# pv_test_out.style


## Bin separation
## Block (0-24) => BIN 1
cut_blocks = [i for i in range(1, 9)]

pv_train_out['Bin'] = pd.cut(pv_train_out['Block'], 8, labels = cut_blocks)
pv_train_out.style



## ANN Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

def ANN():
    # 3 layer NN, 500 epochs, 4 batch size
    reg = Sequential()
    reg.add(Dense(16, input_dim = 5, activation='relu'))
    reg.add(Dense(8, kernel_initializer='normal', activation='relu'))
    reg.add(Dense(1))
    reg.compile(loss='mean_squared_error', optimizer='adam')
    return reg

ann_reg = KerasRegressor(build_fn=ANN, nb_epoch=500, batch_size=4, verbose=False)




## Regressors imports
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import *
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

## Regressor Pipelines
# from linear_model
pl_lr = Pipeline([('lin_regression', LinearRegression())])
pl_ridge = Pipeline([('ridge_regressor', Ridge(random_state=0))])
pl_lnet = Pipeline([('elasticnet_regression', ElasticNet(random_state=0))])
pl_lars = Pipeline([('lars_regression', Lars(random_state=0))])
pl_lasso = Pipeline([('lasso_regression', Lasso(random_state=0))])
pl_lasso_lars = Pipeline([('lasso_lars_regression', LassoLars(random_state=0))])
# from tree
pl_dt = Pipeline([('dt_regression', DecisionTreeRegressor(random_state=0))])
# from ensemble
pl_ada = Pipeline([('adaboost_regression', AdaBoostRegressor(random_state=0))])
pl_bag = Pipeline([('bagging_regression', BaggingRegressor(random_state=0))])
pl_grad = Pipeline([('gradboost_regression', GradientBoostingRegressor(random_state=0))])
pl_grad_optimised = Pipeline([('gradboost_optimised', GradientBoostingRegressor(n_estimators=1577, min_samples_split=10, min_samples_leaf=1,
                                                                                max_features='sqrt', max_depth=888, criterion='squared_error'))])
pl_rf = Pipeline([('rf_regression', RandomForestRegressor(random_state=0))])
# from xgboost
pl_xgb = Pipeline([('xgboost_regression', XGBRegressor())])
# from neural_network
pl_ann_sk = Pipeline([('mlp_regressor', MLPRegressor())])
# created ann model
pl_ann_keras = Pipeline([('ann_regressor', ann_reg)])

## List of pipelines
# pipelines = [pl_lr, pl_dt, pl_rf, pl_ridge, pl_lasso, pl_xgb, pl_ann,
#              pl_lnet]
pipelines = [pl_lr, pl_ridge, pl_lnet, pl_lars, pl_lasso, pl_lasso_lars, 
             pl_dt, 
             pl_ada, pl_bag, pl_grad, pl_grad_optimised, pl_rf, 
             pl_xgb, 
             pl_ann_sk,
             pl_ann_keras]

best_rmse = 100.0
best_regressor = 0
best_pipeline = ""

## Dictionary of pipelines
# pl_dict = {0:'Linear Regression',1: 'Decision Tree Regressor',2:'Random Forest Regressor',
#              3:'Ridge Regressor',4:'Lasso Regressor',5:'XG Boost Regressor',6:'ANN Regressor',}
pl_dict = {0: 'Linear Regression', 
           1: 'Ridge Regressor', 
           2: 'ElasticNet Regression',
           3: 'Lars Regressor',
           4: 'Lasso Regressor',
           5: 'Lasso-Lars Regressor',
           6: 'Decision Tree Regressor',
           7: 'AdaBoost Regressor',
           8: 'Bagging Regressor',
           9: 'Gradient Boosting Regressor',
           10: 'Gradient Boosting Regressor (Optimised)',
           11: 'Random Forest Regressor',
           12: 'XG Boost Regressor',
           13: 'ANN Regressor (Sci-kit)',
           14: 'ANN Regressor (Keras)'}





## k-Fold Cross Validation
import warnings
warnings.filterwarnings("ignore")

useful_cols = ['TempOut', 'OutHum', 'WindSpeed', 'Bar', 'SolarRad']

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=8, random_state=23, shuffle=True)
pv_train_out['kfold'] = -1

for fold,(train_indices, valid_indices) in enumerate(skf.split(X=pv_train_out.iloc[:,:-1], y=pv_train_out['Bin'])):
    pv_train_out.loc[valid_indices, 'kfold'] = fold

for j, model in enumerate(pipelines):
    print(pl_dict[j])
    RMSE = list()
    for i in range(8):
        # copy training data that doesnt match the fold value into xtrain
        xtrain = pv_train_out[pv_train_out['kfold'] != i]
        # copy validation data that match the fold value into xvalid
        xvalid = pv_train_out[pv_train_out['kfold'] == i]

        ytrain = xtrain.P_GEN_MAX
        yvalid = xvalid.P_GEN_MAX

        xtrain = xtrain[useful_cols]
        xvalid = xvalid[useful_cols]

        scaler = StandardScaler()
        scaler.fit_transform(xtrain)
        scaler.transform(xvalid)

        model.fit(xtrain, ytrain)
        rmse = np.sqrt(mean_squared_error(yvalid, model.predict(xvalid)))
        RMSE.append(rmse)


    folds_mean_rmse = np.mean(RMSE)
    print('Mean Validation RMSE: {}\n'.format(folds_mean_rmse))


    if folds_mean_rmse < best_rmse:
        best_rmse = folds_mean_rmse
        best_pipeline = model
        best_regressor = j


print('\n\nRegressor with least RMSE: {}'.format(pl_dict[best_regressor]))
print(best_pipeline)



## Hyperparameter Optimisation
x_train = pv_train_out[useful_cols]
y_train = pv_train_out.P_GEN_MAX

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
max_features = ['None', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(0, 1000, 10)]
min_samples_split = [2, 5, 10, 14]
min_samples_leaf = [1, 2, 4, 6, 8]
criterion = ['friedman_mse', 'squared_error']
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'criterion': criterion}
gboost_rcv = RandomizedSearchCV(estimator=GradientBoostingRegressor(),
                                param_distributions=random_grid,
                                n_iter=100,
                                verbose=2,
                                random_state=30,
                                n_jobs=-1)
gboost_rcv.fit(x_train, y_train)

print(gboost_rcv.best_params_)
gboost_model = GradientBoostingRegressor(n_estimators=1577, min_samples_split=10, min_samples_leaf=1,
                                         max_features='sqrt', max_depth=888, criterion='squared_error')

gboost_model.fit(x_train, y_train)





## Predicting
x_test = pv_test_out[useful_cols]
y_test = pv_test_out.P_GEN_MAX

y_pred_gbo = gboost_model.predict(x_test)
gb_model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
y_pred_gb = gb_model.predict(x_test)

print(f'Root Mean Squared Error for Test Data (Gradient Boosting Optimised): {np.sqrt(mean_squared_error(y_test, y_pred_gbo))}')
print(f'Root Mean Squared Error for Test Data (Gradient Boosting): {np.sqrt(mean_squared_error(y_test, y_pred_gb))}')


from tensorflow import keras as tfk
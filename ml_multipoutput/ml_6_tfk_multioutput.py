## Import packages
import pandas as pd
import numpy as np
# Scikit learn
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
# Tensorflow/Keras
import tensorflow as tf
tfk = tf.keras
tfk_models = tfk.models
tfk_layers = tfk.layers
tfk_callbacks = tfk.callbacks
from scikeras.wrappers import KerasRegressor



## Import data
pv_train = pd.read_csv('YMCA_train.csv')
pv_test = pd.read_csv('YMCA_test.csv')



## Setup Pipeline
# Scikit Learn MLP (BPNN)
ann_sk = MLPRegressor(hidden_layer_sizes=(3), early_stopping=True, solver='lbfgs', verbose=False)
# Keras NN
def ANN_tfk():
    callback = tfk_callbacks.EarlyStopping(monitor='loss', patience=3)
    reg = tfk_models.Sequential()
    reg.add(tfk_layers.Dense(16, input_dim=5, activation='relu'))
    reg.add(tfk_layers.Dense(8, kernel_initializer='normal', activation='relu'))
    reg.add(tfk_layers.Dense(1))
    reg.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return reg

# ann_reg = krsk(build_fn=ANN, nb_epochs=500, batch_size=4, verbose=False)
ann_krs = KerasRegressor(model=ANN_tfk, epochs=100, batch_size=10, verbose=0)



## Initial comparison



## Hyperparameter optimisation



## Train with new hyperparameter



## Predict
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import *

matplotlib.use('Agg')


def MAPE(test, pred) :
    mape = round(np.mean(np.abs((test - pred) / test)) * 100, 2)
    
    return mape


def RMSE(test, pred) :
    rmse = round(np.sqrt(mean_squared_error(test, pred)), 2)
    
    return rmse


def MAE(test, pred) :
    mae = round(mean_absolute_error(test, pred), 2)
    
    return mae


def DrawGraph(history, test, pred, dataName, startTime) :
    loss = history.history['loss']
    valLoss = history.history['val_loss']
    mae = history.history['mae']
    valMae = history.history['val_mae']
    epochs = range(1, len(loss) + 1)
    error = test - pred
    
    _, axes = plt.subplots(3, 2, figsize = (24, 16))
    
    axes[0, 0].plot(epochs, loss, label = 'Training Loss')
    axes[0, 0].plot(epochs, valLoss, label = 'Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs, mae, label = 'Training MAE')
    axes[0, 1].plot(epochs, valMae, label = 'Validation MAE')
    axes[0, 1].set_title('Training and Validation MAE')
    axes[0, 1].legend()
    
    axes[1, 0].scatter(test, pred)
    axes[1, 0].set_title('Actual and Predicted TurnAroundTime')
    axes[1, 0].set_xlabel('Actual TurnAroundTime')
    axes[1, 0].set_ylabel('Predicted TurnAroundTime')

    axes[1, 1].plot(test, label='Actual')
    axes[1, 1].plot(pred, label='Prediction')
    axes[1, 1].set_title('Actual and Predicted TurnAroundTime')
    axes[1, 1].legend()
    
    axes[2, 0].hist(error)
    axes[2, 0].set_xlabel('Prediction Error')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Distribution of Prediction Error')
    
    plt.tight_layout()
    
    assert dataName in ['In', 'Out']
    
    mask = '%Y%m%d_%H%M%S'
    dte = time.strftime(mask, time.localtime(startTime))
    gname = '-{}.png'.format(dte)
    
    if dataName == 'In' :
        plt.savefig(graphInPath + gname)
    elif dataName == 'Out' :
        plt.savefig(graphOutPath + gname)
        

def SaveResult(name, result, model, saveName) :
    if name == 'In':
        if not os.path.exists(resultInPath) :
            result.to_csv(resultInPath, mode = 'w', header = True, index = False)
        else : 
            result.to_csv(resultInPath, mode = 'a', header = False, index = False)
            
        model.save(saveInPath + saveName)
            
    elif name == 'Out':
        if not os.path.exists(resultOutPath) :
            result.to_csv(resultOutPath, mode = 'w', header = True, index = False)
        else : 
            result.to_csv(resultOutPath, mode = 'a', header = False, index = False)
        
        model.save(saveOutPath + saveName)
            
            
def LearnRateScheduler(initLr, decayStep, decayRate, stairCase = True) :
    learnRate = tf.keras.optimizers.schedules.ExponentialDecay(initLr,
                                                               decay_steps = decayStep,
                                                               decay_rate = decayRate,
                                                               staircase = stairCase)
    
    return learnRate

graphInPath = 'result/graph/Graph - In'
graphOutPath = 'result/graph/Graph - Out'
dataInPath = 'data/Feature Window 3 In.csv'
dataOutPath = 'data/Feature Window 3 Out.csv'
rnnInPath = 'data/RNN IN Prediction.csv'
rnnOutPath = 'data/RNN OUT Prediction.csv'
lstmInPath = 'data/LSTM IN Prediction.csv'
lstmOutPath = 'data/LSTM OUT Prediction.csv'
gruInPath = 'data/GRU IN Prediction.csv'
gruOutPath = 'data/GRU OUT Prediction.csv'
saveInPath = 'result/model/Model - In'
saveOutPath = 'result/model/Model - Out'
resultInPath = 'result/result/MetricsResult - In.csv'
resultOutPath = 'result/result/MetricsResult - Out.csv'
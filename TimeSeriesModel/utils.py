import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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


def DrawGraph(history, test, pred, networkName, dataName, startTime) :
    loss = history.history['loss']
    valLoss = history.history['val_loss']
    mae = history.history['mae']
    valMae = history.history['val_mae']
    epochs = range(1, len(loss) + 1)
    
    _, axes = plt.subplots(2, 2, figsize = (24, 16))
    
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
    
    plt.tight_layout()
    
    assert networkName in ['RNN', 'LSTM', 'GRU']
    assert dataName in ['In', 'Out']
    
    mask = '%Y%m%d_%H%M%S'
    dte = time.strftime(mask, time.localtime(startTime))
    gname = '-{}.png'.format(dte)
    
    if networkName == 'RNN' :
        if dataName == 'In' :
            plt.savefig(graphRnnInPath + gname)
        elif dataName == 'Out' :
            plt.savefig(graphRnnOutPath + gname)
            
    elif networkName == 'LSTM' :
        if dataName == 'In' :
            plt.savefig(graphLstmInPath + gname)
        elif dataName == 'Out' :
            plt.savefig(graphLstmOutPath + gname)
            
    else :
        if dataName == 'In' :
            plt.savefig(graphGruInPath + gname)
        elif dataName == 'Out' :
            plt.savefig(graphGruOutPath + gname)
            
            
graphRnnInPath = 'result/graph/RNN-IN'
graphRnnOutPath = 'result/graph/RNN-OUT'
graphLstmInPath = 'result/graph/LSTM-IN'
graphLstmOutPath = 'result/graph/LSTM-OUT'
graphGruInPath = 'result/graph/GRU-IN'
graphGruOutPath = 'result/graph/GRU-OUT'
dataInPath = 'data/Feature In.csv'
dataOutPath = 'data/Feature Out.csv'
resultInPath = 'result/result/MetricsResult - In.csv'
resultOutPath = 'result/result/MetricsResult - Out.csv'
saveRnnInPath = 'result/model/RNN-IN'
saveRnnOutPath = 'result/model/RNN-OUT'
saveLstmInPath = 'result/model/LSTM-IN'
saveLstmOutPath = 'result/model/LSTM-OUT'
saveGruInPath = 'result/model/GRU-IN'
saveGruOutPath = 'result/model/GRU-OUT'
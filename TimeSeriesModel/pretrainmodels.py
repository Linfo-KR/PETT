import multiprocessing
from datetime import datetime as dte

import numpy as np
import pandas as pd
import tensorflow as tf
from dataLoader import *
from tensorflow.keras.models import *

maxCores = multiprocessing.cpu_count() 
print('\n\n Max CPU Cores : ', maxCores, '\n\n')


def LoadPretrainModel(networkName, dataName) :
    assert networkName in ['RNN', 'LSTM', 'GRU']
    assert dataName in ['In', 'Out']

    if dataName == 'In' :
        if networkName == 'RNN' :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-RNN IN.h5')
            timeStep = 23
        elif networkName == 'LSTM' :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-LSTM IN.h5')
            timeStep = 20
        else : 
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-GRU IN.h5')
            timeStep = 17
            
    else :
        if networkName == 'RNN' :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-RNN OUT.h5')
            timeStep = 14
        elif networkName == 'LSTM' :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-LSTM OUT.h5')
            timeStep = 12
        else : 
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-GRU OUT.h5')
            timeStep = 2
            
    return pretrainModel, timeStep


for network in ['RNN', 'LSTM', 'GRU'] :
    for name in ['In', 'Out'] :
        pretrainModel, timeStep = LoadPretrainModel(network, name)
        
        originData = LoadData(name)
        transformTrainX, transformTestX, trainY, testY, trainTimeIndex, testTimeIndex = processingData(originData, timeStep)
        
        predTrain = pretrainModel.predict(transformTrainX, verbose = 2, workers = maxCores, use_multiprocessing = True)
        predTest = pretrainModel.predict(transformTestX, verbose = 2, workers = maxCores, use_multiprocessing = True)
        pred = np.concatenate((predTrain, predTest), axis = 0)
        
        dateFormat = '%Y-%m-%d %H:%M'
        timeIndex = np.concatenate((trainTimeIndex, testTimeIndex), axis = 0)
        
        hourList = []
        dayList = []
        monthList = []
        yearList = []
        
        for index in timeIndex :
            hour = dte.strptime(index[0], dateFormat)
            hour = hour.strftime('%H')
            hourList.append(int(hour))
            
            day = dte.strptime(index[0], dateFormat)
            day = day.strftime('%d')
            dayList.append(int(day))
            
            month = dte.strptime(index[0], dateFormat)
            month = month.strftime('%m')
            monthList.append(int(month))
            
            year = dte.strptime(index[0], dateFormat)
            year = year.strftime('%Y')
            yearList.append(int(year))
        
        predValue = pd.DataFrame({'YEAR' : yearList, 'MONTH' : monthList, 'DAY' : dayList, 'HOUR' : hourList, 'PRED' : [value[0] for value in pred]})
        predValue['PRED'] = predValue['PRED'].apply(lambda x : round(x, 2))
                
        if name == 'In' :
            if network == 'RNN' :
                predValue.to_csv('result/prediction/RNN IN Prediction.csv', index = False)
            elif network == 'LSTM' :
                predValue.to_csv('result/prediction/LSMT IN Prediction.csv', index = False)
            else : 
                predValue.to_csv('result/prediction/GRU IN Prediction.csv', index = False)
            
        else :
            if network == 'RNN' :
                predValue.to_csv('result/prediction/RNN OUT Prediction.csv', index = False)
            elif network == 'LSTM' :
                predValue.to_csv('result/prediction/LSTM OUT Prediction.csv', index = False)
            else : 
                predValue.to_csv('result/prediction/GRU OUT Prediction.csv', index = False)
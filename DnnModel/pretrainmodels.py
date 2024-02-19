import multiprocessing

import tensorflow as tf
from tensorflow.keras.models import *

from dataloader import *

maxCores = multiprocessing.cpu_count()
print('\n\n Max CPU Cores : ', maxCores, '\n\n')


def LoadPretrainModel(dataName, modeName) :
    assert dataName in ['In', 'Out']
    assert modeName in ['With Hour', 'Without Hour']

    if dataName == 'In' :
        if modeName == 'With Hour' :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-WITH HOUR IN.h5')
        else :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-WITHOUT HOUR IN.h5')
            
    else :
        if modeName == 'With Hour' :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-WITH HOUR OUT.h5')
        else :
            pretrainModel = tf.keras.models.load_model('pretrain/pretrain-WITHOUT HOUR OUT.h5')

    return pretrainModel


for name in ['In', 'Out'] :
    originData, timeSeriesData = LoadData(name, 'RNN')
    originData = MergeTimeSeriesData(originData, timeSeriesData)
    
    for mode in ['With Hour', 'Without Hour'] :
        trainY, testY, trainNormX, testNormX = ProcessingData(originData, mode)
        testY = testY.reset_index(drop = True).values
        
        pretrainModel = LoadPretrainModel(name, mode)
        
        pred = pretrainModel.predict(testNormX, verbose = 2, workers = maxCores, use_multiprocessing = True)
         
        if name == 'In' :
            if mode == 'With Hour' :
                pred.to_csv('result/prediction/WITH HOUR IN Prediction.csv', index = False)
            else :
                pred.to_csv('result/prediction/WITHOUT HOUR IN Prediction.csv', index = False)
            
        else :
            if mode == 'With Hour' :
                pred.to_csv('result/prediction/WITH HOUR OUT Prediction.csv', index = False)
            else :
                pred.to_csv('result/prediction/WITHOUT HOUR OUT Prediction.csv', index = False)
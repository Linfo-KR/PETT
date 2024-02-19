import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from utils import *


def LoadData(dataName, networkName) :
    assert dataName in ['In', 'Out']
    assert networkName in ['RNN', 'LSTM', 'GRU']
        
    if dataName == 'In' :
        originData = pd.read_csv(dataInPath)
        
        if networkName == 'RNN' :
            timeSeriesData = pd.read_csv(rnnInPath)
        elif networkName == 'LSTM' :
            timeSeriesData = pd.read_csv(lstmInPath)
        else :
            timeSeriesData = pd.read_csv(gruInPath)
            
    elif dataName == 'Out' :
        originData = pd.read_csv(dataOutPath)
        
        if networkName == 'RNN' :
            timeSeriesData = pd.read_csv(rnnOutPath)
        elif networkName == 'LSTM' :
            timeSeriesData = pd.read_csv(lstmOutPath)
        else :
            timeSeriesData = pd.read_csv(gruOutPath)
        
    return originData, timeSeriesData


def MergeTimeSeriesData(originData, timeSeriesData) :  
    mergeCondition = ['YEAR', 'MONTH', 'DAY', 'HOUR']
    mergeData = originData.merge(timeSeriesData, on = mergeCondition, how = 'left')
    mergeData.rename(columns = {'PRED' : 'PRED_AVG_TAT'}, inplace = True)
    originData = mergeData
    
    return originData


def ProcessingData(originData, mode) :
    if originData.columns.str.contains('Unnamed: 5').any() == True :
        originData = originData.drop(columns = ['Unnamed: 5'])
        
    else :
        originData
    
    originData = originData.dropna()
    idxSplit = int(len(originData) * 0.8)
    
    assert mode in ['General Model', 'Append Previous Mean TAT']
    
    dependentVar = ['TAT']
    if mode == 'General Model' :
        independentVars = ['HOUR', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN', 'UNDER', 'OVER',
                           'GTE_JOB', 'FIRST_TIME', 'SECOND_TIME', 'THIRD_TIME', 'PRE_TAT_1', 'PRE_TAT_2', 'PRE_TAT_3']
        
    # elif mode == 'Append Previous Mean TAT' :
    #    independentVars = ['HOUR', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN', 'UNDER', 'OVER',
    #                       'GTE_JOB', 'FIRST_TIME', 'SECOND_TIME', 'THIRD_TIME', 'PRE_AVG_TAT']

    weekDay = originData.pop('WEEKDAY')
    cctTag = originData.pop('CCT_TAG')
    timeTag = originData.pop('TIME_TAG')
    
    originData['MON'] = (weekDay == 1) * 1.0
    originData['TUE'] = (weekDay == 2) * 1.0
    originData['WED'] = (weekDay == 3) * 1.0
    originData['THU'] = (weekDay == 4) * 1.0
    originData['FRI'] = (weekDay == 5) * 1.0
    originData['SAT'] = (weekDay == 6) * 1.0
    originData['SUN'] = (weekDay == 7) * 1.0
    
    originData['UNDER'] = (cctTag == 0) * 1.0
    originData['OVER'] = (cctTag == 1) * 1.0
    
    originData['FIRST_TIME'] = (timeTag == 1) * 1.0
    originData['SECOND_TIME'] = (timeTag == 2) * 1.0
    originData['THIRD_TIME'] = (timeTag == 3) * 1.0

    trainX = originData.loc[:idxSplit, independentVars]
    trainY = originData.loc[:idxSplit, dependentVar]
    testX = originData.loc[idxSplit:, independentVars]
    testY = originData.loc[idxSplit:, dependentVar]
    
    trainCloneX = trainX
    testCloneX = testX
    
    scalerDict = {}
    
    for cols in independentVars :
        scalerX = MinMaxScaler()
        trainCloneX[cols] = scalerX.fit_transform(trainCloneX[[cols]])
        testCloneX[cols] = scalerX.transform(testCloneX[[cols]])
        scalerDict[cols] = scalerX
    
    trainNormX = trainCloneX
    testNormX = testCloneX
    
    return trainY, testY, trainNormX, testNormX


def ConvertTensorData(data, label, globalBatchSize, shuffle = True, validation = True, seed = None) :
    validationRatio = 0.8
    validationIndex = int(len(data) * validationRatio)
    
    tensorData = []
        
    if validation :
        validData = data[validationIndex:]
        validLabel = label[validationIndex:]
        
        validTensor = tf.data.Dataset.from_tensor_slices((validData, validLabel))
        validTensor = validTensor.batch(globalBatchSize, drop_remainder = True)
        validTensor = validTensor.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        
        if shuffle :
            validTensor = validTensor.shuffle(buffer_size = len(validData), seed = seed)
            
        tensorData.append(validTensor)
        
    else :
        tensor = tf.data.Dataset.from_tensor_slices((data, label))
        tensor = tensor.batch(globalBatchSize, drop_remainder = True)
        tensor = tensor.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        
        if shuffle :
            tensor = tensor.shuffle(buffer_size = len(data), seed = seed)
            
        tensorData.append(tensor)
    
    return tensorData
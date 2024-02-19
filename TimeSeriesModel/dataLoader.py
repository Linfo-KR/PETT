import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import *


def LoadData(dataName) :
    assert dataName in ['In', 'Out']
    
    if dataName == 'In' :
        originData = pd.read_csv(dataInPath)
        
    elif dataName == 'Out' :
        originData = pd.read_csv(dataOutPath)
        
    return originData
        
        
def processingData(originData, seqLen) :        
    idxTrain = int(len(originData) * 0.8)
    train = originData.loc[:idxTrain, ['AVG_TAT']]
    test = originData.loc[idxTrain:, ['AVG_TAT']]
    trainTimeIndex = originData.loc[:idxTrain, ['DATE']]
    testTimeIndex = originData.loc[idxTrain:, ['DATE']]
    
    timeStep = seqLen
    for i in range(1, timeStep + 1) :
        train['shift_{}'.format(i)] = train['AVG_TAT'].shift(i)
        test['shift_{}'.format(i)] = test['AVG_TAT'].shift(i)
        trainTimeIndex['shift_{}'.format(i)] = trainTimeIndex['DATE'].shift(i)
        testTimeIndex['shift_{}'.format(i)] = testTimeIndex['DATE'].shift(i)
            
    trainX = (train.dropna().drop('AVG_TAT', axis = 1)).values
    trainY = (train.dropna()[['AVG_TAT']]).values
    testX = (test.dropna().drop('AVG_TAT', axis = 1)).values
    testY = (test.dropna()[['AVG_TAT']]).values
    trainTimeIndex = (trainTimeIndex.dropna()[['DATE']]).values
    testTimeIndex = (testTimeIndex.dropna()[['DATE']]).values
    
    sc = MinMaxScaler()
    trainNormX = sc.fit_transform(trainX)
    testNormX = sc.transform(testX)
    
    transformTrainX = trainNormX.reshape(trainNormX.shape[0], timeStep, 1)
    transformTestX = testNormX.reshape(testNormX.shape[0], timeStep, 1)
    
    return transformTrainX, transformTestX, trainY, testY, trainTimeIndex, testTimeIndex
import multiprocessing
import os
import time

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *

from dataloader import *
from utils import *

maxCores = multiprocessing.cpu_count()
print('\n\n Max CPU Cores : ', maxCores, '\n\n')

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1 :
    strategy = tf.distribute.MirroredStrategy(devices = [f'/device:GPU:{i}' for i in range(len(gpus))], cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    print('\n\n Running on Multiple GPUs, Devices Info : ', [gpu.name for gpu in gpus], '\n\n')
elif len(gpus) == 1 :
    strategy = tf.distribute.get_strategy()
    print('\n\n Running on Single GPU, Device Info : ', gpus[0].name)
    print('\n\n #Accelerators : ', strategy.num_replicas_in_sync, '\n\n')
else :
    strategy = tf.distribute.get_strategy()
    print('\n\n Running on CPU', '\n\n')


def BuildModel(stackLayers, stackNeurons, dropRate, learnRate, lambdaL2, trainNormX) :
    with strategy.scope() :
        unit = 2
        
        model = Sequential()
        model.add(Dense(unit ** stackNeurons, input_shape = [len(trainNormX.keys())]))
        
        for layer in range(1, stackLayers + 1) :
            model.add(Dense(unit ** stackNeurons, activation = 'swish', kernel_regularizer = l2(lambdaL2)))
            model.add(BatchNormalization())
            model.add(Dropout(rate = dropRate))
                
        model.add(Dense(unit ** 0))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = learnRate)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
        model.summary()
        
        numLayers = 0
        for layer in model.layers :
            if isinstance(layer, Dense) :
                numLayers += 1
                
            for weight in layer.weights :
                print(f"Layer : {layer.name}, Parameter : {weight.name}, Data Type : {weight.dtype}")
                
        numLayers = numLayers - 2
        numNeurons = [layer.units for layer in model.layers if isinstance(layer, Dense)]
    
    return model, numLayers, numNeurons


def TrainingModel(nameList, modeList, layerList, neuronList, drList, lrList, epochList, batchList, ldList) :
    for name in nameList :
        originData, timeSeriesData = LoadData(name, 'RNN')
        originData = MergeTimeSeriesData(originData, timeSeriesData)
        
        for mode in modeList :
            trainY, testY, trainNormX, testNormX = ProcessingData(originData, mode)
            testY = testY.reset_index(drop = True).values
            
            for layer in layerList :
                for neuron in neuronList :
                    for dr in drList :
                        for lr in lrList :
                            for epochs in epochList :
                                for batchs in batchList :
                                    for ld in ldList :
                                        sTime = time.time()
                                        initLambda = 0.1
                                        weightLambda = 1.1
                                        
                                        lambdaL2 = initLambda * (weightLambda ** ld)
                                        
                                        model, numLayers, numNeurons = BuildModel(layer, neuron, dr, lr, ld, trainNormX)

                                        history = model.fit(trainNormX,
                                                            trainY,
                                                            epochs=epochs,
                                                            batch_size=batchs,
                                                            validation_split=0.2,
                                                            shuffle=False,
                                                            verbose=2,
                                                            use_multiprocessing=True)
                                        
                                        eTime = time.time()
                                        
                                        predY = model.predict(testNormX, verbose = 2, workers = maxCores, use_multiprocessing = True)
                                        
                                        DrawGraph(history, testY, predY, name, sTime)
                                        
                                        mapeAct = MAPE(testY, predY)
                                        rmseAct = RMSE(testY, predY)
                                        accuracy = 100 - mapeAct
                                        runTime = round((eTime - sTime) / 60, 2)
                                        
                                        mask = '%Y%m%d_%H%M%S'
                                        dte = time.strftime(mask, time.localtime(sTime))
                                        sname = '-{}.h5'.format(dte)
                            
                                        metricsResult = pd.DataFrame(
                                            {
                                                "Test Time": [dte],
                                                "Learning Time": [runTime],
                                                "Mode" : [mode],
                                                "Num Layers" : [numLayers],
                                                "Num Neurons" : [numNeurons],
                                                "Epoch": [epochs],
                                                "Batch Size": [batchs],
                                                "Learning Rate" : [lr],
                                                "Dropout Rate" : [dr],
                                                "Lambda L2" : [lambdaL2],
                                                "Accuracy" : [accuracy],
                                                "MAPE": [mapeAct],
                                                "RMSE": [rmseAct]
                                            }
                                        )
                                        
                                        print(metricsResult)
                                        SaveResult(name, metricsResult, model, sname)
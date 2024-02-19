import multiprocessing
import os
import time

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from dataLoader import *
from utils import *

maxCores = multiprocessing.cpu_count()
print('\n\n Max CPU Cores : ', maxCores, '\n\n')

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    

def BuildModel(networkName, stackLayers, stackNeurons, activation, dropRate, learnRate, transformTrainX) :
    with strategy.scope() :
        unit = 2
        inputShape = (transformTrainX.shape[1], transformTrainX.shape[2])
        
        model = Sequential()
        
        assert networkName in ['RNN', 'LSTM', 'GRU']
        
        if networkName == 'RNN' :
            model.add(SimpleRNN(unit ** stackNeurons, activation = activation, return_sequences = True, input_shape = inputShape))
        
            for layer in range(1, (stackLayers + 1)) :
                if (stackNeurons + layer) != (stackNeurons + stackLayers) :
                    model.add(SimpleRNN(unit ** (stackNeurons + layer), activation = activation, return_sequences = True))
                    model.add(Dropout(rate = dropRate))

                else :
                    model.add(SimpleRNN(unit ** (stackNeurons + layer), activation = activation, return_sequences = False))
                    model.add(Dropout(rate = dropRate))
                        
            model.add(Dense(unit ** 0))
        
        elif networkName == 'LSTM' :
            model.add(LSTM(unit ** stackNeurons, activation = activation, return_sequences = True, input_shape = inputShape))
        
            for layer in range(1, (stackLayers + 1)) :
                if (stackNeurons + layer) != (stackNeurons + stackLayers) :
                    model.add(LSTM(unit ** (stackNeurons + layer), activation = activation, return_sequences = True))
                    model.add(Dropout(rate = dropRate))
                    
                else :
                    model.add(LSTM(unit ** (stackNeurons + layer), activation = activation, return_sequences = False))
                    model.add(Dropout(rate = dropRate))
                        
            model.add(Dense(unit ** 0))
            
        else :
            model.add(GRU(unit ** stackNeurons, activation = activation, return_sequences = True, input_shape = inputShape))
        
            for layer in range(stackLayers - 1) :
                model.add(GRU(unit ** stackNeurons, activation = activation, return_sequences = True))
                model.add(Dropout(rate = dropRate))
                
            model.add(GRU(unit ** stackNeurons, activation = activation, return_sequences = False))
            model.add(Dropout(rate = dropRate))   
                
            model.add(Dense(unit ** 0))
        
        optimizer = tf.keras.optimizers.Adam(lr = learnRate)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
        model.summary()
        
        numLayers = 0
        for layer in model.layers :
            if isinstance(layer, SimpleRNN) or isinstance(layer, LSTM) or isinstance(layer, GRU) :
                numLayers += 1
                
            for weight in layer.weights :
                print(f"Layer: {layer.name}, Parameter: {weight.name}, Data Type: {weight.dtype}")
                
        numLayers = numLayers - 1
        numNeurons = [layer.units for layer in model.layers if isinstance(layer, SimpleRNN) or isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Dense)]
    
    return model, numLayers, numNeurons


def TrainingModel(networkList, nameList, stepList, layerList, neuronList, activationList, drList, lrList, epochList, batchList) :
    for network in networkList :
        for name in nameList :
            for step in stepList :
                originData = LoadData(name)
                transformTrainX, transformTestX, trainY, testY, trainTimeIndex, testTimeIndex = processingData(originData, step)
                
                for layer in layerList :
                    for neuron in neuronList :
                        for activation in activationList :
                            for dr in drList :
                                for lr in lrList :
                                    for epochs in epochList :
                                        for batchs in batchList :
                                            sTime = time.time()

                                            model, numLayers, numNeurons = BuildModel(network, layer, neuron, activation, dr, lr, transformTrainX)

                                            history = model.fit(transformTrainX,
                                                                trainY,
                                                                epochs=epochs,
                                                                batch_size=batchs,
                                                                validation_split=0.2,
                                                                shuffle=False,
                                                                verbose=2,
                                                                use_multiprocessing=True)
                                            
                                            eTime = time.time()
                                            
                                            predY = model.predict(transformTestX, verbose = 2, workers = maxCores, use_multiprocessing = True)
                                            predY = predY.reshape(-1, predY.shape[-1])
                                            
                                            DrawGraph(history, testY, predY, network, name, sTime)
                                            
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
                                                    "Network" : [network],
                                                    "TimeStep" : [step],
                                                    "Num Layers" : [numLayers],
                                                    "Num Neurons" : [numNeurons],
                                                    "Epoch": [epochs],
                                                    "Batch Size": [batchs],
                                                    "Learning Rate" : [lr],
                                                    "Dropout Rate" : [dr],
                                                    "Accuracy" : [accuracy],
                                                    "MAPE": [mapeAct],
                                                    "RMSE": [rmseAct]
                                                }
                                            )
                                            
                                            print(metricsResult)
                                            
                                            if name == 'In':
                                                if not os.path.exists(resultInPath) :
                                                    metricsResult.to_csv(resultInPath, mode = 'w', header = True, index = False)
                                                else : 
                                                    metricsResult.to_csv(resultInPath, mode = 'a', header = False, index = False)
                                                
                                                if network == 'RNN' :
                                                    model.save(saveRnnInPath + sname)
                                                elif network == 'LSTM' :
                                                    model.save(saveLstmInPath + sname)
                                                else :
                                                    model.save(saveGruInPath + sname)
                                                    
                                            elif name == 'Out':
                                                if not os.path.exists(resultOutPath) :
                                                    metricsResult.to_csv(resultOutPath, mode = 'w', header = True, index = False)
                                                else : 
                                                    metricsResult.to_csv(resultOutPath, mode = 'a', header = False, index = False)
                                                
                                                if network == 'RNN' :
                                                    model.save(saveRnnOutPath + sname)
                                                elif network == 'LSTM' :
                                                    model.save(saveLstmOutPath + sname)
                                                else :
                                                    model.save(saveGruOutPath + sname)
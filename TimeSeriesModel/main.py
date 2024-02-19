## Predict External Trucks TurnAroundTime(PETT)
## Using Recurrent Neural Networks(RNN) and Long-Short Term Memory(LSTM) and Gated Recurrent Unit(GRU)

from dataLoader import *
from models import *
from utils import *

# Configuration
networkList = ['RNN', 'LSTM', 'GRU']
nameList = ['In', 'Out']
stepList = range(1, 25)
layerList = [3, 4, 5]
neuronList = [6, 7]
activationList = ['tanh']
drList = [0, 0.1, 0.2]
lrList = [0.0001, 0.0005, 0.001, 0.0025]
epochList = [500, 1000]
batchList = [512, 1024]

def main():
    TrainingModel(networkList, nameList, stepList, layerList, neuronList, activationList, drList, lrList, epochList, batchList)
            
if __name__ == '__main__' :
    main()
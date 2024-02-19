## Predict External Trucks TurnAroundTime(PETT)
## Using Deep Neural Networks(DNN)

from dataloader import *
from models import *
from utils import *

# Configuration
nameList = ['In', 'Out']
modeList = ['General Model']
layerList = [2]
neuronList = [6]
drList = [0]
lrList = [0.001]
epochList = [10]
batchList = [64]
ldList = [0]    

def main():
    TrainingModel(nameList, modeList, layerList, neuronList, drList, lrList, epochList, batchList, ldList)
        
if __name__ == '__main__' :
    main()
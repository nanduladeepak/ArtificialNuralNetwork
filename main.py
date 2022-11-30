from ann import ANN 
import numpy as np

from save_model import saveModelNpy , LoadModelNpy

# writh the ann as a class and take the ann structure for the constructor
randomWaightsArr = np.random.random((2, 2))

inputVector = [1, 0.5]

expextedOutput = [1, 0.5]

ann = ANN(1, 2, 2)

ann.setupAnn([randomWaightsArr, randomWaightsArr])
savedModelNpy = LoadModelNpy('savedModels/npyFile','v1')
ann.setModel(savedModelNpy)

for i in range(0,1000):
    ann.trainAnn(inputVector, expextedOutput)


model = ann.getModel()

print(model)

# saveModelCsv(model,'csvFile','v1')

saveModelNpy(model,'savedModels/npyFile','v1')

# savedModelCsv = LoadModelCsv('npyFile','v1')
savedModelNpy = LoadModelNpy('savedModels/npyFile','v1')

# print('csv \n')
# print(savedModelCsv)

print('\nnpy\n')
print(savedModelNpy)


from ann import ANN 
import numpy as np
import matplotlib.pyplot as plt
from save_model import saveModelNpy , LoadModelNpy

# writh the ann as a class and take the ann structure for the constructor
randomWaightsArr = np.random.uniform(low=0.1, high=1.0,size=(2, 2))

inputVector = [1, 0.5]

expextedOutput = [1, 0.5]

ann = ANN(2, 4, 1, 2)

ann.setupAnn()

# savedModelNpy = LoadModelNpy('savedModels/npyFile','v1')
# ann.setModel(savedModelNpy)

rsmeList = []

for i in range(0,10000):
    rsmeList.append(ann.trainAnn(inputVector, expextedOutput))


plt.plot(rsmeList)
plt.show()

# model = ann.getModel()

# print(model)

# saveModelCsv(model,'csvFile','v1')

# saveModelNpy(model,'savedModels/npyFile','v1')

# # savedModelCsv = LoadModelCsv('npyFile','v1')
# savedModelNpy = LoadModelNpy('savedModels/npyFile','v1')

# # print('csv \n')
# # print(savedModelCsv)

# print('\nnpy\n')
# print(savedModelNpy)


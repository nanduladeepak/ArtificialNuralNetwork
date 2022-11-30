from ann import ANN 
import numpy as np

# writh the ann as a class and take the ann structure for the constructor
randomWaightsArr = np.random.random((2, 2))

inputVector = [1, 0.5]

expextedOutput = [1, 0.5]

ann = ANN(1, 2, 2)

ann.setupAnn([randomWaightsArr, randomWaightsArr])

for i in range(0,1000):
    ann.trainAnn(inputVector, expextedOutput)




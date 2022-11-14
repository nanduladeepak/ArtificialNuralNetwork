import math
import random
import numpy as np
from layer import Layer

hiddenVectorLength = 2
outputVectorLength = 2
expextedOutput = [1, 0.5]

inputVector = [1,0.5]

lamda = 0.3
epsilon = 0.6

randomWaightsArr = np.random.random((2,2))

layer1  = Layer(2)
layer2  = Layer(2)

layer1.assignNurons(randomWaightsArr)
layer2.assignNurons(randomWaightsArr)

layer1Output = layer1.getOutput(inputVector)
print(layer1Output)

layer2Output = layer2.getOutput(layer1Output)

print(layer2Output)
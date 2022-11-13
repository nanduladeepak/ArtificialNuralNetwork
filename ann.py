import math
import random
# import numpy as np

hiddenVectorLength = 2
outputVectorLength = 2
expextedOutput = [1, 0.5]

lamda = 0.3
epsilon = 0.6


def sigmoid(x: float):
    return 1 / (1 + math.exp(-lamda * x))


def getValue(inputVec: list, weight: list, node: int):
    sum = 0
    for index, val in enumerate(inputVec):
        sum += (val*weight[node][index])
    return sigmoid(sum)


def localGradient(nuronValue: float, error:float):
    return lamda*nuronValue*(1-nuronValue)*error


inputVector = random.sample(range(1, 10), 2)
weight1 = np.random.random_sample(size=(hiddenVectorLength, len(inputVector)))
weight2 = np.random.random_sample(size=(hiddenVectorLength, len(inputVector)))

# getValue(inputVector,weight1,0),getValue(inputVector,weight1,1)
hiddenVector = []

for i in range(0, hiddenVectorLength):
    hiddenVector.append(getValue(inputVector, weight1, i))

# getValue(hiddenVector,weight2,0),getValue(hiddenVector,weight2,1)
outputVector = []

for i in range(0, outputVectorLength):
    outputVector.append(getValue(hiddenVector, weight2, i))

outputLayerErrors = []
localGradientVector = []

for output, expectedOutput in zip(outputLayerErrors, expextedOutput):
    outputLayerErrors.append(())
    error = output-expectedOutput
    outputLayerErrors.append(error)
    localGradientVector.append(localGradient(output,error))

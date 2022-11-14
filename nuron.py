import math
import numpy as np

lamda = 0.3

def sigmoid(x: float):
    return 1 / (1 + math.exp(-lamda * x))


class Nuron:
    def __init__(self,inWeights):
        self.inWeights = inWeights

    def getWeights(self):
        return self.inWeights

    def getOutputValue(self, nuronValues):
        return sigmoid(np.dot(nuronValues,self.inWeights))


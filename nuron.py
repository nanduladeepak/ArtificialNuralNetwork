import math
import numpy as np

lamda = 0.3


class Nuron:
    def __init__(self, inWeights):
        self.inWeights = inWeights
        self.value = None

    def getWeights(self):
        return self.inWeights

    def getOutputValue(self, nuronValues):
        self.value = self.sigmoid(np.dot(nuronValues, self.inWeights))
        return self.value

    def sigmoid(self, x: float):
        return 1 / (1 + math.exp(-lamda * x))

    def getNuronVal(self):
        return self.value

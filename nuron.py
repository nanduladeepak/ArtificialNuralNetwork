import math
import numpy as np


class Nuron:
    def __init__(self, lamda, inWeights):
        self.lamda = lamda
        self.inWeights = inWeights
        self.value = None

    def getWeights(self):
        return self.inWeights

    def getOutputValue(self, nuronValues):
        self.value = self.sigmoid(np.dot(nuronValues, self.inWeights))
        return self.value

    def sigmoid(self, x: float):
        return 1 / (1 + math.exp(-self.lamda * x))

    def getLocalGradient(self, error: int):
        return (self.lamda * self.value * (1-self.value) * error)

    def getNuronVal(self):
        return self.value

    def updateInWeights(self,updatedInWeights):
        self.inWeights = updatedInWeights

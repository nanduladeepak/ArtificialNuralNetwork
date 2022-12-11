import math
import numpy as np
import copy

# class representing nuron
class Nuron:
    def __init__(self, lamda:float, inWeights:list):
        self.lamda = lamda
        self.inWeights = copy.deepcopy(inWeights)
        self.value = None

    # used to get the attached weights to the nuron
    def getWeights(self):
        return copy.deepcopy(self.inWeights)

    # used to get the output of this nuron
    def getOutputValue(self, nuronValues:list):
        self.value = self.sigmoid(np.dot(nuronValues.copy(), self.inWeights.copy()))
        return copy.deepcopy(self.value)

    # activation function
    def sigmoid(self, x: float):
        return 1 / (1 + math.exp(-self.lamda * x))

    # function for calculatiiong local gradient of the nuron
    def getLocalGradient(self, error: int):
        return (self.lamda * self.value * (1-self.value) * error)

    # to get nuron output which was calculated and stored in the object
    def getNuronVal(self):
        return self.value

    # function to update the attached weights of the nurons
    def updateInWeights(self,updatedInWeights:list):
        self.inWeights = copy.deepcopy(updatedInWeights)

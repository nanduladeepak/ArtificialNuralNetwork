import numpy as np
from layer import Layer
# import math
# import random



# layer1 = Layer(2)
# layer2 = Layer(2)

# layer1.assignNurons(randomWaightsArr)
# layer2.assignNurons(randomWaightsArr)

# layer1Output = layer1.getOutput(inputVector)
# print(layer1Output)

# layer2Output = layer2.getOutput(layer1Output)

# print(layer2Output)


class ANN:
    def __init__(self, noOfHiddenLayers: int, noOfNurons: int, noOfOutputs: int):
        # below constants are not passed down do it later
        self.lamda = 0.3
        self.epsilon = 0.6
        self.alpha = 0.6

        self.noOfHiddenLayers = noOfHiddenLayers
        self.noOfNurons = noOfNurons
        self.noOfOutputs = noOfOutputs
        self.layers = []
        self.localGradient = []
        self.deltaW = []

    def setupAnn(self,inBoundWeightsVectors):
        for index in range(0,self.noOfHiddenLayers-1):
            self.layers.append(Layer(self.noOfNurons,inBoundWeightsVectors[index]))
            self.layers[index].assignNurons()
        self.layers.append(Layer(self.noOfOutputs,inBoundWeightsVectors[-1]))
        self.layers[-1].assignNurons()

    def getPredOutput(self,inputVector):
        for layer in self.layers:
            inputVector = layer.getOutput(inputVector)
        return inputVector

    # def trainAnn(self,inputVector, expectedOutput):
    #     predictedoutput = self.getPredOutput(inputVector)
    #     for index in range(len(self.layers),0,-1):
    #         self.localGradient.insert(0,)





# writh the ann as a class and take the ann structure for the constructor


randomWaightsArr = np.random.random((2, 2))

inputVector = [1, 0.5]

expextedOutput = [1, 0.5]

ann = ANN(1,2,2)

ann.setupAnn([randomWaightsArr,randomWaightsArr])
print(ann.getPredOutput(inputVector))
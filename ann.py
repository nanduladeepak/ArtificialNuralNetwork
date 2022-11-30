from layer import Layer
import math
import numpy as np

class ANN:
    def __init__(self, noOfHiddenLayers: int, noOfNurons: int, noOfOutputs: int):
        self.lamda = 0.3
        self.epsilon = 0.6
        self.alpha = 0.6

        self.noOfHiddenLayers = noOfHiddenLayers
        self.noOfNurons = noOfNurons
        self.noOfOutputs = noOfOutputs
        self.layers = []
        self.localGradient = []
        self.deltaW = []
        self.firstPass = True

    def setupAnn(self, inBoundWeightsVectors):
        for index in range(0, self.noOfHiddenLayers):
            self.deltaW.append([])
            self.localGradient.append([])
            self.layers.append(
                Layer(self.lamda, self.epsilon, self.noOfNurons, inBoundWeightsVectors[index]))
            self.layers[index].assignNurons()
        self.layers.append(Layer(self.lamda, self.epsilon,
                                 self.noOfOutputs, inBoundWeightsVectors[-1]))

        self.deltaW.append([])
        self.localGradient.append([])
        for i in range(self.noOfNurons):
            for j in range(0, self.noOfHiddenLayers):
                self.deltaW[j].append(None)
                self.localGradient[j].append(None)

        for i in range(self.noOfOutputs):
            self.deltaW[-1].append(None)
            self.localGradient[-1].append(None)

        self.layers[-1].assignNurons()

    def getPredOutput(self, inputVector):
        for layer in self.layers:
            inputVector = layer.getOutput(inputVector)
        return inputVector

    def trainAnn(self, inputVector, expectedOutput):
        predictedoutput = self.getPredOutput(inputVector)
        errors = []
        for pred, exp in zip(predictedoutput, expectedOutput):
            errors.append(exp - pred)
        self.calculateLocalGradient(errors)

        for layer, deltaWvector in zip(self.layers, self.deltaW):
            layer.updateWeights(deltaWvector)
        self.firstPass = False
        predictedoutput = self.getPredOutput(inputVector)
        # print(predictedoutput)
        MSE = np.square(np.subtract(expectedOutput,predictedoutput)).mean() 
        RMSE = math.sqrt(MSE)
        # print("Root Mean Square Error:\n")
        print(RMSE)

    def calculateLocalGradient(self, errors):
        self.localGradient[-1] = self.layers[-1].getLocalGradients(errors)
        index = self.noOfHiddenLayers
        while (index >= 0):
            errors = self.layers[index].getHiddenLayerError(
                self.localGradient[index])
            # self.deltaW[index] = self.layers[index].getDeltaW(
            #     self.localGradient[index]) + (0 if (self.deltaW[index] is None) else (self.alpha*self.deltaW[index]))
            deltaRes = self.layers[index].getDeltaW(
                self.localGradient[index])
            if(self.firstPass == False):
                for j , (deltaVal,deltaTmOne) in enumerate(zip(deltaRes,self.deltaW[index])):
                    self.deltaW[index][j] = deltaVal+self.alpha*deltaTmOne
            else:
                self.deltaW[index] = deltaRes
            if index != 0:
                self.localGradient[index -
                                   1] = self.layers[index].getLocalGradients(errors)
            index -= 1

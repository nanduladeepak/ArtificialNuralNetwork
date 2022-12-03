from layer import Layer
import math
import numpy as np


class ANN:
    def __init__(self, noOfInputs: int, noOfNurons: int, noOfHiddenLayers: int, noOfOutputs: int):
        self.lamda = 0.6
        self.epsilon = 1
        self.alpha = 0.9

        self.noOfHiddenLayers = noOfHiddenLayers
        self.noOfNurons = noOfNurons
        self.noOfOutputs = noOfOutputs
        self.noOfInputs = noOfInputs
        self.layers = []
        self.localGradient = []
        self.deltaW = []
        self.firstPass = True

    def setupAnn(self, inBoundWeightsVectors=None):
        if inBoundWeightsVectors is None:
            inBoundWeightsVectors = []
            index = 0
            while (index < self.noOfHiddenLayers):
                inBoundWeightsVectors.append(np.random.random(
                    (self.noOfNurons, self.noOfInputs if (index == 0) else self.noOfNurons)))
                index += 1
            inBoundWeightsVectors.append(
                np.random.random((self.noOfOutputs, self.noOfNurons)))
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
        MSE = np.square(np.subtract(expectedOutput, predictedoutput)).mean()
        RMSE = math.sqrt(MSE)
        # print("Root Mean Square Error:\n")
        return RMSE

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
            if (self.firstPass == False):
                for j, (deltaVal, deltaTmOne) in enumerate(zip(deltaRes, self.deltaW[index])):
                    self.deltaW[index][j] = deltaVal+self.alpha*deltaTmOne
            else:
                self.deltaW[index] = deltaRes
            if index != 0:
                self.localGradient[index -
                                   1] = self.layers[index].getLocalGradients(errors)
            index -= 1

    def getModel(self):
        model = []
        for layer in self.layers:
            # weights = layer.getInBoundWeights()
            # model.append(weights.tolist() if type(weights) is np.ndarray else weights) 
            model.append(layer.getInBoundWeights()) 
        return model

    def setModel(self, model):
        for layer, weights in zip(self.layers, model):
            layer.updateInBoundWeights(weights)

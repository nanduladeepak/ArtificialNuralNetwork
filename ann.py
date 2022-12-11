from layer import Layer
import math
import numpy as np


class ANN:
    def __init__(self, noOfInputs: int, noOfNurons: int, noOfHiddenLayers: int, noOfOutputs: int):
        self.lamda = 0.01
        self.epsilon = 0.001
        self.alpha = 0.001

        self.noOfHiddenLayers = noOfHiddenLayers
        self.noOfNurons = noOfNurons
        self.noOfOutputs = noOfOutputs
        self.noOfInputs = noOfInputs
        self.noOfInputs+=1
        self.layers = []
        self.localGradient = []
        self.deltaW = []
        self.inputVector = None

    def setupAnn(self, inBoundWeightsVectors=None):
        if inBoundWeightsVectors is None:
            inBoundWeightsVectors = []
            index = 0
            while (index < self.noOfHiddenLayers):
                inBoundWeightsVectors.append(np.random.uniform(low=0.1, high=1.0,
                    size=(self.noOfNurons, self.noOfInputs if (index == 0) else self.noOfNurons)).tolist())
                index += 1
            inBoundWeightsVectors.append(
                np.random.uniform(low=0.1, high=1.0,size=(self.noOfOutputs, self.noOfNurons)).tolist())
        for index in range(0, self.noOfHiddenLayers):
            self.deltaW.append([])
            self.localGradient.append([])
            self.layers.append(
                Layer(self.lamda, self.noOfNurons, inBoundWeightsVectors[index]))
            self.layers[index].assignNurons()
        self.layers.append(Layer(self.lamda, self.noOfOutputs, inBoundWeightsVectors[-1]))

        self.deltaW.append([])
        self.localGradient.append([])
        for i in range(self.noOfNurons):
            self.deltaW[0].append([])
            for k in range(self.noOfInputs):
                self.deltaW[0][i].append(None)
        # fix delta w from below
        for i in range(self.noOfNurons):
            for j in range(0, self.noOfHiddenLayers):
                self.localGradient[j].append(None)

        i = 1
        while (i < self.noOfHiddenLayers):
            self.deltaW[i] = []
            for j in range(self.noOfNurons):
                self.deltaW[i].append([])
                for k in range(self.noOfNurons):
                    self.deltaW[i][j].append(None)
            i += 1
        self.deltaW[-1] = []
        for i in range(self.noOfOutputs):
            self.deltaW[-1].append([])
            for k in range(self.noOfNurons):
                self.deltaW[-1][i].append(None)

        for i in range(self.noOfOutputs):
            self.localGradient[-1].append(None)

        self.layers[-1].assignNurons()

    def getPredOutput(self, inputVector,addBias=False):
        if(addBias):
            if(isinstance(inputVector, np.ndarray)):
                inputVector = inputVector.tolist()
            inputVector.append(1)
        self.inputVector = inputVector.copy()
        for layer in self.layers:
            inputVector = layer.getOutput(inputVector.copy())
        return inputVector

    def trainAnn(self, inputVector, expectedOutput):
        if(isinstance(inputVector, np.ndarray)):
            inputVector = inputVector.tolist()
        inputVector.append(1)
        predictedoutput = self.getPredOutput(inputVector,False)
        errors = []
        for pred, exp in zip(predictedoutput, expectedOutput):
            errors.append(exp - pred)
        self.calculateLocalGradient(errors)

        for layer, deltaWvector in zip(self.layers, self.deltaW):
            layer.updateWeights(deltaWvector)
        predictedoutput = self.getPredOutput(inputVector)
        MSE = np.square(np.subtract(expectedOutput, predictedoutput)).mean()
        RMSE = math.sqrt(MSE)
        return RMSE

    def getRsme(self, x, y):
        y_pred = self.getPredOutput(x,True)
        MSE = np.square(np.subtract(y, y_pred)).mean()
        RMSE = math.sqrt(MSE)
        return RMSE

    def calculateLocalGradient(self, errors):
        self.localGradient[-1] = self.layers[-1].getLocalGradients(errors)
        self.deltaW[-1] = self.getDeltaW(
            self.localGradient[-1], self.deltaW[-1],self.layers[-2].getNuronsOutputs())
        index = self.noOfHiddenLayers-1
        while (index >= 0):
            errors = self.layers[index].getHiddenLayerError(
                self.localGradient[index+1])

            self.localGradient[index] = self.layers[index].getLocalGradients(
                errors)
            if(index!=0):
                self.deltaW[index] = self.getDeltaW(
                    self.localGradient[index], self.deltaW[index],self.layers[index-1].getNuronsOutputs())
            else:
                self.deltaW[index] = self.getDeltaW(
                    self.localGradient[index], self.deltaW[index],self.inputVector)

            index -= 1

    def getDeltaW(self, gradients, deltaWs,nuronsOutputs):
        for i, gradient in enumerate(gradients):
            for j, nuronVal in enumerate(nuronsOutputs):
                deltaWs[i][j] = self.epsilon*gradient*nuronVal + (self.alpha*deltaWs[i][j] if deltaWs[i][j] is not None else 0)
        return deltaWs

    def getModel(self):
        model = []
        for layer in self.layers:
            model.append(layer.getInBoundWeights())
        return model

    def setModel(self, model):
        for layer, weights in zip(self.layers, model):
            layer.updateInBoundWeights(weights)

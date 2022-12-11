from layer import Layer
import math
import numpy as np
import copy

# class for artificial nural network
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

    # set saved or random weights
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

    # function to predict using ann
    def getPredOutput(self, inputVector:list,addBias:bool=False):
        if(addBias):
            if(isinstance(inputVector, np.ndarray)):
                inputVector = inputVector.tolist()
            inputVector.append(1)
        self.inputVector = inputVector.copy()
        for layer in self.layers:
            inputVector = layer.getOutput(inputVector.copy())
        return inputVector

    # function to train ann
    def trainAnn(self, inputVector:list, expectedOutput:list):
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

    # function to calculate root mean square error
    def getRsme(self, x:list, y:list):
        y_pred = self.getPredOutput(x,True)
        MSE = np.square(np.subtract(y, y_pred)).mean()
        RMSE = math.sqrt(MSE)
        return RMSE

    # function to trigger the calculation of local gradient and delta-w
    def calculateLocalGradient(self, errors:list):
        self.localGradient[-1] = copy.deepcopy(self.layers[-1].getLocalGradients(errors))
        self.deltaW[-1] = copy.deepcopy(self.getDeltaW(
            self.localGradient[-1], self.deltaW[-1],self.layers[-2].getNuronsOutputs()))
        index = self.noOfHiddenLayers-1
        while (index >= 0):
            errors = copy.deepcopy(self.layers[index].getHiddenLayerError(
                self.localGradient[index+1]))

            self.localGradient[index] = copy.deepcopy(self.layers[index].getLocalGradients(
                errors))
            if(index!=0):
                self.deltaW[index] = copy.deepcopy(self.getDeltaW(
                    self.localGradient[index], self.deltaW[index],self.layers[index-1].getNuronsOutputs()))
            else:
                self.deltaW[index] = copy.deepcopy(self.getDeltaW(
                    self.localGradient[index], self.deltaW[index],self.inputVector))

            index -= 1

    # function to calculate delta-w
    def getDeltaW(self, gradients:list, deltaWs:list,nuronsOutputs:list):
        for i, gradient in enumerate(gradients):
            for j, nuronVal in enumerate(nuronsOutputs):
                deltaWs[i][j] = self.epsilon*gradient*nuronVal + (self.alpha*deltaWs[i][j] if deltaWs[i][j] is not None else 0)
        return deltaWs

    # function to get weights of the ann
    def getModel(self):
        model = []
        for layer in self.layers:
            model.append(layer.getInBoundWeights())
        return model

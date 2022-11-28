from layer import Layer

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

    def setupAnn(self, inBoundWeightsVectors):
        for index in range(0, self.noOfHiddenLayers):
            self.layers.append(
                Layer(self.lamda, self.epsilon, self.noOfNurons, inBoundWeightsVectors[index]))
            self.layers[index].assignNurons()
        self.layers.append(Layer(self.lamda, self.epsilon,
                                 self.noOfOutputs, inBoundWeightsVectors[-1]))
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
        self.deltaW.insert(0, [])
        for gradient, input in zip(self.localGradient[0], inputVector):
            self.deltaW[0].append(self.epsilon*gradient*input)

        # print(self.localGradient)
        # print(self.deltaW)
        for layer, deltaWvector in zip(self.layers, self.deltaW):
            layer.updateWeights(deltaWvector)
        
        predictedoutput = self.getPredOutput(inputVector)
        print(predictedoutput)
        # self.localGradient.insert(0, self.layers[-1].getLocalGradients(errors))
        # for index in range(self.noOfHiddenLayers, 0, -1):
        #     errors = self.layers[index].getHiddenLayerError(
        #         self.localGradient[0])
        #     self.localGradient.insert(
        #         0, self.layers[index].getLocalGradients(errors))
        # print(self.localGradient)

    def calculateLocalGradient(self, errors):
        self.localGradient.insert(0, self.layers[-1].getLocalGradients(errors))
        for index in range(self.noOfHiddenLayers, 0, -1):
            errors = self.layers[index].getHiddenLayerError(
                self.localGradient[0])
            self.deltaW.insert(
                0, self.layers[index].getDeltaW(self.localGradient[0]))
            self.localGradient.insert(
                0, self.layers[index].getLocalGradients(errors))

    # def calculateDeltaW(self):
    #     if (len(self.deltaW) == 0):
    #         for layer in self.layers:
    #             layer.getDeltaW()

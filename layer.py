from nuron import Nuron

class Layer:

    def __init__(self, lamda, noOfNurons: int, inBoundWeightsVectors):
        self.lamda = lamda
        self.noOfNurons = noOfNurons
        self.nurons = []
        self.inBoundWeightsVectors = inBoundWeightsVectors

    def assignNurons(self):
        for index in range(0, self.noOfNurons):
            self.nurons.append(
                Nuron(self.lamda, self.inBoundWeightsVectors[index]))

    def getLocalGradients(self, error):
        localGradient = []
        for index, nuron in enumerate(self.nurons):
            localGradient.append(nuron.getLocalGradient(error[index]))
        return localGradient

    def getHiddenLayerError(self, gradients):
        error = []
        for weights in self.inBoundWeightsVectors:
            error.append(sum([grad*weight for grad,weight in zip(gradients,weights)]))
        return error

    def getNuronsOutputs(self):
        outputs = []
        for nuron in self.nurons:
            outputs.append(nuron.getNuronVal())
        return outputs

    def updateWeights(self, deltaWVector):
        for index, (deltaW, weights, nuron) in enumerate(zip(deltaWVector, self.inBoundWeightsVectors, self.nurons)):
            self.inBoundWeightsVectors[index] = [
                x + dw for x, dw in zip(weights, deltaW)]
            nuron.updateInWeights(self.inBoundWeightsVectors[index])

    def getOutput(self, inputVector):
        output = []
        for nuron in self.nurons:
            output.append(nuron.getOutputValue(inputVector))
        return output

    def getInBoundWeights(self):
        return self.inBoundWeightsVectors

    def updateInBoundWeights(self, storedWeights):
        self.inBoundWeightsVectors = storedWeights
        for weights, nuron in zip(self.inBoundWeightsVectors, self.nurons):
            nuron.updateInWeights(weights)

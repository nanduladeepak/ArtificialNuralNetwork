from nuron import Nuron
import copy

# class for each layer in nural network
class Layer:

    def __init__(self, lamda:float, noOfNurons: int, inBoundWeightsVectors:list):
        self.lamda = lamda
        self.noOfNurons = noOfNurons
        self.nurons = []
        self.inBoundWeightsVectors = copy.deepcopy(inBoundWeightsVectors)

    # setting up nurons in this layer
    def assignNurons(self):
        for index in range(0, self.noOfNurons):
            self.nurons.append(
                Nuron(self.lamda, self.inBoundWeightsVectors[index]))

    # function to calculate localGradient of all the nurons int he layer
    def getLocalGradients(self, error:list):
        localGradient = []
        for index, nuron in enumerate(self.nurons):
            localGradient.append(nuron.getLocalGradient(error[index]))
        return localGradient

    # calculating error for hidden layers
    def getHiddenLayerError(self, gradients:list):
        error = []
        for weights in self.inBoundWeightsVectors:
            error.append(sum([grad*weight for grad,weight in zip(gradients,weights)]))
        return error

    # getting saved outputs of each nurons
    def getNuronsOutputs(self):
        outputs = []
        for nuron in self.nurons:
            outputs.append(nuron.getNuronVal())
        return outputs

    # updating weights gor each nurons in the layer
    def updateWeights(self, deltaWVector:list):
        for index, (deltaW, weights, nuron) in enumerate(zip(deltaWVector, self.inBoundWeightsVectors, self.nurons)):
            self.inBoundWeightsVectors[index] = [
                x + dw for x, dw in zip(weights, deltaW)]
            nuron.updateInWeights(self.inBoundWeightsVectors[index])

    # calculating and getting outpust of this layer
    def getOutput(self, inputVector:list):
        output = []
        for nuron in self.nurons:
            output.append(nuron.getOutputValue(inputVector))
        return output

    # getting weights attached to all nurons in the layer
    def getInBoundWeights(self):
        return self.inBoundWeightsVectors

    # update weights for all the nurons 
    def updateInBoundWeights(self, storedWeights:list):
        self.inBoundWeightsVectors = copy.deepcopy(storedWeights)
        for weights, nuron in zip(self.inBoundWeightsVectors, self.nurons):
            nuron.updateInWeights(weights)

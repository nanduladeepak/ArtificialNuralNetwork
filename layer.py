from nuron import Nuron

class Layer:

    def __init__(self, lamda, epsilon, noOfNurons: int, inBoundWeightsVectors):
        self.epsilon = epsilon
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
        for grad, weights in zip(gradients, self.inBoundWeightsVectors):
            sum = 0
            for weight in weights:
                sum += (grad * weight)
            error.append(sum)
        return error

    def getDeltaW(self, gradients):
        deltaW = []
        for nuron, gradient in zip(self.nurons, gradients):
            deltaW.append(self.epsilon*gradient*nuron.getNuronVal())
        return deltaW
    
    def updateWeights(self,deltaWVector):
        for index , (deltaW , weights) in enumerate(zip(deltaWVector,self.inBoundWeightsVectors)):
            self.inBoundWeightsVectors[index] = [x + deltaW for x in weights]

    def getOutput(self, inputVector):
        try:
            output = []
            for nuron in self.nurons:
                output.append(nuron.getOutputValue(inputVector))
            return output
        except:
            print("Failed to get output")

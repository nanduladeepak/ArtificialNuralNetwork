from nuron import Nuron

class Layer:

    def __init__(self, noOfNurons: int, inBoundWeightsVectors):
        self.noOfNurons = noOfNurons
        self.nurons = []
        self.inBoundWeightsVectors = inBoundWeightsVectors

    def assignNurons(self):
        for index in range(0, self.noOfNurons):
            self.nurons.append(Nuron(self.inBoundWeightsVectors[index]))

    def getOutput(self, inputVector):
        try:
            output = []
            for nuron in self.nurons:
                output.append(nuron.getOutputValue(inputVector))
            return output
        except:
            print("Failed to get output")

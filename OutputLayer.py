from AbstractNeuralLayer import AbstractNeuralLayer
from SigmoidNeuron import SigmoidNeuron


class OutputLayer(AbstractNeuralLayer):
    def __init__(self,outputClasses):
        super().__init__()
        self.initializeWeights = False
        self.buildRandomLayer(outputClasses)
        self.setLearningRate(0.01)

    def backPropagation(self, expected_output):
        index = 0
        for neuron in self.neuron_array:
            error = expected_output[index] - neuron.output
            neuron.delta = error * self.transferDerivative(neuron.output)
            index += 1
        self.previous_layer.backPropagation(expected_output)

    def setWeights(self,numWeights):
        self.setRandomWeights(numWeights,-2,2)
        self.initializeWeights = True

    def forwardPropagation(self, inputs):
        if (not self.initializeWeights):
            self.setWeights(len(inputs))

        self.outputs = []
        for neuron in self.neuron_array:
            neuron.output = neuron.getOutput(inputs)
            self.outputs.append(neuron.output)

    def applyPropagationChanges(self, inputs):
        for neuron in self.neuron_array:
            neuron.updateWeights(inputs)
            neuron.updateBias()

    def setLearningRate(self, learning_rate):
        for neuron in self.neuron_array:
            neuron.setC(learning_rate)

    def getOutput(self):
        return self.outputs
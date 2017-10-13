from AbstractNeuralLayer import AbstractNeuralLayer
from SigmoidNeuron import SigmoidNeuron


class OutputLayer(AbstractNeuralLayer):
    def __init__(self,outputClasses):
        super().__init__()
        self.initializeWeights = False
        self.buildRandomLayer(outputClasses)
        self.setLearningRate(0.05)

    def backPropagation(self, expected_output):
        for index in range(len(self.neuron_array)):
            error = expected_output[index] - self.neuron_array[index].output
            self.neuron_array[index].delta = error * self.transferDerivative(self.neuron_array[index].output)
        self.previous_layer.backPropagation(expected_output)

    def setWeights(self,numWeights):
        self.setRandomWeights(numWeights,-1,2)
        self.initializeWeights = True

    def forwardPropagation(self, inputs):
        if (not self.initializeWeights):
            self.setWeights(len(inputs))

        self.outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(inputs)
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
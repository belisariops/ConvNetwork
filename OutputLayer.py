from AbstractNeuralLayer import AbstractNeuralLayer


class OutputLayer(AbstractNeuralLayer):
    def backPropagation(self, expected_output):
        index = 0
        for neuron in self.neuron_array:
            error = expected_output[index] - neuron.output
            neuron.delta = error * self.transferDerivative(neuron.output)
            index += 1
        self.previous_layer.backPropagation(expected_output)

    def forwardPropagation(self, inputs):
        self.outputs = []
        for neuron in self.neuron_array:
            neuron.output = neuron.getOutput(inputs)
            self.outputs.append(neuron.output)

    def applyChanges(self, inputs):
        for neuron in self.neuron_array:
            neuron.updateWeights(inputs)
            neuron.updateBias()

    def setLearningRate(self, learning_rate):
        for neuron in self.neuron_array:
            neuron.setC(learning_rate)

    def getOutput(self):
        return self.outputs
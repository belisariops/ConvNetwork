from abc import ABC, abstractmethod

from SigmoidNeuron import SigmoidNeuron


class AbstractNeuralLayer(ABC):
    def __init__(self, neuron_array=None):
        self.neuron_array = neuron_array
        if neuron_array is None:
            self.neuron_array = []
        self.next_layer = None
        self.previous_layer = None
        self.outputs = []

    def buildRandomLayer(self, number_of_neurons):
        neuron = None

        for i in range(number_of_neurons):
            neuron = SigmoidNeuron()
            neuron.setRandomParameters()
            self.neuron_array.append(neuron)

    def setLearningRate(self,learning_rate):
        for neuron in self.neuron_array:
            neuron.setC(learning_rate)
        self.next_layer.setLearningRate(learning_rate)

    def applyPropagationChanges(self,inputs):
        outputs = []
        for neuron in self.neuron_array:
            neuron.updateWeights(inputs)
            neuron.updateBias()
            outputs.append(neuron.output)
        self.next_layer.applyPropagationChanges(outputs)


    def transferDerivative(self,output):
        return output*(1.0 - output)



    def setPreviousLayer(self, previous_layer):
        self.previous_layer = previous_layer

    def setNextLayer(self, next_layer):
        self.next_layer = next_layer

    def getNumberofNeurons(self):
        return len(self.neuron_array)


    def getOutputs(self,inputs):
        pass

    def connect(self,otherLayer):
        self.next_layer = otherLayer
        otherLayer.previous_layer = self

    def setRandomWeights(self,number_of_weights,min_value,max_value):
        for neuron in self.neuron_array:
            neuron.setRandomWeights(number_of_weights,min_value,max_value)

    def calculateDelta(self,expected_output):
        for index in range(len(self.neuron_array)):
            error = 0
            for next_neuron in self.next_layer.neuron_array:
                error += next_neuron.weights[index]*next_neuron.delta
            self.neuron_array[index].delta = error * self.transferDerivative(self.neuron_array[index].output)




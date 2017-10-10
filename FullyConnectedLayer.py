from AbstractNeuralLayer import AbstractNeuralLayer
import numpy as np

class FullyConnectedLayer(AbstractNeuralLayer):
    def __init__(self):
        super().__init__()
        self.isBuilded = False

    def buildNeurons(self,numNeurons):
        self.buildRandomLayer(numNeurons)
        self.setLearningRate(0.01)
        self.isBuilded = True
        self.setRandomWeights(numNeurons,-3,3)

    def forwardPropagation(self,input):
        if (not self.isBuilded):
            self.buildNeurons(len(input))
        self.outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(input)
            neuron.output = neuron.getOutput(input)
            self.outputs.append(neuron.output)
        return self.next_layer.forwardPropagation(self.outputs)

    def backPropagation(self,expected_output):
        # # Revisar caso en que la imagen es de un canal (blanco y negro)
        # channels = len(self.previous_layer.deltas)
        # count = 0
        # for channel in range(channels):
        #     deltasHeight = self.previous_layer.deltas[channel].shape[0]
        #     deltasWidth = self.previous_layer.deltas[channel].shape[1]
        #     for h in range(deltasWidth):
        #         for w in range(deltasWidth):
        #             self.previous_layer.deltas[channels][h][w] = self.neuron_array[count].delta
        #             count += 1
        self.calculateDelta(expected_output)
        self.previous_layer.backPropagation(expected_output)
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

    def forwardPropagation(self,input):
        if (not self.isBuilded):
            self.buildNeurons(len(input))
        outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(input)
            neuron.output = neuron.getOutput(input)
            outputs.append(neuron.output)
        return self.next_layer.forwardPropagation(outputs)

    def backPropagation(self):
        # Revisar caso en que la imagen es de un canal (blanco y negro)
        channels = len(self.previous_layer.deltas)
        count = 0
        for channel in range(channels):
            deltasHeight = self.previous_layer.deltas[channel].shape[0]
            deltasWidth = self.previous_layer.deltas[channel].shape[1]
            for h in range(deltasWidth):
                for w in range(deltasWidth):
                    self.previous_layer.deltas[channels][h][w] = self.neuron_array[count].delta
                    count += 1

        self.previous_layer.backPropagation()
from AbstractNeuralLayer import AbstractNeuralLayer
from InnerNeuralLayer import InnerNeuralLayer
import numpy as np

class FlattenLayer(AbstractNeuralLayer):
    def __init__(self):
        super().__init__()
        self.isBuilded = False

    def buildNeurons(self,numNeurons):
        self.buildRandomLayer(numNeurons)
        self.setLearningRate(0.01)
        self.isBuilded = True

    def forwardPropagation(self,input):
        # Aplanar el input primero y usar una capa de perceptron multiple
        myInput = input.flatten()
        # Revisar caso en que la imagen es de un canal (blanco y negro)
        try:
            channels = input.shape[2]
        except IndexError:
            channels = 1
        inputHeight = input.shape[0]
        inputWidth = input.shape[1]
        self.deltas = []
        for channel in range(channels):
            self.deltas.append(np.zeros(inputHeight,inputWidth))

        if (not self.isBuilded):
            self.buildNeurons(len(myInput))
        outputs = []
        for neuron in self.neuron_array:
            neuron.setInputsList(myInput)
            neuron.output = neuron.getOutput(myInput)
            outputs.append(neuron.output)
        return self.next_layer.forwardPropagation(outputs)

    def backPropagation(self):
        self.previous_layer.deltas = self.deltas

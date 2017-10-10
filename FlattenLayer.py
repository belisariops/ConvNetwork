from AbstractNeuralLayer import AbstractNeuralLayer
from InnerNeuralLayer import InnerNeuralLayer
import numpy as np

class FlattenLayer(AbstractNeuralLayer):
    def __init__(self):
        super().__init__()
        self.isBuilded = False
        self.thisShape = None
        self.deltas = []
        self.outputs = []


    def buildNeurons(self,numNeurons):
        self.buildRandomLayer(numNeurons)
        self.setLearningRate(0.01)
        self.isBuilded = True
        self.setRandomWeights(1,-2,2)

    def forwardPropagation(self,input):
        self.thisShape = input.shape
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
            self.deltas.append(np.zeros((inputHeight,inputWidth)))

        if (not self.isBuilded):
            self.buildNeurons(len(myInput))
        self.outputs = []
        for index,neuron in enumerate(self.neuron_array):
            neuron.setInputsList(myInput[index])
            neuron.output = neuron.getOutput([myInput[index]])
            self.outputs.append(neuron.output)
        return self.next_layer.forwardPropagation(self.outputs)

    def backPropagation(self,expected_output):
        self.calculateDelta(expected_output)
        self.deltas = np.zeros(len(self.neuron_array))
        for index,neuron in enumerate(self.neuron_array):
            self.deltas[index] = neuron.delta
        self.previousLayer.deltas = np.reshape(self.deltas,self.thisShape)
        self.previousLayer.backPropagation()


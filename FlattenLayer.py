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
        self.myInput = None


    def buildNeurons(self,numNeurons):
        self.buildRandomLayer(numNeurons)
        # self.setLearningRate(0.01)
        self.isBuilded = True
        self.setRandomWeights(1,-2,2)

    def forwardPropagation(self,input):
        self.thisShape = input.shape
        # Aplanar el input primero y usar una capa de perceptron multiple
        self.myInput = input.flatten()
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
            self.buildNeurons(len(self.myInput))
        self.outputs = []
        for index,neuron in enumerate(self.neuron_array):
            neuron.setInputsList(self.myInput[index])
            neuron.output = neuron.getOutput([self.myInput[index]])
            self.outputs.append(neuron.output)
        return self.next_layer.forwardPropagation(self.outputs)

    def backPropagation(self,expected_output):
        self.calculateDelta(expected_output)
        self.deltas = []
        size = self.thisShape[0]*self.thisShape[1]
        self.kernel = []
        aux_deltas = np.zeros(size)
        for index,neuron in enumerate(self.neuron_array):
            aux_deltas[index % size] = neuron.delta
            if (index + 1) % size == 0:
                local_shape = (self.thisShape[0],self.thisShape[1])
                self.deltas.append(np.reshape(aux_deltas,local_shape))
                aux_deltas = np.zeros(size)


        self.previousLayer.deltas = self.deltas
        self.previousLayer.backPropagation()

    def applyPropagationChanges(self,inputs):
        outputs = []
        for index,neuron in enumerate(self.neuron_array):
            neuron.updateWeightEscalarInput(self.myInput[index])
            neuron.updateBias()
            outputs.append(neuron.output)
        self.next_layer.applyPropagationChanges(outputs)


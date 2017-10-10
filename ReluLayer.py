from NeuralLayer import NeuralLayer
import numpy as np

class ReluLayer(NeuralLayer):
    def forwardPropagation(self,input):
        # Revisar caso en que la imagen es de un canal (blanco y negro)
        try:
            channels = input.shape[2]
        except IndexError:
            channels = 1
        inputHeigth = input.shape[0]
        inputWidth = input.shape[1]
        self.outputFeatureMap = input.clip(min=0)
        self.deltas = []
        for channel in range(channels):
            self.deltas.append(np.zeros(inputHeigth,inputWidth))
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def backPropagation(self):
        self.previousLayer.deltas = self.deltas


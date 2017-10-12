from NeuralLayer import NeuralLayer
import numpy as np


class ReluLayer(NeuralLayer):
    def forwardPropagation(self,input):
        # Revisar caso en que la imagen es de un canal (blanco y negro)
        input_heigth, input_width,channels = input.shape
        self.outputFeatureMap = input.clip(min=0)
        self.deltas = []
        for channel in range(channels):
            self.deltas.append(np.zeros((input_heigth,input_width)))
        self.nextLayer.forwardPropagation(self.outputFeatureMap)

    def backPropagation(self):
        self.previousLayer.deltas = self.deltas
        self.previousLayer.backPropagation()


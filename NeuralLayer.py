from abc import ABC, abstractmethod

class NeuralLayer(ABC):
    def __init__(self):
        self.outputFeatureMap = []
        self.kernels = []
        self.deltas = []
        self.inputFeatureMap = []
        self.nextLayer = None
        self.previousLayer = None

    @abstractmethod
    def forwardPropagation(self,input):
        pass

    def updateWeights(self,input):
        pass

    def applyPropagationChanges(self,input):
        self.updateWeights(input)
        self.nextLayer.applyPropagationChanges(input)

    @abstractmethod
    def backPropagation(self):
        pass

    def connect(self,layer):
        self.nextLayer = layer
        layer.previousLayer = self